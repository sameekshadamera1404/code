"""
engine.py
---------
RecommendationService wraps HybridRecommendationEngine with risk enrichment.

Change from original: the constructor now accepts an optional pre-built
`engine` argument so main.py can inject a persisted model instead of
always training a fresh one.
"""

from recommender import HybridRecommendationEngine


class RecommendationService:
    def __init__(self, df, engine: HybridRecommendationEngine | None = None):
        self.df = df

        # Accept an externally created (possibly loaded-from-disk) engine,
        # or fall back to training a new one if none is supplied.
        if engine is not None:
            self.engine = engine
        else:
            self.engine = HybridRecommendationEngine()
            self.engine.fit(df)

        # Precompute normalization factors from the *current* DB snapshot.
        # These are only used for risk enrichment; the ML predictions come
        # from the persisted model which may know more rows than df does.
        self.max_likelihood = df["likelihood"].max() or 1
        self.max_severity = df["severity"].max() or 1
        self.max_impact = df["impact_rating"].max() or 1

    # ── Risk helpers ────────────────────────────────────────────────────

    def _compute_threat_risk(self, threat_name, ml_score):
        threat_norm = str(threat_name).strip().lower()
        sub = self.df[self.df["threat_name_norm"] == threat_norm]

        if sub.empty:
            # Threat was deleted from DB but is still known to the ML model.
            # Return a sensible default rather than zeros.
            return 1.0, 1.0, ml_score / 100

        likelihood = sub["likelihood"].mean()
        severity = sub["severity"].mean()
        impact_rating = sub["impact_rating"].mean()

        raw_probability = likelihood / self.max_likelihood
        probability = 1 + 4 * raw_probability
        impact = (impact_rating / self.max_impact) * 5
        risk_score = probability * impact * (ml_score / 100)

        return probability, impact, risk_score

    def _compute_control_risk(self, control_name, ml_score):
        control_norm = str(control_name).strip().lower()
        sub = self.df[self.df["control_name_norm"] == control_norm]

        if sub.empty:
            return 1.0, 1.0, ml_score / 100

        likelihood = sub["likelihood"].mean()
        severity = sub["severity"].mean()
        impact_rating = sub["impact_rating"].mean()

        raw_probability = likelihood / self.max_likelihood
        probability = 1 + 4 * raw_probability
        impact = (impact_rating / self.max_impact) * 5
        risk_score = probability * impact * (ml_score / 100)

        return probability, impact, risk_score

    # ── Direct DB lookups ───────────────────────────────────────────────

    def _get_direct_controls(self, threats, top_n=10):
        direct_controls = []
        for threat in threats:
            threat_norm = str(threat).strip().lower()
            threat_data = self.df[self.df["threat_name_norm"] == threat_norm]
            if not threat_data.empty:
                controls = threat_data.sort_values("effectiveness", ascending=False)[
                    ["control_name", "effectiveness"]
                ].head(top_n)
                for _, row in controls.iterrows():
                    prob, impact, risk = self._compute_control_risk(
                        row["control_name"], row["effectiveness"]
                    )
                    direct_controls.append({
                        "name": row["control_name"],
                        "probability": round(prob, 2),
                        "impact": round(impact, 2),
                        "risk_score": round(risk, 2),
                    })
        seen, unique = set(), []
        for ctrl in direct_controls:
            if ctrl["name"] not in seen:
                seen.add(ctrl["name"])
                unique.append(ctrl)
        unique.sort(key=lambda x: x["risk_score"], reverse=True)
        return unique[:top_n]

    def _get_direct_threats(self, controls, top_n=10):
        direct_threats = []
        for control in controls:
            control_norm = str(control).strip().lower()
            control_data = self.df[self.df["control_name_norm"] == control_norm]
            if not control_data.empty:
                threats = control_data.sort_values("effectiveness", ascending=False)[
                    ["threat_name", "effectiveness"]
                ].head(top_n)
                for _, row in threats.iterrows():
                    prob, impact, risk = self._compute_threat_risk(
                        row["threat_name"], row["effectiveness"]
                    )
                    direct_threats.append({
                        "name": row["threat_name"],
                        "probability": round(prob, 2),
                        "impact": round(impact, 2),
                        "risk_score": round(risk, 2),
                    })
        seen, unique = set(), []
        for thr in direct_threats:
            if thr["name"] not in seen:
                seen.add(thr["name"])
                unique.append(thr)
        unique.sort(key=lambda x: x["risk_score"], reverse=True)
        return unique[:top_n]

    # ── Public API ──────────────────────────────────────────────────────

    def get_threats(self, controls, top_n=10):
        resolved_controls = self.engine.resolve_controls(controls)

        direct_results = self._get_direct_threats(resolved_controls, top_n)
        if direct_results:
            return direct_results

        results = self.engine.recommend_threats(controls, top_n)
        enriched = []
        for name, score in results:
            prob, impact, risk = self._compute_threat_risk(name, score)
            enriched.append({
                "name": name,
                "probability": round(prob, 2),
                "impact": round(impact, 2),
                "risk_score": round(risk, 2),
            })
        enriched.sort(key=lambda x: x["risk_score"], reverse=True)
        return enriched

    def get_controls(self, threats, top_n=10):
        resolved_threats = self.engine.resolve_threats(threats)

        direct_results = self._get_direct_controls(resolved_threats, top_n)
        if direct_results:
            return direct_results

        results = self.engine.recommend_controls(threats, top_n)
        enriched = []
        for name, score in results:
            prob, impact, risk = self._compute_control_risk(name, score)
            enriched.append({
                "name": name,
                "probability": round(prob, 2),
                "impact": round(impact, 2),
                "risk_score": round(risk, 2),
            })
        enriched.sort(key=lambda x: x["risk_score"], reverse=True)
        return enriched
