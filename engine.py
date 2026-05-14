from recommender import HybridRecommendationEngine
class RecommendationService:
    def __init__(self, df):
        self.df = df

        # initialize hybrid ML engine
        self.engine = HybridRecommendationEngine()
        self.engine.fit(df)

        # ✅ precompute normalization factors
        self.max_likelihood = df["likelihood"].max() or 1
        self.max_severity = df["severity"].max() or 1
        self.max_impact = df["impact_rating"].max() or 1

    # ✅ compute risk metrics for a threat
    def _compute_threat_risk(self, threat_name, ml_score):
        threat_norm = str(threat_name).strip().lower()
        sub = self.df[self.df["threat_name_norm"] == threat_norm]

        if sub.empty:
            return 0.0, 0.0, 0.0

        likelihood = sub["likelihood"].mean()
        severity = sub["severity"].mean()
        impact_rating = sub["impact_rating"].mean()

        # ✅ normalize values
        raw_probability = likelihood / self.max_likelihood
        probability = 1 + 4 * raw_probability
        severity_norm = severity / self.max_severity
        impact_norm = impact_rating / self.max_impact

        # ✅ combined impact on a 0–5 scale
        impact = impact_norm * 5

        # ✅ final risk (ML + business logic)
        risk_score = probability * impact * (ml_score / 100)

        return probability, impact, risk_score

    # ✅ compute risk metrics for a control
    def _compute_control_risk(self, control_name, ml_score):
        control_norm = str(control_name).strip().lower()
        sub = self.df[self.df["control_name_norm"] == control_norm]

        if sub.empty:
            return 0.0, 0.0, 0.0

        likelihood = sub["likelihood"].mean()
        severity = sub["severity"].mean()
        impact_rating = sub["impact_rating"].mean()

        raw_probability = likelihood / self.max_likelihood
        probability = 1 + 4 * raw_probability
        severity_norm = severity / self.max_severity
        impact_norm = impact_rating / self.max_impact

        impact = impact_norm * 5
        risk_score = probability * impact * (ml_score / 100)
        return probability, impact, risk_score

    # ✅ get direct controls from DB for given threats
    def _get_direct_controls(self, threats, top_n=10):
        direct_controls = []
        for threat in threats:
            threat_norm = str(threat).strip().lower()
            # Filter df for this threat
            threat_data = self.df[self.df["threat_name_norm"] == threat_norm]
            if not threat_data.empty:
                # Get controls mapped to this threat, sorted by effectiveness
                controls = threat_data.sort_values("effectiveness", ascending=False)[["control_name", "effectiveness"]].head(top_n)
                for _, row in controls.iterrows():
                    prob, impact, risk = self._compute_control_risk(row["control_name"], row["effectiveness"])
                    direct_controls.append({
                        "name": row["control_name"],
                        "probability": round(prob, 2),
                        "impact": round(impact, 2),
                        "risk_score": round(risk, 2)
                    })
        # Remove duplicates and sort by risk_score
        seen = set()
        unique_controls = []
        for ctrl in direct_controls:
            if ctrl["name"] not in seen:
                seen.add(ctrl["name"])
                unique_controls.append(ctrl)
        unique_controls.sort(key=lambda x: x["risk_score"], reverse=True)
        return unique_controls[:top_n] if unique_controls else []

    # ✅ get direct threats from DB for given controls
    def _get_direct_threats(self, controls, top_n=10):
        direct_threats = []
        for control in controls:
            control_norm = str(control).strip().lower()
            # Filter df for this control
            control_data = self.df[self.df["control_name_norm"] == control_norm]
            if not control_data.empty:
                # Get threats mapped to this control, sorted by effectiveness
                threats = control_data.sort_values("effectiveness", ascending=False)[["threat_name", "effectiveness"]].head(top_n)
                for _, row in threats.iterrows():
                    prob, impact, risk = self._compute_threat_risk(row["threat_name"], row["effectiveness"])
                    direct_threats.append({
                        "name": row["threat_name"],
                        "probability": round(prob, 2),
                        "impact": round(impact, 2),
                        "risk_score": round(risk, 2)
                    })
        # Remove duplicates and sort by risk_score
        seen = set()
        unique_threats = []
        for thr in direct_threats:
            if thr["name"] not in seen:
                seen.add(thr["name"])
                unique_threats.append(thr)
        unique_threats.sort(key=lambda x: x["risk_score"], reverse=True)
        return unique_threats[:top_n] if unique_threats else []

    # ✅ get threats from controls
    def get_threats(self, controls, top_n=10):
        resolved_controls = self.engine.resolve_controls(controls)

        # First, try to get direct mappings from DB
        direct_results = self._get_direct_threats(resolved_controls, top_n)
        if direct_results:
            return direct_results

        # If no direct mappings, fall back to ML prediction
        results = self.engine.recommend_threats(controls, top_n)

        enriched = []

        for name, score in results:
            prob, impact, risk = self._compute_threat_risk(name, score)

            enriched.append({
                "name": name,
                "probability": round(prob, 2),
                "impact": round(impact, 2),
                "risk_score": round(risk, 2)
            })

        # ✅ sort again by risk_score (important)
        enriched.sort(key=lambda x: x["risk_score"], reverse=True)

        return enriched

    # ✅ get controls from threats
    def get_controls(self, threats, top_n=10):
        resolved_threats = self.engine.resolve_threats(threats)

        # First, try to get direct mappings from DB
        direct_results = self._get_direct_controls(resolved_threats, top_n)
        if direct_results:
            return direct_results

        # If no direct mappings, fall back to ML prediction
        results = self.engine.recommend_controls(threats, top_n)

        enriched = []

        for name, score in results:
            prob, impact, risk = self._compute_control_risk(name, score)

            enriched.append({
                "name": name,
                "probability": round(prob, 2),
                "impact": round(impact, 2),
                "risk_score": round(risk, 2)
            })

        # ✅ sort again by risk_score
        enriched.sort(key=lambda x: x["risk_score"], reverse=True)

        return enriched
