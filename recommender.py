"""
recommender.py
--------------
HybridRecommendationEngine — unchanged public API, plus one new method:

    engine.merge_fit(new_df)

merge_fit() merges new_df with the data the model was originally trained on,
then re-fits.  This means:
  • Mappings that existed at previous training time are still in the matrix
    even if they were deleted from the DB afterwards.
  • New rows from new_df extend the model's knowledge.
"""

import numpy as np
from difflib import get_close_matches

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommendationEngine:
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components)
        self.regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.name_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))

        self.matrix = None
        self.threats = []
        self.controls = []
        self.threat_name_map = {}
        self.control_name_map = {}

        self.threat_name_matrix = None
        self.control_name_matrix = None

        self.latent = None
        self.control_latent = None

        # ── NEW: keep a snapshot of every row ever seen ──────────────────
        # This is what lets the model "remember" deleted DB rows.
        self._training_snapshot: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_name(self, name):
        return str(name).strip().lower()

    def _find_best_match(self, query, candidates, cutoff=0.5):
        if not candidates:
            return None
        query_vec = self.name_vectorizer.transform([query])
        candidate_vec = self.name_vectorizer.transform(candidates)
        sim_scores = cosine_similarity(query_vec, candidate_vec)[0]
        best_idx = int(sim_scores.argmax())
        if sim_scores[best_idx] >= cutoff:
            return candidates[best_idx]
        return None

    def _resolve_items(self, items, reference_names):
        resolved = []
        for item in items:
            norm_name = self._normalize_name(item)
            if norm_name in reference_names:
                resolved.append(norm_name)
                continue
            close = get_close_matches(norm_name, reference_names, n=1, cutoff=0.7)
            if close:
                resolved.append(close[0])
                continue
            fuzzy = self._find_best_match(norm_name, reference_names, cutoff=0.5)
            if fuzzy:
                resolved.append(fuzzy)
        return list(dict.fromkeys(resolved))

    def resolve_threats(self, threats):
        return self._resolve_items(threats, self.threats)

    def resolve_controls(self, controls):
        return self._resolve_items(controls, self.controls)

    def _get_threat_vector(self, threat_input):
        indices = []
        for t in threat_input:
            t_norm = self._normalize_name(t)
            if t_norm in self.threats:
                indices.append(self.threats.index(t_norm))
                continue
            fuzzy = self._find_best_match(t_norm, self.threats, cutoff=0.5)
            if fuzzy:
                indices.append(self.threats.index(fuzzy))
        if not indices:
            return None
        return np.mean(self.latent[indices], axis=0)

    def _get_control_vector(self, control_input):
        indices = []
        for c in control_input:
            c_norm = self._normalize_name(c)
            if c_norm in self.controls:
                indices.append(self.controls.index(c_norm))
                continue
            fuzzy = self._find_best_match(c_norm, self.controls, cutoff=0.5)
            if fuzzy:
                indices.append(self.controls.index(fuzzy))
        if not indices:
            return None
        return np.mean(self.control_latent[indices], axis=0)

    def _prepare_interaction_texts(self, threats, controls):
        texts = []
        for threat in threats:
            threat_norm = self._normalize_name(threat)
            for control in controls:
                control_norm = self._normalize_name(control)
                texts.append(f"{threat_norm} || {control_norm}")
        return texts

    def _predict_effectiveness(self, threats, controls):
        if not threats or not controls:
            return np.zeros((len(controls),), dtype=float)
        texts = self._prepare_interaction_texts(threats, controls)
        features = self.vectorizer.transform(texts)
        predictions = self.regressor.predict(features)
        predictions = predictions.reshape(len(threats), len(controls))
        return np.mean(predictions, axis=0)

    def _normalize(self, scores):
        min_s, max_s = np.min(scores), np.max(scores)
        if max_s == min_s:
            return np.ones_like(scores) * 50
        return (scores - min_s) / (max_s - min_s) * 100

    # ------------------------------------------------------------------
    # Core fit (trains on whatever DataFrame is passed in)
    # ------------------------------------------------------------------

    def _fit_internal(self, df: pd.DataFrame):
        """Fit all sub-models on df and cache the snapshot."""
        threat_col = "threat_name_norm" if "threat_name_norm" in df.columns else "threat_name"
        control_col = "control_name_norm" if "control_name_norm" in df.columns else "control_name"

        pivot = df.pivot_table(
            index=threat_col,
            columns=control_col,
            values="effectiveness",
            aggfunc="mean",       # handles duplicate (threat, control) pairs
            fill_value=0,
        )

        self.matrix = pivot.values
        self.threats = list(pivot.index)
        self.controls = list(pivot.columns)

        self.threat_name_map = {
            norm: orig
            for orig, norm in zip(df["threat_name"], df[threat_col])
        }
        self.control_name_map = {
            norm: orig
            for orig, norm in zip(df["control_name"], df[control_col])
        }

        # Latent-factor model
        n_comp = min(self.n_components, min(self.matrix.shape) - 1)
        n_comp = max(n_comp, 1)
        self.svd = TruncatedSVD(n_components=n_comp)
        self.latent = self.svd.fit_transform(self.matrix)
        self.control_latent = self.svd.components_.T

        # Supervised interaction model
        feature_texts = (
            df[threat_col].astype(str).str.strip().str.lower()
            + " || "
            + df[control_col].astype(str).str.strip().str.lower()
        )
        features = self.vectorizer.fit_transform(feature_texts)
        labels = df["effectiveness"].fillna(0).astype(float).values
        self.regressor.fit(features, labels)

        # Name-similarity vectors for fuzzy matching
        self.name_vectorizer.fit(self.threats + self.controls)
        self.threat_name_matrix = self.name_vectorizer.transform(self.threats)
        self.control_name_matrix = self.name_vectorizer.transform(self.controls)

    # ------------------------------------------------------------------
    # Public: initial fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame):
        """
        Train the model from scratch on df.
        Saves a snapshot so future merge_fit() calls can accumulate knowledge.
        """
        self._training_snapshot = df.copy()
        self._fit_internal(df)

    # ------------------------------------------------------------------
    # Public: incremental / merge fit  ← KEY NEW METHOD
    # ------------------------------------------------------------------

    def merge_fit(self, new_df: pd.DataFrame):
        """
        Merge new_df with the historical snapshot and re-fit.

        This means:
          • Every threat-control pair the model has *ever* seen is preserved
            even if it no longer exists in the live DB.
          • New pairs from new_df are added.
          • Where both snapshots have a row for the same (threat, control),
            the new_df value wins (higher weight via duplication).
        """
        if self._training_snapshot is None:
            # No prior snapshot → behave like a plain fit
            self.fit(new_df)
            return

        # ── Merge: old snapshot + new data ──────────────────────────────
        # Give new data 2× weight so it steers the model toward current truth
        # while old data is still influential.
        merged = pd.concat(
            [self._training_snapshot, new_df, new_df],   # new_df appears twice
            ignore_index=True,
        )

        # Keep the snapshot up-to-date (union of all rows ever seen)
        key_cols = ["threat_name_norm", "control_name_norm"]
        combined_snapshot = pd.concat(
            [self._training_snapshot, new_df], ignore_index=True
        ).drop_duplicates(subset=key_cols, keep="last")
        self._training_snapshot = combined_snapshot

        self._fit_internal(merged)

    # ------------------------------------------------------------------
    # Recommend Controls
    # ------------------------------------------------------------------

    def recommend_controls(self, threat_input, top_n=10):
        query_vec = self._get_threat_vector(threat_input)
        pred_scores = self._predict_effectiveness(threat_input, self.controls)
        pred_scores = self._normalize(pred_scores)

        if query_vec is None:
            ranked_idx = pred_scores.argsort()[::-1][:top_n]
            return [
                (self._original_control_name(self.controls[i]), float(pred_scores[i]))
                for i in ranked_idx
            ]

        sim_scores = cosine_similarity(
            query_vec.reshape(1, -1), self.control_latent
        )[0]
        sim_scores = self._normalize(sim_scores)
        combined = pred_scores * 0.6 + sim_scores * 0.4

        ranked_idx = combined.argsort()[::-1][:top_n]
        return [
            (self._original_control_name(self.controls[i]), float(combined[i]))
            for i in ranked_idx
        ]

    # ------------------------------------------------------------------
    # Recommend Threats
    # ------------------------------------------------------------------

    def recommend_threats(self, control_input, top_n=10):
        query_vec = self._get_control_vector(control_input)

        pred_scores = np.array(
            [self._predict_effectiveness([t], control_input)[0] for t in self.threats],
            dtype=float,
        )
        pred_scores = self._normalize(pred_scores)

        if query_vec is None:
            ranked_idx = pred_scores.argsort()[::-1][:top_n]
            return [
                (self._original_threat_name(self.threats[i]), float(pred_scores[i]))
                for i in ranked_idx
            ]

        sim_scores = cosine_similarity(
            query_vec.reshape(1, -1), self.latent
        )[0]
        sim_scores = self._normalize(sim_scores)
        combined = pred_scores * 0.6 + sim_scores * 0.4

        ranked_idx = combined.argsort()[::-1][:top_n]
        return [
            (self._original_threat_name(self.threats[i]), float(combined[i]))
            for i in ranked_idx
        ]

    # ------------------------------------------------------------------
    # Name helpers
    # ------------------------------------------------------------------

    def _original_threat_name(self, name_norm):
        return self.threat_name_map.get(name_norm, name_norm)

    def _original_control_name(self, name_norm):
        return self.control_name_map.get(name_norm, name_norm)
