import numpy as np
from difflib import get_close_matches
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

        self.latent = None   # threat latent factors
        self.control_latent = None

    #  Fit model using pandas df
    def fit(self, df):
        threat_col = "threat_name_norm" if "threat_name_norm" in df.columns else "threat_name"
        control_col = "control_name_norm" if "control_name_norm" in df.columns else "control_name"

        pivot = df.pivot_table(
            index=threat_col,
            columns=control_col,
            values="effectiveness",
            fill_value=0
        )

        self.matrix = pivot.values
        self.threats = list(pivot.index)
        self.controls = list(pivot.columns)

        self.threat_name_map = {
            norm_name: orig_name
            for orig_name, norm_name in zip(df["threat_name"], df[threat_col])
        }
        self.control_name_map = {
            norm_name: orig_name
            for orig_name, norm_name in zip(df["control_name"], df[control_col])
        }

        # ✅ train latent-factor recommender
        self.latent = self.svd.fit_transform(self.matrix)
        self.control_latent = self.svd.components_.T

        # ✅ train supervised interaction model
        feature_texts = (
            df[threat_col].astype(str).str.strip().str.lower()
            + " || "
            + df[control_col].astype(str).str.strip().str.lower()
        )
        features = self.vectorizer.fit_transform(feature_texts)
        labels = df["effectiveness"].fillna(0).astype(float).values
        self.regressor.fit(features, labels)

        # ✅ train name similarity vectors for fuzzy matching and misspellings
        self.name_vectorizer.fit(self.threats + self.controls)
        self.threat_name_matrix = self.name_vectorizer.transform(self.threats)
        self.control_name_matrix = self.name_vectorizer.transform(self.controls)

    # ✅ Get threat vector
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

    # ✅ Recommend Controls
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

        # ✅ similarity in latent space
        sim_scores = cosine_similarity(
            query_vec.reshape(1, -1),
            self.control_latent
        )[0]
        sim_scores = self._normalize(sim_scores)

        # ✅ combine both scores for better accuracy
        combined = pred_scores * 0.6 + sim_scores * 0.4

        ranked_idx = combined.argsort()[::-1][:top_n]

        return [
            (self._original_control_name(self.controls[i]), float(combined[i]))
            for i in ranked_idx
        ]

    # ✅ Recommend Threats from controls
    def recommend_threats(self, control_input, top_n=10):
        query_vec = self._get_control_vector(control_input)

        pred_scores = []
        for threat in self.threats:
            predictions = self._predict_effectiveness([threat], control_input)
            pred_scores.append(predictions[0])
        pred_scores = self._normalize(np.array(pred_scores, dtype=float))

        if query_vec is None:
            ranked_idx = pred_scores.argsort()[::-1][:top_n]
            return [
                (self._original_threat_name(self.threats[i]), float(pred_scores[i]))
                for i in ranked_idx
            ]

        # ✅ similarity in latent space
        sim_scores = cosine_similarity(
            query_vec.reshape(1, -1),
            self.latent
        )[0]
        sim_scores = self._normalize(sim_scores)

        combined = pred_scores * 0.6 + sim_scores * 0.4

        ranked_idx = combined.argsort()[::-1][:top_n]

        return [
            (self._original_threat_name(self.threats[i]), float(combined[i]))
            for i in ranked_idx
        ]

    def _original_threat_name(self, name_norm):
        return self.threat_name_map.get(name_norm, name_norm)

    def _original_control_name(self, name_norm):
        return self.control_name_map.get(name_norm, name_norm)

    # ✅ Normalize scores to 0–100
    def _normalize(self, scores):
        min_s = np.min(scores)
        max_s = np.max(scores)

        if max_s == min_s:
            return np.ones_like(scores) * 50

        norm = (scores - min_s) / (max_s - min_s)

        return norm * 100