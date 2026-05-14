from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.matrix = None
        self.items = []

    def fit(self, items):
        self.items = list(set(items))
        self.matrix = self.vectorizer.fit_transform(self.items)

    def find_similar(self, query_list, top_n=10):
        query_vec = self.vectorizer.transform(query_list)
        sim_scores = cosine_similarity(query_vec, self.matrix)

        avg_scores = sim_scores.mean(axis=0)
        idx = avg_scores.argsort()[::-1][:top_n]

        return [(self.items[i], float(avg_scores[i])) for i in idx]