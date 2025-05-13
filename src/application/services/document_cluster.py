from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

class DocumentClusterService:
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42)

    def cluster(self, text: str) -> Dict[int, List[str]]:
        # Split text into segments by double newlines
        segments = [seg.strip() for seg in text.split("\n\n") if len(seg.strip()) > 20]
        if not segments:
            return {}
        # Vectorize segments and cluster
        X = self.vectorizer.fit_transform(segments)
        labels = self.model.fit_predict(X)
        clusters: Dict[int, List[str]] = {}
        for label, segment in zip(labels, segments):
            clusters.setdefault(int(label), []).append(segment)
        return clusters
