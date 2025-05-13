import os
import requests

class APIClient:
    """
    Simple REST client for ExamCollector API.
    Base URL can be set via EXAMCOLLECTOR_API_URL env var; defaults to localhost:8000
    """
    def __init__(self):
        self.base_url = os.getenv("EXAMCOLLECTOR_API_URL", "http://localhost:8000")

    def convert(self, file_path: str) -> list:
        """Convert a document file to base64-encoded images."""
        url = f"{self.base_url}/exams/upload/convert"
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
            r = requests.post(url, files=files)
        r.raise_for_status()
        return r.json()

    def label(self, file_path: str) -> list:
        """Detect and label objects in an image file."""
        url = f"{self.base_url}/exams/upload/label"
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'image/jpeg')}
            r = requests.post(url, files=files)
        r.raise_for_status()
        return r.json()

    def tag(self, file_path: str, agent: str = "default", retry: bool = False) -> dict:
        """Cluster and tag exam questions from a document file."""
        url = f"{self.base_url}/exams/upload/tag"
        params = {"agent": agent, "retry": str(retry).lower()}
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
            r = requests.post(url, params=params, files=files)
        r.raise_for_status()
        return r.json()
