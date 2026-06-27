import json
from src.features import FeatureExtractor
from src.knn_model import DifficultyKNN

class DifficultyPredictor:
    def __init__(self, dataset_path):
        with open(dataset_path, "r") as f:
            data = json.load(f)

        self.texts = [d["text"] for d in data]
        self.labels = [d["label"] for d in data]

        self.extractor = FeatureExtractor()
        self.model = DifficultyKNN(k=5)

        X = self.extractor.fit_transform(self.texts)
        self.model.train(X, self.labels)

    def predict(self, text):
        X_test = self.extractor.transform([text])
        return self.model.predict(X_test)
