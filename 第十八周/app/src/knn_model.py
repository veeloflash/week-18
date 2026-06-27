from sklearn.neighbors import KNeighborsClassifier

class DifficultyKNN:
    def __init__(self, k=5):
        self.k = k
        self.model = None

    def train(self, X, y):
        k = min(self.k, X.shape[0])
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)[0]

