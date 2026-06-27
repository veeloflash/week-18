from src.predictor import DifficultyPredictor

predictor = DifficultyPredictor("app/data/difficulty_dataset.json")

while True:
    q = input("Enter question: ")
    print("Difficulty:", predictor.predict(q))
