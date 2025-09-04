import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Resolve project root (two levels up from this file)
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "spam.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"

df = pd.read_csv(DATA_PATH, encoding="utf-8")

X = df["Message"]
y = df["Category"]

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# build a text classification pipeline: TF-IDF -> Logistic Regression
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("logreg", LogisticRegression(max_iter=1000)),
])

# train the model pipeline
pipeline.fit(X_train, y_train)

# ensure models directory exists and save the model
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
# D:\Documents\HETIC\BI\
# data\spam.csv

