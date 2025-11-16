# train_multi_models.py
"""
Train 3 models on combined_dataset.csv and save pipelines in saved_models/
"""
import os, json, re
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_CSV = "combined_dataset.csv"
OUT_DIR = "saved_models"
RANDOM_STATE = 42
TEST_SIZE = 0.15
N_FOLDS = 5

os.makedirs(OUT_DIR, exist_ok=True)

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#', ' ', text)
    # conservative cleaning to keep expressive tokens like 'baddd'
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # reduce elongation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"{DATA_CSV} not found. Run merge_and_prepare.py first.")
    df = pd.read_csv(DATA_CSV)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("combined_dataset.csv must contain 'text' and 'label' columns.")
    df = df.dropna(subset=['text','label']).reset_index(drop=True)
    df['text_clean'] = df['text'].apply(clean_text)
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label_id'] = df['label'].map(label_map)
    return df, label_map

def build_pipelines():
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    pipelines = {
        'logreg': Pipeline([('tfidf', tfidf), ('clf', LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced'))]),
        'nb': Pipeline([('tfidf', tfidf), ('clf', MultinomialNB())]),
        'rf': Pipeline([('tfidf', tfidf), ('clf', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced'))])
    }
    return pipelines

def train_and_save():
    df, label_map = load_data()
    X = df['text_clean']
    y = df['label_id']
    print("Total samples:", len(df))
    print("Label distribution:\n", df['label'].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    pipelines = build_pipelines()
    summary = {}
    for name, pipe in pipelines.items():
        print("\n--- Training:", name, "---")
        try:
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='f1_weighted', n_jobs=-1)
            print("CV f1-weighted mean/std:", cv_scores.mean(), cv_scores.std())
        except Exception as e:
            print("CV skipped:", e)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("Test acc:", acc)
        print("Classification report:\n", classification_report(y_test, y_pred))
        fname = os.path.join(OUT_DIR, f"{name}_pipeline.joblib")
        joblib.dump({'pipeline': pipe, 'label_map': label_map}, fname)
        summary[name] = {'test_accuracy': float(acc), 'file': fname}
        print("Saved", fname)

    with open(os.path.join(OUT_DIR, 'models_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print("\nSaved models to", OUT_DIR)

if __name__ == "__main__":
    train_and_save()
