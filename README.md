
# Sentiment Analysis Project – Version 3

A complete end‑to‑end Sentiment Analysis system built using Machine Learning and Deep Learning models, including dataset preprocessing, multimodel training, evaluation, and a Flask web interface for real‑time predictions.

---

## 📂 Project Structure

```
senti ana version 3/
│
├── app.py                        # Flask backend for the web UI
├── index.html                    # Frontend interface
├── background.jpg                # UI background image
│
├── train_multi_models.py         # Trains multiple ML/DL models
├── merge_and_prepare.py          # Merges + cleans raw datasets
├── test_predict.py               # Tests prediction using saved models
│
├── combined_dataset.csv          # Final cleaned dataset
├── datasets/                     # Raw dataset folder
│
├── saved_models/                 # Serialized trained ML/DL models
│
├── aiml accuracy output of two models.txt  # Model accuracy comparison
├── requirements.txt              # Dependencies
```

---

## 🚀 Features

### ✔ Dataset Preprocessing
- Merge multiple raw datasets  
- Clean text (stopwords, lowercase, lemmatization, etc.)  
- Remove duplicates + missing values  
- Output: `combined_dataset.csv`

### ✔ Model Training
Models trained in `train_multi_models.py` include:
- Logistic Regression  
- SVM  
- Naive Bayes  
- Random Forest  
- LSTM / GRU / Deep Learning models  

All saved into `saved_models/`.

### ✔ Evaluation
- Accuracy  
- Precision, Recall, F1  
- Confusion Matrix  
- Model comparison saved in `.txt` file

### ✔ Web Interface
- Clean HTML UI  
- Flask backend  
- Real‑time sentiment prediction  

---

## ⚙️ Installation

### 1️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 2️⃣ Preprocess datasets
```
python merge_and_prepare.py
```

### 3️⃣ Train models
```
python train_multi_models.py
```

### 4️⃣ Test predictions
```
python test_predict.py
```

---

## 🌐 Run the Web App

```
python app.py
```

Open in browser:

```
http://127.0.0.1:5000/
```

---

## 📊 Model Performance
The accuracy comparison is stored in:

```
aiml accuracy output of two models.txt
```

---

## 📁 Saved Models
Contains all trained models + vectorizers used by the app.

---

## 🧱 Tech Stack
- Python  
- Flask  
- Scikit‑learn  
- TensorFlow / Keras  
- NLTK  
- HTML / CSS  

---

## 🙌 Author
Developed as a complete sentiment analysis pipeline for academic and experimental usage.

