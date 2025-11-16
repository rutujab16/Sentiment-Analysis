Sentiment Analysis Using Machine Learning
🎯 Project Objective

To build a Machine Learning–based sentiment analysis system that classifies text into Positive, Negative, and Neutral sentiments.
This project includes text preprocessing, dataset merging, model training, evaluation, and a simple interface for testing predictions.

📂 Dataset Details

Final prepared dataset: combined_dataset.csv

Total samples: 31515

Label distribution:

neutral: 12746

positive: 9851

negative: 8918

Preprocessing steps:

Lowercasing text

Removing stopwords

Removing special characters & punctuation

(Optional) Lemmatization

Duplicate removal

Merging datasets inside the datasets/ folder

🧠 Algorithms / Models Used

The following Machine Learning algorithms were trained:

Logistic Regression (logreg_pipeline.joblib)

Naive Bayes (nb_pipeline.joblib)

Random Forest (rf_pipeline.joblib)

Vectorization Method

TF–IDF Vectorizer used to convert text into numerical features.

📊 Results (Accuracy & Metrics)
Cross-validation (f1-weighted mean ± std)

Logistic Regression: 0.6978 ± 0.0063

Naive Bayes: 0.6316 ± 0.0068

Random Forest: 0.6896 ± 0.0015

Test Accuracy

Logistic Regression: 0.7100

Naive Bayes: 0.6571

Random Forest: 0.7045

Short classification report summaries (test set)

Logistic Regression (test)

Accuracy: 0.7100

Weighted F1 ~ 0.71

Naive Bayes (test)

Accuracy: 0.6571

Weighted F1 ~ 0.65

Random Forest (test)

Accuracy: 0.7045

Weighted F1 ~ 0.70

Full classification reports (precision/recall/f1/support per class) are printed during training and saved in console logs.

🗂 Saved Models

Trained models were saved to the saved_models/ folder:

saved_models/logreg_pipeline.joblib

saved_models/nb_pipeline.joblib

saved_models/rf_pipeline.joblib

Note: Some saved model files (e.g., Random Forest) can be large. If you plan to push to GitHub, add saved_models/ to .gitignore or use Git LFS for large files.

📝 Conclusion

Traditional Machine Learning models with proper preprocessing achieve good performance on this dataset. Logistic Regression achieved the highest test accuracy (71.00%), closely followed by Random Forest (70.45%). Naive Bayes performed slightly lower (65.71%) but remains a lightweight and fast baseline.

🚀 Future Scope

Add a separate Neutral handling strategy (if needed for downstream tasks) or re-balance classes.

Perform hyperparameter tuning (Grid Search / Random Search) for each model.

Try advanced text models (transformers like BERT) for potential accuracy gains.

Deploy the best model through Flask on a cloud service (Heroku / Render / AWS).

Provide downloadable model files via GitHub Releases or cloud storage (instead of tracking binaries in repo).

📚 References

Scikit-learn Documentation

NLTK Documentation

Any dataset sources listed in the datasets/ folder

---

## 🙌 Author
Developed as a complete sentiment analysis pipeline for academic and experimental usage.

