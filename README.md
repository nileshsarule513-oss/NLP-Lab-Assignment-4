# NLP Lab Assignment 4 - Text Preprocessing and Classification

**Course:** Machine Learning / Natural Language Processing  
**Assignment:** Lab Assignment 4 - Individual Submission  
**Title:** NLP Preprocessing and Text Classification  
**Author:** Nilesh  

---

## Objective

Implement NLP preprocessing techniques and build a text classification model using machine learning, covering the full pipeline from raw text to evaluated model output.

---

## Learning Outcomes

- Apply NLP preprocessing - tokenization, stopword removal, stemming, lemmatization
- Implement text vectorization - TF-IDF and CountVectorizer (Bag of Words)
- Build machine learning classification models (Naive Bayes and Logistic Regression)
- Evaluate model performance using classification metrics

---

## Repository Structure

```
NLP-Lab-Assignment-4/
|
|-- NLP_Lab_Assignment4_Nilesh.ipynb   # Main Jupyter notebook
|-- README.md                          # This file
```

---

## Dataset

A custom dataset of **40 labelled news headlines** across **4 categories**:

| Category   | Count |
|------------|-------|
| Technology | 10    |
| Sports     | 10    |
| Politics   | 10    |
| Health     | 10    |
| **Total**  | **40**|

---

## Notebook Structure

| Section | Title | Description |
|---------|-------|-------------|
| 1 | Setup and Imports | Install/import NLTK, sklearn, matplotlib |
| 2 | Dataset Creation | 40 labelled news headlines |
| 3 | NLP Preprocessing | Tokenization, stopword removal, stemming, lemmatization |
| 4 | Text Vectorization | Bag of Words (CountVectorizer) + TF-IDF |
| 5 | Model Building | Multinomial Naive Bayes + Logistic Regression |
| 6 | Model Evaluation | Classification report, accuracy, precision, recall, F1 |
| 7 | Visualizations | Confusion matrix, metrics bar chart, TF-IDF keyword charts |
| 8 | Prediction | Classify new/unseen headlines with confidence scores |
| 9 | Summary | Conclusions for all learning outcomes |

---

## Preprocessing Pipeline

Each headline goes through the following steps:

```
Raw Text
   |
Lowercasing
   |
Tokenization (NLTK word_tokenize)
   |
Stopword Removal (198 NLTK English stopwords)
   |
Stemming (PorterStemmer)
   |
Lemmatization (WordNetLemmatizer with POS tagging)
   |
Cleaned Text
```

---

## Models Used

### 1. Multinomial Naive Bayes
- Baseline probabilistic classifier
- Well-suited for TF-IDF/count features
- Fast and interpretable

### 2. Logistic Regression
- Linear classifier with `lbfgs` solver
- `max_iter=1000`, `C=1.0`, `random_state=42`
- Strong baseline for multi-class text tasks

---

## Vectorization

| Method | Description |
|--------|-------------|
| **CountVectorizer** | Bag of Words - raw term counts |
| **TF-IDF** | Term Frequency-Inverse Document Frequency with unigrams + bigrams, max 500 features |

---

## Results

Both models were trained on **30 samples** and tested on **10 samples** (75/25 split).

> **Note:** The dataset is intentionally small (lab exercise). Low accuracy (~40%) is expected behavior for a 4-class classification task with only 40 samples. The random baseline would be 25%, so both models are learning signal from the data.

---

## How to Run

### Option 1: Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/)
2. Click **File - Upload notebook**
3. Upload `NLP_Lab_Assignment4_Nilesh.ipynb`
4. Click **Runtime - Run all**

### Option 2: Local Jupyter
```bash
# Install dependencies
pip install nltk scikit-learn matplotlib seaborn pandas numpy

# Launch notebook
jupyter notebook NLP_Lab_Assignment4_Nilesh.ipynb
```

### Required Libraries
```python
nltk
scikit-learn
matplotlib
seaborn
pandas
numpy
```

---

## NLTK Downloads (Auto-handled in notebook)

```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

---

## References

- [NLTK Documentation](https://www.nltk.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)

---

## License

This project is submitted as a lab assignment for academic purposes only.
