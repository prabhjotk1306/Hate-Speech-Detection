
# Hate Speech Detection

This project presents a machine learning pipeline to classify tweets into three categories: Hate Speech, Offensive Language, or Neither. It uses various natural language processing (NLP) techniques, feature engineering, and classification algorithms to detect harmful content in online social media posts.

## Dataset

The dataset used is a labeled CSV file `HateSpeechData.csv`, which contains:

- `tweet`: The content of the tweet.  
- `class`: Label (0 = Hate Speech, 1 = Offensive Language, 2 = Neither)

---

## Project Pipeline

### 1. Exploratory Data Analysis

- Visualization of text lengths across classes using histograms and boxplots  
- Class distribution analysis  
- WordClouds for frequently used words per class

### 2. Preprocessing

- Lowercasing all text  
- Removal of:
  - User mentions (e.g. `@user`)
  - URLs
  - Punctuation and numbers  
- Tokenization  
- Stopword removal (including Twitter-specific words like "rt")  
- Stemming using the Porter stemmer

### 3. Feature Engineering

**Text-Based Features:**
- TF-IDF (Term Frequency-Inverse Document Frequency) vectors using unigrams and bigrams

**Sentiment Features:**
- VADER sentiment scores (Negative, Neutral, Positive, Compound)  
- Count of hashtags, mentions, and URLs

**Document Embeddings:**
- Doc2Vec vectors trained using Gensim to generate document-level embeddings

**Readability and Complexity Metrics:**
- Flesch-Kincaid Grade and Reading Ease scores  
- Syllable count  
- Word and character counts  
- Vocabulary richness

---

## Model Training

**Models Used:**
- Logistic Regression  
- Random Forest Classifier  
- Naive Bayes Classifier  
- Linear Support Vector Classifier (SVM)

**Feature Set Combinations:**
- F1: TF-IDF only  
- F2: TF-IDF + Sentiment features  
- F3: TF-IDF + Sentiment + Doc2Vec  
- F4: All features combined (TF-IDF, sentiment, Doc2Vec, and readability metrics)

**Evaluation Metrics:**
- Accuracy  
- Precision, Recall, F1-Score (via classification report)  
- Confusion Matrix (normalized per class)

---

## Key Findings

- TF-IDF features contributed the most to classification accuracy  
- Doc2Vec embeddings added minimal improvement to performance  
- Readability-based features added slight gains in some models  
- SVM and Logistic Regression worked best with feature-rich datasets  
- Naive Bayes performed better with simpler feature combinations like TF-IDF + sentiment

---

## Visualizations

- WordClouds for each class (hate, offensive, neutral)  
- Histogram plots for tweet lengths and class distributions  
- Accuracy comparison bar charts for different models and feature sets  
- Confusion matrices for deeper performance insights

---

## Dependencies

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn nltk sklearn gensim wordcloud textstat
```

Download necessary NLTK corpora:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

---

## How to Run

1. Clone the repository or download the code  
2. Open the Jupyter notebook `hate_speech_detection.ipynb`  
3. Run all cells step by step to preprocess, train, and evaluate models
