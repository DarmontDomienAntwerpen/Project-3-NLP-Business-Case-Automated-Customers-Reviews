# Amazon Review Analyzer
**IronHack Project — Domien Darmont**

An end-to-end NLP pipeline that classifies, clusters, and summarizes Amazon product reviews — deployed as an interactive Streamlit web app.

---

## Pipeline and notebooks (01,02,03,04 + app)

```
Raw Reviews (46,169)
    ↓
01 Preprocessing      → reviews_clean.csv
    ↓
02 Classification     → Fine-tuned BERT (accuracy: 79.4%)
    ↓
03 Clustering         → KMeans + all-mpnet embeddings (k=5)
    ↓
04 Summarization      → Llama3.2 3B via Ollama serving
    ↓
    App         → Streamlit web app.py
```

---

## Project Structure

```
project/
|-- readme.md
|-- ironhack_assignment.md (ironhack assignment details)
├── app.py                          # Streamlit web app
├── requirements.txt
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_review_classification_base.ipynb
│   ├── 02b_review_classification_finetuned.ipynb (final model)
│   ├── 03_clustering.ipynb
│   ├── 04_summarization.ipynb
└── data/
    ├── raw/                        # Original Kaggle CSVs (3)
    ├── processed/                  # Generated files (not all where used finally)
    │   ├── data_processedreviews_clean.csv
    │   ├── clustered_revieuws + reviews_full_clustered.csv (full clustered for deployment app)
    │   ├── classified_reviews.csv
    │   ├── articles.csv (generated summary and text generation)
    │   └── rouge_scores.csv
    └── finetuned_model/
        └── bert_sentiment_finetuned/
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama (for summarization)
```bash
brew install ollama
ollama pull llama3.2
```

### 3. Run notebooks in order
```
01 → 02 → 02b → 03 → 04
```

Deployment :

### 4. Start Ollama server (keep terminal open)
```bash
ollama serve
```

### 5. Run the web app
```bash
streamlit run app.py
```

---

## Models

| Component | Model | Performance |
|-----------|-------|-------------|
| Classification | `nlptown/bert-base-multilingual-uncased-sentiment` (fine-tuned) | Accuracy: 79.4% |
| Clustering | `all-mpnet-base-v2` + KMeans (k=5) | Silhouette: 0.343 |
| Summarization | `llama3.2` (3B) via Ollama | ROUGE-1: 0.276 avg |

---

## Dataset

[Amazon Product Reviews](https://www.kaggle.com/) — 46,169 reviews across 5 product clustered categories:
`Fire Tablets` · `Batteries` · `Echo & Smart Speakers` · `Kindle E-readers` · `Fire TV & Accessories`

---

## Web App Pages

| Page | Description |
|------|-------------|
| 📊 Dashboard | Sentiment & rating stats per category |
| 🏆 Top Products | Wirecutter-style product recommendations |
| 📂 Classify Review | Live BERT sentiment classifier |
| 📝 Add Review | Submit a review — classified in real time |
