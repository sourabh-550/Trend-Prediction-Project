# 🎯 YouTube Trend Prediction System

An end-to-end Machine Learning system that predicts which YouTube videos/topics are likely to trend in the near future and presents insights through an interactive dashboard.

This project focuses on predictive analytics, feature engineering, model evaluation, and real-world deployment considerations.

---

## 📌 Project Objective

To build a machine learning system that:
- Predicts whether a YouTube video will trend within a defined time window.
- Assigns a **Trending Probability Score** instead of a simple binary output.
- Provides interpretable insights via an interactive dashboard.
- Enables data-driven content strategy decisions.

---

## 🚀 Why This Project Matters

- Helps creators identify high-potential content early.
- Assists marketers and brands in planning timely campaigns.
- Moves beyond static trending lists by providing **predictive insights**.
- Demonstrates real-world ML pipeline development (data → model → deployment).

---

## 🗂️ Data Sources

- **YouTube Data API v3**
  - Video title
  - Description
  - Tags
  - Category ID
  - Publish time
  - View count
  - Like count
  - Comment count
  - Region-based trending list

- Historical trending data used for supervised labeling.

---

## 🧠 Problem Formulation

### Classification Task
Predict whether a video will trend (Yes / No) within X days of publishing.

### Probability-Based Output
Model outputs a **Trending Probability (%)** instead of a strict label.

### Ranking System
Videos can be ranked based on predicted trend likelihood.

---

## 🏗️ Feature Engineering

### 1️⃣ Engagement Features
- views
- likes
- comments
- like_ratio = likes / (views + 1)
- comment_ratio = comments / (views + 1)
- engagement_score = (likes + comments) / (views + 1)

### 2️⃣ Temporal Features
- publish_hour
- publish_day_of_week
- time_since_publish

### 3️⃣ Textual Features
- TF-IDF vectors from title + description
- Keyword frequency
- N-grams
- Cleaned & normalized text

### 4️⃣ Metadata Features
- category_id
- historical performance signals (if available)

---

## 🧹 Data Preprocessing

- Removed punctuation and special characters
- Lowercased all text
- Removed stopwords
- Tokenization
- Optional lemmatization
- Combined title + description + tags into single feature

---

## 🤖 Model Development

### Models Used
- Logistic Regression (Baseline)
- XGBoost (Primary Model)

### Why XGBoost?
- Handles heterogeneous feature types
- Works well with tabular + sparse data
- Built-in handling of missing values
- Strong performance on structured ML problems

---

## ⚖️ Handling Class Imbalance

Since very few videos actually trend:

- Used class weighting
- Evaluated using F1-score
- Focused on precision-recall tradeoff
- Avoided accuracy as primary metric

---

## 📊 Evaluation Strategy

- Time-based train-test split (realistic deployment simulation)
- Metrics:
  - F1 Score
  - ROC-AUC
  - Precision@K
  - Confusion Matrix

This ensures the model generalizes to future unseen data.

---

## 📈 Explainability

- Feature importance analysis
- Human-readable reasoning in dashboard:
  - "High engagement velocity"
  - "Strong keyword presence"
  - "Category trending pattern"

This improves trust and transparency.

---

## 🖥️ Dashboard (Streamlit)

Features:
- Trending Probability (%) display
- “Likely to Trend” vs “Not Trending Yet” labels
- Search functionality
- Threshold slider
- Download predictions as CSV
- Engagement visualizations
- Clean, non-technical UI

---

## 🛠️ Technology Stack

- Python
- Pandas / NumPy
- Scikit-Learn
- XGBoost
- TF-IDF
- NLTK / SpaCy
- Streamlit
- Matplotlib / Plotly
- GitHub

---

## 🧩 Project Architecture

1. Data Collection (YouTube API)
2. Preprocessing & Feature Engineering
3. Model Training
4. Evaluation
5. Deployment via Streamlit

---

## 🔒 Production Considerations

- API keys stored in environment variables
- Caching to reduce API rate limits
- Periodic model retraining (trend drift handling)
- Error handling for API failures

---

## ⚠️ Limitations

- Dependent on available engagement metrics
- Trend patterns shift over time (concept drift)
- Predictions are probabilistic, not guarantees
- Platform bias may influence outcomes

---

## 🔮 Future Improvements

- Incorporate Google Trends signals
- Add cross-platform trend validation (Reddit, Twitter)
- Use transformer-based fine-tuning
- Add SHAP explainability visuals
- Automate retraining pipeline

---

## 📌 Key Outcomes

- Built a complete ML pipeline from scratch.
- Implemented real-world feature engineering.
- Designed a probability-based ranking system.
- Delivered an interpretable ML dashboard.
- Demonstrated strong applied ML + deployment skills.

---

## 📎 How to Run

1. Clone the repository
2. Install dependencies
3. Add YouTube API key to environment variables
4. Run:
   streamlit run app.py

---

## 👨‍💻 Author

Developed as a full end-to-end Machine Learning project focusing on real-world trend prediction, feature engineering, model evaluation, and deployment.

---

⭐ If you found this project interesting, feel free to connect or discuss improvements.
