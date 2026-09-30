# 🎬 Movie Recommender System

A machine learning and deep learning recommender system built using the **MovieLens 20M dataset**. This repository covers the complete end-to-end pipeline: data preprocessing, memory-based collaborative filtering, matrix factorization using **SVD / SVD++ (Surprise)**, deep learning with **Hybrid Neural Collaborative Filtering (HybridNCF in PyTorch)**, evaluation with ranking metrics, and an interactive **Gradio** web application.

---

## 📌 Project Overview

Recommender systems are essential for handling information overload and delivering personalized content. This project explores multiple recommendation paradigms:

1. **Memory-Based Collaborative Filtering**: User-item similarity using Cosine Similarity on sparse interaction matrices.
2. **Matrix Factorization (SVD / SVD++)**: Latent factor modeling using the `scikit-surprise` library to capture underlying user preference profiles.
3. **Hybrid Neural Collaborative Filtering (HybridNCF)**: A PyTorch deep learning architecture combining user and movie embeddings with multi-hot genre features passed through a multi-layer perceptron (MLP).
4. **Interactive Deployment**: A Gradio web application for real-time Top-$K$ personalized movie recommendations.

---

## 📊 Dataset

This project utilizes the **MovieLens 20M Dataset** provided by GroupLens Research:
- **Ratings**: ~20,000,000 ratings applied to 27,000+ movies by 138,000+ users.
- **Movies**: Movie metadata including `movieId`, `title`, and pipe-separated `genres`.
- **Additional Metadata**: Tags, genome tags, and genome scores.

> **Note**: Due to GitHub's file size limits (>100 MB), the raw dataset CSVs and large zip archives are excluded via `.gitignore`. You can obtain the dataset from [GroupLens MovieLens 20M](https://grouplens.org/datasets/movielens/20m/) or [Kaggle MovieLens 20M](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset).

---

## 🧠 Model Architectures & Approaches

### 1. Singular Value Decomposition (SVD)
Factorizes the user-movie rating matrix $R \approx U \cdot \Sigma \cdot V^T$ into latent user and item representations.
- **Library**: `scikit-surprise`
- **Tuned Hyperparameters**:
  - Latent Factors (`n_factors`): $100$
  - Learning Rate (`lr_all`): $0.005$
  - Regularization (`reg_all`): $0.02$
  - Training Epochs: $20$
- **Optimization**: Hyperparameter tuning evaluated with `GridSearchCV`.

### 2. Hybrid Neural Collaborative Filtering (HybridNCF)
Implemented in **PyTorch** to combine collaborative signals with content metadata (genres):
- **User Embeddings**: Dimension $64$
- **Item Embeddings**: Dimension $64$
- **Genre Representation**: Multi-hot binarized genre vectors projected via a linear layer (dim $64$)
- **MLP Layers**: Concatenated input ($192 \rightarrow 128 \rightarrow 64 \rightarrow 32 \rightarrow 1$) with ReLU activations
- **Loss Function**: Mean Squared Error (MSE) / Adam Optimizer

---

## 📈 Evaluation Metrics

The system is evaluated using both rating prediction error and top-$K$ ranking quality:
- **RMSE (Root Mean Squared Error)**: Measures the precision of predicted rating values.
- **Precision@K**: Proportion of recommended items in the top-$K$ that are relevant.
- **Recall@K**: Proportion of relevant items captured within the top-$K$ recommendations.
- **NDCG@K (Normalized Discounted Cumulative Gain)**: Evaluates ranking quality by giving higher weight to relevant items ranked near the top.

---

## 📂 Repository Structure

```text
├── app.py                           # Gradio web interface for interactive recommendations
├── Movie Recomendation.ipynb        # Main notebook: EDA, SVD, HybridNCF & evaluation
├── Movie Recomendation grid.ipynb   # Grid search experiments & SVD tuning
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Git ignore rules for datasets, models, and docs
└── README.md                        # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Running the Interactive App
Make sure your trained model file (`svd_model_7866.pkl`) and `movies.csv` (or `movie.csv`) are placed in the root directory:
```bash
python app.py
```
Open the local URL displayed in your terminal (usually `http://127.0.0.1:7860`) in your web browser.

---

## 🖥️ Gradio Web Application

The Gradio interface provides:
- **User ID Input**: Enter any valid user ID from the dataset.
- **Top-K Slider**: Choose between 1 and 20 recommendations.
- **Interactive Output**: Returns a ranked list of recommended movie titles, genres, and predicted rating scores.

---

## 📄 Notes for Git & GitHub

- **Large Files Excluded**: Datasets (`*.csv`, `*.zip`) and large serialized models (`*.pkl`, `*.pth` exceeding 100MB) are ignored by `.gitignore` to prevent GitHub push errors.
- **Reports Excluded**: Per project requirements, document and presentation files (`*.pdf`, `*.docx`) are ignored by `.gitignore`.
