# 🎵 Spotify Recommendation System

A music recommendation system built using both **Content-Based Filtering** and **Collaborative Filtering** techniques to generate personalized song recommendations.

## 📌 Project Overview

The objective of this project is to build a recommendation engine capable of suggesting relevant songs using:

* 🎯 Content-Based Filtering (song similarity)
* 👥 Collaborative Filtering (user listening behavior)

These two recommendation engines will later be combined into a Hybrid Recommendation System.

---

# 📊 Dataset

### Songs Dataset

* ~50,000 songs
* Audio features
* Artist metadata
* User-generated tags

### User Interaction Dataset

* ~96,000 users
* Millions of listening events
* Play count information

---

# 🔍 Exploratory Data Analysis

Key observations:

* Removed 815 duplicate songs
* Genre column contained ~56% missing values and was dropped
* Tags contained only ~2% missing values and became the primary text signal
* More than 20,000 songs had zero user interactions, highlighting the cold-start problem

---

# 🎯 Content-Based Filtering

The content-based recommender suggests songs based on metadata and audio characteristics.

## Features Used

### Audio Features

* Danceability
* Energy
* Loudness
* Speechiness
* Acousticness
* Instrumentalness
* Liveness
* Valence
* Tempo

### Metadata Features

* Artist
* Release Year
* Key
* Time Signature
* Tags

## Feature Engineering

| Feature Type                | Technique                     |
| --------------------------- | ----------------------------- |
| Audio Features              | MinMaxScaler / StandardScaler |
| Artist, Key, Time Signature | OneHotEncoder                 |
| Release Year                | CountEncoder                  |
| Tags                        | TF-IDF                        |

A Scikit-Learn ColumnTransformer was used to combine all preprocessing steps into a single pipeline.

## Similarity Computation

Songs are converted into feature vectors and stored as a sparse feature matrix.

Recommendations are generated using Cosine Similarity:

```python
cosine_similarity(song_vector, transformed_data)
```

Final feature matrix:

* 50,674 songs
* 8,431 engineered features
* Sparse representation for memory efficiency

---

# 👥 Collaborative Filtering

Collaborative Filtering recommends songs based on listening behavior rather than song attributes.

## Interaction Matrix

The listening history was transformed into a Track × User interaction matrix.

| Dimension | Description |
| --------- | ----------- |
| Rows      | Songs       |
| Columns   | Users       |
| Values    | Play Counts |

## Processing Pipeline

* Dask for large-scale data processing
* User and Track indexing
* Sparse matrix generation using SciPy CSR format
* Cosine similarity on listener vectors

## Generated Artifacts

* interaction_matrix.npz
* track_ids.npy
* collab_filtered_data.csv

## Recommendation Process

1. Extract the target song's listener vector
2. Compute cosine similarity against all songs
3. Rank songs by similarity score
4. Return Top-K recommendations

Unlike Content-Based Filtering, songs can be recommended even if they have very different audio characteristics, provided they attract similar audiences.

---

# 🎛️ Streamlit Application

The project includes an interactive Streamlit application.

### Features

* Song Search
* Artist Validation
* Top-K Recommendations
* Spotify Song Preview Playback
* Content-Based Recommendations
* Collaborative Filtering Recommendations
* Recommendation Method Selection via Dropdown

---

# ⚙️ DVC Pipeline

The project uses DVC to create reproducible machine learning workflows.

### Content-Based Pipeline

Raw Data
→ Data Cleaning
→ Feature Engineering
→ Feature Matrix Generation
→ Model Artifacts

### Collaborative Pipeline

Listening History
→ Interaction Filtering
→ Sparse Matrix Construction
→ Model Artifacts

Benefits:

* Reproducible pipelines
* Version-controlled datasets
* Automated artifact tracking

---

# 🛠️ Tech Stack

### Data Processing

* Pandas
* NumPy
* Dask

### Machine Learning

* Scikit-Learn
* SciPy
* Category Encoders

### MLOps

* DVC

### Frontend

* Streamlit

---

# 🚀 Next Steps

* Hybrid Recommendation Engine
* Dynamic Recommendation Weighting
* Cold-Start Handling
* Dockerization
* CI/CD Pipeline
* AWS Deployment

---

# 🎯 Goal

Build a production-ready Hybrid Spotify Recommendation System that combines content similarity and user behavior to deliver personalized and diverse music recommendations.
