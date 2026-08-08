# Project 3 — Customer Segmentation using Unsupervised Learning

## DecodeLabs Industrial Training | Data Science

### 📌 Project Overview

This project focuses on customer segmentation using unsupervised
machine learning techniques.

The objective is to discover hidden groups of customers based on
their demographic, financial, and purchasing characteristics and
translate those mathematical clusters into actionable business
personas.

---

## 🎯 Objectives

- Explore customer data
- Select relevant clustering features
- Standardize numerical features
- Apply Principal Component Analysis (PCA)
- Determine the optimal number of clusters
- Apply K-Means clustering
- Evaluate clusters using the Elbow Method
- Evaluate cluster quality using Silhouette Score
- Profile customer segments
- Create actionable business personas

---

## 📊 Dataset

The dataset contains customer purchasing information.

### Features

| Feature | Description |
|---|---|
| Age | Customer age |
| Income | Customer income |
| CreditScore | Customer credit score |
| PreviousPurchases | Number of previous purchases |
| Purchased | Purchase outcome |
| index | Customer identifier |

### Clustering Features

The following features were used:

- Age
- Income
- CreditScore
- PreviousPurchases

The `index` column was excluded because it is an identifier.

The `Purchased` column was excluded because it represents a
known supervised-learning target and should not be used to
create unsupervised clusters.

---

## 🧠 Methodology

```text
Customer Dataset
       ↓
Data Exploration
       ↓
Feature Selection
       ↓
StandardScaler
       ↓
PCA
       ↓
Elbow Method
       ↓
Silhouette Score
       ↓
K-Means
       ↓
Cluster Profiling
       ↓
Business Personas
