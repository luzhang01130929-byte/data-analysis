# Data-analysis
This repository contains the custom Python scripts for our analytical pipeline.we used these code files to perform a clustering analysis. We analyzed the heavy water assimilation rate (CD%) of the cable bacteria cells.
# Clustering Analysis: Hierarchical Clustering and K-Means Cross-Validation

## 📊 Project Overview
This project performs an unsupervised clustering analysis on a dataset containing 100 samples using Python. The project first utilizes Hierarchical Clustering (Ward linkage, Euclidean distance) to partition the data into 3 clusters, followed by cross-validation using the K-Means algorithm. The Adjusted Rand Index (ARI) is calculated to evaluate the consistency between the two methods.

## 📈 Core Visualization Results

### 1. Hierarchical Clustering Distribution and Silhouette Coefficient Evaluation
The figure below displays the hierarchical clustering dendrogram, the 2D spatial distribution after clustering, and the silhouette plot used to assess clustering quality.(Figure S1.tif) 

### 2.K-Means Validation and Method Consistency
To verify the stability of the hierarchical clustering, this study introduces K-Means clustering. The heatmap below illustrates the confusion matrix of the classification results from both algorithms.(Figure S2.tif)
**Conclusion** : The Adjusted Rand Index (ARI) reached 0.943, indicating an extremely high level of consistency between the partitioning results of the two clustering methods on the current dataset.

## 💻 核心使用的技术栈
* **Python**
* **scikit-learn** (AgglomerativeClustering, KMeans, evaluation metrics)
* **scipy** (dendrogram, linkage)
* **matplotlib & seaborn** (Data visualization)

*(For the complete code, please refer to the clustering_analysis.py file in this repository)*
DOI 10.5281/zenodo.19388345
