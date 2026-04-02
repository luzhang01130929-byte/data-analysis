# data-analysis
This repository contains the custom Python scripts for our analytical pipeline.we used these code files to perform a clustering analysis. We analyzed the heavy water assimilation rate (CD%) of the cable bacteria cells.
# 聚类分析：层次聚类与 K-Means 交叉验证

## 📊 项目简介
本项目使用 Python 对包含 100 个样本的数据集进行了无监督聚类分析。项目首先采用**层次聚类（Ward 链接，欧式距离）**将数据划分为 3 类，随后使用 **K-均值（K-means）**算法进行交叉验证，并计算了调整兰德指数（ARI）以评估方法间的一致性。

## 📈 核心可视化结果

### 1. 层次聚类分布与轮廓系数评估
下图展示了层次聚类的树状图（Dendrogram）、聚类后的二维空间分布以及用于评估聚类质量的轮廓系数图。
![层次聚类结果](Figure S1.tif) 

### 2. K-Means 验证与方法一致性
为验证层次聚类的稳定性，本研究引入了 K-Means 聚类。下方热力图展示了两种算法分类结果的混淆矩阵。
![方法一致性热力图](Figure S2.tif)
**结论：** 调整兰德指数 (ARI) 达到 0.943，表明两种聚类方法在当前数据集上的划分结果具有极高的一致性。

## 💻 核心使用的技术栈
* **Python**
* **scikit-learn** (AgglomerativeClustering, KMeans, 评价指标)
* **scipy** (dendrogram, linkage)
* **matplotlib & seaborn** (数据可视化)

*(完整代码请查看本仓库中的 `clustering_analysis.py` 文件)*
