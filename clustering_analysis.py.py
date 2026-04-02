import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
import pandas as pd
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
data = np.array([data001]).reshape(-1, 1)
print("=" * 60)
print("HIERARCHICAL CLUSTERING ANALYSIS")
print("=" * 60)
print("\n1. Calculating distance matrix and linkage...")
dist_matrix = pdist(data, metric='euclidean')
Z = linkage(dist_matrix, method='ward' )
print("\n2. Evaluating optimal number of clusters...")
silhouette_scores = []
n_clusters_range = range(3, 10)
for n_clusters in n_clusters_range:
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    silhouette_avg = silhouette_score(data, labels)
    silhouette_scores.append(silhouette_avg)
    print(f"  - {n_clusters} clusters: silhouette score = {silhouette_avg:.3f}")
optimal_n = n_clusters_range[np.argmax(silhouette_scores)]
print(f"\n3. Optimal number of clusters: {optimal_n} (max silhouette score = {max(silhouette_scores):.3f})")
hierarchical_labels = fcluster(Z, optimal_n, criterion='maxclust')
hierarchical_labels_adjusted = hierarchical_labels - 1  # 调整标签从0开始
print("\n4. Cluster statistics:")
unique_labels = np.unique(hierarchical_labels_adjusted)
cluster_stats = []
for label in unique_labels:
    cluster_data = data[hierarchical_labels_adjusted == label].flatten()
    stats = {
        'Cluster': label,
        'Size': len(cluster_data),
        'Mean': np.mean(cluster_data),
        'Std': np.std(cluster_data),
        'Min': np.min(cluster_data),
        'Max': np.max(cluster_data),
        'Range': np.max(cluster_data) - np.min(cluster_data)
    }
    cluster_stats.append(stats)
    print(f"  Cluster {label}: n={stats['Size']}, "
          f"mean={stats['Mean']:.3f}±{stats['Std']:.3f}, "
          f"range=[{stats['Min']:.3f}, {stats['Max']:.3f}]")

print("\n5. Silhouette scores for each cluster:")
sample_silhouette_values = silhouette_samples(data, hierarchical_labels_adjusted)

for label in unique_labels:
    cluster_silhouette_avg = sample_silhouette_values[hierarchical_labels_adjusted == label].mean()
    print(f"  Cluster {label}: {cluster_silhouette_avg:.3f}")

df_hierarchical = pd.DataFrame({
    'Value': data.flatten(),
    'Hierarchical_Cluster': hierarchical_labels_adjusted,
    'Silhouette_Value': sample_silhouette_values
})

print("\n" + "=" * 60)
print("K-MEANS CLUSTERING VALIDATION")
print("=" * 60)

kmeans = KMeans(n_clusters=optimal_n, random_state=42, n_init=20)
kmeans_labels = kmeans.fit_predict(data)

print("\n1. K-means cluster statistics:")
kmeans_stats = []
for label in range(optimal_n):
    cluster_data = data[kmeans_labels == label].flatten()
    stats = {
        'Cluster': label,
        'Size': len(cluster_data),
        'Mean': np.mean(cluster_data),
        'Std': np.std(cluster_data),
        'Min': np.min(cluster_data),
        'Max': np.max(cluster_data),
        'Centroid': kmeans.cluster_centers_[label][0]
    }
    kmeans_stats.append(stats)
    print(f"  Cluster {label}: n={stats['Size']}, "
          f"mean={stats['Mean']:.3f}±{stats['Std']:.3f}, "
          f"centroid={stats['Centroid']:.3f}")

kmeans_silhouette_avg = silhouette_score(data, kmeans_labels)
print(f"\n2. K-means overall silhouette score: {kmeans_silhouette_avg:.3f}")

print("\n3. Method consistency analysis:")
ari = adjusted_rand_score(hierarchical_labels_adjusted, kmeans_labels)
print(f"  - Adjusted Rand Index: {ari:.3f}")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(hierarchical_labels_adjusted, kmeans_labels)
print(f"  - Confusion matrix:\n{cm}")

df_kmeans = pd.DataFrame({
    'Value': data.flatten(),
    'KMeans_Cluster': kmeans_labels,
    'KMeans_Silhouette': silhouette_samples(data, kmeans_labels)
})
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)
fig = plt.figure(figsize=(18, 12))
ax1 = plt.subplot(3, 4, 1)
dendrogram(Z, truncate_mode='lastp', p=optimal_n,
           show_leaf_counts=True, leaf_rotation=90, leaf_font_size=8,
           show_contracted=True)
plt.title(f'Hierarchical Clustering Dendrogram\n(Optimal: {optimal_n} clusters)',
          fontsize=10, fontweight='bold')
plt.xlabel('Sample Index')
plt.ylabel('Distance (Ward Linkage)')
ax2 = plt.subplot(3, 4, 2)
plt.plot(n_clusters_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=optimal_n, color='r', linestyle='--', alpha=0.7)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Optimal Cluster Number Determination', fontsize=10, fontweight='bold')
plt.grid(True, alpha=0.3)
ax3 = plt.subplot(3, 4, 3)
colors = plt.cm.Set2(np.linspace(0, 1, optimal_n))
for label in unique_labels:
    cluster_data = data[hierarchical_labels_adjusted == label].flatten()
    plt.scatter(cluster_data, np.zeros_like(cluster_data) + np.random.normal(0, 0.02, len(cluster_data)),
                color=colors[label], s=30, alpha=0.7, label=f'Cluster {label} (n={len(cluster_data)})')
plt.xlabel('Value')
plt.ylabel('')
plt.title('Hierarchical Clustering Distribution', fontsize=10, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.yticks([])
ax4 = plt.subplot(3, 4, 4)
y_lower = 10
for label in unique_labels:
    label_silhouette_values = sample_silhouette_values[hierarchical_labels_adjusted == label]
    label_silhouette_values.sort()
    size = len(label_silhouette_values)
    y_upper = y_lower + size
    color = colors[label]
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, label_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size, f'Cluster {label}')
    y_lower = y_upper + 10
plt.axvline(x=silhouette_score(data, hierarchical_labels_adjusted),
            color="red", linestyle="--")
plt.xlabel("Silhouette Coefficient Values")
plt.ylabel("Cluster Label")
plt.title(f"Silhouette Plot (Hierarchical)\nAverage: {silhouette_score(data, hierarchical_labels_adjusted):.3f}",
          fontsize=10, fontweight='bold')
ax5 = plt.subplot(3, 4, 5)
for label in range(optimal_n):
    cluster_data = data[kmeans_labels == label].flatten()
    plt.scatter(cluster_data, np.zeros_like(cluster_data) + np.random.normal(0, 0.02, len(cluster_data)),
                color=colors[label], s=30, alpha=0.7, label=f'Cluster {label} (n={len(cluster_data)})')
plt.xlabel('Value')
plt.ylabel('')
plt.title('K-means Clustering Distribution', fontsize=10, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.yticks([])
ax6 = plt.subplot(3, 4, 6)
x_pos = np.arange(optimal_n)
width = 0.35
for i, label in enumerate(unique_labels):
    cluster_data = data[hierarchical_labels_adjusted == label].flatten()
    plt.bar(x_pos[i] - width/2, np.mean(cluster_data), width,
            color=colors[label], alpha=0.7, label=f'Hierarchical {label}')
    plt.bar(x_pos[i] + width/2, kmeans.cluster_centers_[label][0], width,
            color=colors[label], alpha=0.7, hatch='//', label=f'K-means {label}')
plt.xlabel('Cluster')
plt.ylabel('Mean Value / Centroid')
plt.title('Cluster Centers Comparison', fontsize=10, fontweight='bold')
plt.xticks(x_pos, [f'Cluster {i}' for i in range(optimal_n)])
ax7 = plt.subplot(3, 4, 7)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=[f'K-means {i}' for i in range(optimal_n)],
            yticklabels=[f'Hierarchical {i}' for i in range(optimal_n)])
plt.title(f'Method Consistency\nARI = {ari:.3f}', fontsize=10, fontweight='bold')
plt.xlabel('K-means Clusters')
plt.ylabel('Hierarchical Clusters')
ax8 = plt.subplot(3, 4, 8)
sorted_indices = np.argsort(data.flatten())
sorted_data = data[sorted_indices].flatten()
hierarchical_sorted = hierarchical_labels_adjusted[sorted_indices]
kmeans_sorted = kmeans_labels[sorted_indices]

plt.scatter(range(len(sorted_data)), sorted_data,
            c=hierarchical_sorted, cmap='Set2', s=20, alpha=0.6,
            label='Hierarchical')
plt.scatter(range(len(sorted_data)), sorted_data,
            c=kmeans_sorted, cmap='Set2', s=5, alpha=0.6, marker='s',
            label='K-means')
plt.xlabel('Sorted Index')
plt.ylabel('Value')
plt.title('Data Values with Cluster Assignments', fontsize=10, fontweight='bold')
plt.legend(fontsize=8)
ax9 = plt.subplot(3, 4, 9)
hierarchical_sizes = [len(data[hierarchical_labels_adjusted == i]) for i in range(optimal_n)]
kmeans_sizes = [len(data[kmeans_labels == i]) for i in range(optimal_n)]
x = np.arange(optimal_n)
plt.bar(x - 0.2, hierarchical_sizes, 0.4, label='Hierarchical', alpha=0.7)
plt.bar(x + 0.2, kmeans_sizes, 0.4, label='K-means', alpha=0.7)
plt.xlabel('Cluster')
plt.ylabel('Number of Points')
plt.title('Cluster Size Comparison', fontsize=10, fontweight='bold')
plt.xticks(x, [f'Cluster {i}' for i in range(optimal_n)])
plt.legend(fontsize=8)
ax10 = plt.subplot(3, 4, 10)
hierarchical_silhouette = silhouette_score(data, hierarchical_labels_adjusted)
kmeans_silhouette = silhouette_score(data, kmeans_labels)
methods = ['Hierarchical', 'K-means']
scores = [hierarchical_silhouette, kmeans_silhouette]
colors_bar = ['#2E86AB', '#A23B72']
bars = plt.bar(methods, scores, color=colors_bar, alpha=0.7)
for bar, score in zip(bars, scores):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{score:.3f}', ha='center', va='bottom', fontsize=9)
plt.ylabel('Silhouette Score')
plt.title('Overall Silhouette Score Comparison', fontsize=10, fontweight='bold')
plt.ylim(0, max(scores) * 1.2)
ax11 = plt.subplot(3, 4, 11)
x_min, x_max = data.min() - 0.05, data.max() + 0.05
xx = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
for label in unique_labels:
    cluster_data = data[hierarchical_labels_adjusted == label].flatten()
    from scipy.stats import gaussian_kde
    if len(cluster_data) > 1:
        kde = gaussian_kde(cluster_data)
        xx_dense = np.linspace(cluster_data.min(), cluster_data.max(), 200)
        plt.plot(xx_dense, kde(xx_dense), color=colors[label],
                linewidth=2, label=f'Cluster {label}')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Cluster Density Estimation', fontsize=10, fontweight='bold')
plt.legend(fontsize=8)

ax12 = plt.subplot(3, 4, 12)
ax12.axis('off')
summary_text = (
    f"CLUSTERING ANALYSIS SUMMARY\n\n"
    f"Dataset: 100 single-cell measurements\n"
    f"Optimal clusters: {optimal_n}\n\n"
    f"HIERARCHICAL CLUSTERING\n"
    f"• Overall silhouette: {silhouette_score(data, hierarchical_labels_adjusted):.3f}\n"
    f"• Cluster sizes: {', '.join([str(s) for s in hierarchical_sizes])}\n\n"
    f"K-MEANS CLUSTERING\n"
    f"• Overall silhouette: {kmeans_silhouette_avg:.3f}\n"
    f"• Cluster sizes: {', '.join([str(s) for s in kmeans_sizes])}\n\n"
    f"METHOD CONSISTENCY\n"
    f"• Adjusted Rand Index: {ari:.3f}\n"
    f"• Interpretation: {'Excellent' if ari > 0.8 else 'Good' if ari > 0.6 else 'Moderate'}"
)
plt.text(0.05, 0.95, summary_text, transform=ax12.transAxes,
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Comprehensive Clustering Analysis: Hierarchical vs K-means',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('comprehensive_clustering_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n" + "=" * 60)
print("EXPORTING RESULTS")
print("=" * 60)
df_complete = pd.DataFrame({
    'Index': range(1, len(data) + 1),
    'Value': data.flatten(),
    'Hierarchical_Cluster': hierarchical_labels_adjusted,
    'Hierarchical_Silhouette': sample_silhouette_values,
    'KMeans_Cluster': kmeans_labels,
    'KMeans_Silhouette': silhouette_samples(data, kmeans_labels)
})

def get_cluster_description(row, method='hierarchical'):
    if method == 'hierarchical':
        for stats in cluster_stats:
            if stats['Cluster'] == row['Hierarchical_Cluster']:
                return f"{stats['Mean']:.3f}±{stats['Std']:.3f}"
    else:
        for stats in kmeans_stats:
            if stats['Cluster'] == row['KMeans_Cluster']:
                return f"{stats['Mean']:.3f}±{stats['Std']:.3f}"
    return ""

df_complete['Hierarchical_Cluster_Desc'] = df_complete.apply(
    lambda row: get_cluster_description(row, 'hierarchical'), axis=1)
df_complete['KMeans_Cluster_Desc'] = df_complete.apply(
    lambda row: get_cluster_description(row, 'kmeans'), axis=1)

df_complete.to_csv('clustering_results_detailed.csv', index=False)
print("Detailed results exported to 'clustering_results_detailed.csv'")

df_summary = pd.DataFrame({
    'Method': ['Hierarchical'] * optimal_n + ['K-means'] * optimal_n,
    'Cluster': list(range(optimal_n)) * 2,
    'Size': hierarchical_sizes + kmeans_sizes,
    'Mean': [stats['Mean'] for stats in cluster_stats] + [stats['Mean'] for stats in kmeans_stats],
    'Std': [stats['Std'] for stats in cluster_stats] + [stats['Std'] for stats in kmeans_stats],
    'Min': [stats['Min'] for stats in cluster_stats] + [stats['Min'] for stats in kmeans_stats],
    'Max': [stats['Max'] for stats in cluster_stats] + [stats['Max'] for stats in kmeans_stats],
    'Silhouette_Mean': [
        sample_silhouette_values[hierarchical_labels_adjusted == i].mean()
        for i in range(optimal_n)
    ] + [
        silhouette_samples(data, kmeans_labels)[kmeans_labels == i].mean()
        for i in range(optimal_n)
    ]
})
df_summary.to_csv('clustering_statistics_summary.csv', index=False)
print("Statistics summary exported to 'clustering_statistics_summary.csv'")

print("\nAnalysis complete!")