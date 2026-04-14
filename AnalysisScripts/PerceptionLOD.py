import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# --------------------------
# 数据预处理与特征计算
# --------------------------
# 加载数据（示例数据需保存为CSV）
data = pd.read_csv("D:\\Yunlei_Data\\LOD_Scene\\real_experiment_data\\results\\aoi_pivot_summary.csv")

# 定义关键AOI分类
target_aoi = ['sign', 'window', 'decoration']
roaming_aoi = ['green', 'Background']
all_aoi = ['building','Background','wall','green','window','sign',
           'undefined','lid','door','decoration','manhole']

# 计算特征指标
def calculate_features(row):
    # 强制类型转换 + 百分比转概率
    prob = row[all_aoi].astype(float).values / 100
    
    # 目标导向指数
    target = np.sum(row[target_aoi].astype(float))
    
    # 漫游指数
    roaming = np.sum(row[roaming_aoi].astype(float))
    
    # 信息熵计算（鲁棒性修正）
    if np.all(prob <= 0.01):
        # 保留至少一个最大AOI
        prob = [np.max(prob)]
    else:
        prob = prob[prob > 0.01]
    
    if np.sum(prob) == 0:
        entropy = 0.0
    else:
        prob_normalized = prob / np.sum(prob)
        entropy = -np.sum(prob_normalized * np.log2(prob_normalized, where=prob_normalized>0))
    
    return pd.Series([target, roaming, entropy])

# 生成特征矩阵
features = data.apply(calculate_features, axis=1)
features.columns = ['Target_Score', 'Roaming_Score', 'Information_Entropy']

# 数据标准化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# --------------------------
# 聚类分析（K-means）
# --------------------------
# 执行聚类
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# 评估聚类质量
silhouette_avg = silhouette_score(scaled_features, clusters)
print(f"轮廓系数: {silhouette_avg:.2f}")

# 将聚类结果标记到原始数据
data['Cluster'] = clusters

# --------------------------
# 统计验证
# --------------------------
# ANOVA检验三类差异
f_val_target, p_val_target = stats.f_oneway(
    features[data.Cluster == 0]['Target_Score'],
    features[data.Cluster == 1]['Target_Score'],
    features[data.Cluster == 2]['Target_Score']
)

print(f"目标导向指数ANOVA: F={f_val_target:.1f}, p={p_val_target:.4f}")

# --------------------------
# 可视化模块
# --------------------------
def plot_3d_clusters():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 定义颜色与标签
    colors = ['r', 'g', 'b']
    labels = ['Goal-oriented', 'Perception roaming', 'Information picking']
    
    for i in range(3):
        cluster_data = scaled_features[data.Cluster == i]
        ax.scatter(cluster_data[:,0], cluster_data[:,1], cluster_data[:,2],
                   c=colors[i], label=labels[i], s=50, alpha=0.6)
    
    ax.set_xlabel('Goal-oriented）')
    ax.set_ylabel('Perception roaming')
    ax.set_zlabel('Information picking')
    ax.legend()
    plt.title("Three-dimensional cluster distribution (Silhouette coefficient =%.2f）" % silhouette_avg)
    plt.show()

def plot_radar_chart():
    # 计算类中心原始值
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # 雷达图参数
    categories = ['目标导向', '环境漫游', '信息熵']
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    for i, center in enumerate(cluster_centers):
        values = center.tolist()
        values += values[:1]  # 闭合曲线
        ax.plot(angles + angles[:1], values, linewidth=2, 
                label=f'Cluster {i}')
        ax.fill(angles + angles[:1], values, alpha=0.25)
    
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles), categories)
    ax.legend(loc='upper right')
    plt.title("聚类中心雷达图")
    plt.show()

# 执行可视化
plot_3d_clusters()
plot_radar_chart()

# --------------------------
# 输出分析报告
# --------------------------
# 将聚类结果合并到特征数据中
features_with_cluster = features.copy()
features_with_cluster['Cluster'] = clusters
features_with_cluster['LOD'] = data['LOD']  # 从原始数据中提取LOD列

# 生成类别描述报告
cluster_report = features_with_cluster.groupby('Cluster').agg({
    'Target_Score': 'mean',
    'Roaming_Score': 'mean',
    'Information_Entropy': 'mean',
    'LOD': lambda x: x.mode()[0]
}).reset_index()

print("\n聚类特征报告:")
print(cluster_report.round(2))