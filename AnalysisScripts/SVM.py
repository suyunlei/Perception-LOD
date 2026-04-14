import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import re

def extract_scene_labels(scene_names):
    """从场景名称中提取标签，如 HHL 分别代表 Feature Complexity, Appearance, Semantics"""
    labels = {}
    unmatched_scenes = []  # 用于记录未匹配的场景

    for scene in scene_names:
        # 使用正则表达式匹配 HHL 格式（三个字符，每个可以是 H/M/L）
        match = re.search(r'([HML])-([HML])-([HL])', scene)
        if match:
            # 将 H/M/L 转换为数值
            feature_complexity_value = 3 if match.group(1) == 'H' else (2 if match.group(1) == 'M' else 1)
            appearance_value = 3 if match.group(2) == 'H' else (2 if match.group(2) == 'M' else 1)
            semantics_value = 2 if match.group(3) == 'H' else 1

            labels[scene] = {
                'feature_complexity': feature_complexity_value,
                'appearance': appearance_value,
                'semantics': semantics_value
            }
        else:
            unmatched_scenes.append(scene)

    if unmatched_scenes:
        print(f"未匹配的场景: {unmatched_scenes}")

    return labels

def extract_eye_tracking_features(lod_coordinates):
    """从眼动坐标数据中提取有意义的特征"""
    features = {}
    
    for scene, coordinates in lod_coordinates.items():
        if not coordinates:
            continue
            
        # 将坐标转换为numpy数组，便于计算
        coords_array = np.array(coordinates)
        x_coords = coords_array[:, 0]
        y_coords = coords_array[:, 1]
        
        # 基本统计特征
        features[scene] = {
            # 数据点数量
            'num_points': len(coordinates),
            
            # 空间分布特征
            'x_mean': np.mean(x_coords),
            'y_mean': np.mean(y_coords),
            'x_std': np.std(x_coords),
            'y_std': np.std(y_coords),
            'x_range': np.max(x_coords) - np.min(x_coords),
            'y_range': np.max(y_coords) - np.min(y_coords),
            
            # 集中度/分散度特征
            'spatial_density': len(coordinates) / ((np.max(x_coords) - np.min(x_coords)) * 
                                                 (np.max(y_coords) - np.min(y_coords)) + 1e-10),
            
            # 几何特征
            'aspect_ratio': np.std(x_coords) / (np.std(y_coords) + 1e-10),
            
            # 四分位数特征
            'x_25': np.percentile(x_coords, 25),
            'x_50': np.percentile(x_coords, 50),
            'x_75': np.percentile(x_coords, 75),
            'y_25': np.percentile(y_coords, 25),
            'y_50': np.percentile(y_coords, 50),
            'y_75': np.percentile(y_coords, 75),
            
            # 视线移动特征（相邻点之间的距离统计）
            'mean_saccade': np.mean([np.sqrt((x_coords[i+1]-x_coords[i])**2 + 
                                           (y_coords[i+1]-y_coords[i])**2) 
                                   for i in range(len(coordinates)-1)]) if len(coordinates) > 1 else 0,
            
            # 热点区域特征 - 使用网格分割法统计点的集中程度
            'entropy': calculate_spatial_entropy(x_coords, y_coords, num_bins=10)
        }
        
        # 添加注视点聚类特征 - 将空间分为4x4的网格，统计每个区域的点的密度
        grid_densities = calculate_grid_density(x_coords, y_coords, grid_size=4)
        for idx, density in enumerate(grid_densities):
            features[scene][f'grid_density_{idx}'] = density
    
    return features

def calculate_spatial_entropy(x_coords, y_coords, num_bins=10):
    """计算空间熵 - 眼动点的分布均匀程度"""
    # 创建2D直方图
    hist, _, _ = np.histogram2d(x_coords, y_coords, bins=num_bins, range=[[0, 1], [0, 1]])
    
    # 归一化直方图，使其概率和为1
    hist = hist / np.sum(hist)
    
    # 计算熵
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    return entropy

def calculate_grid_density(x_coords, y_coords, grid_size=4):
    """将眼动区域分为grid_size x grid_size的网格，统计每个网格中的点的密度"""
    # 初始化网格密度
    densities = np.zeros(grid_size * grid_size)
    
    # 计算每个点所在的网格
    x_bins = np.floor(x_coords * grid_size).astype(int)
    y_bins = np.floor(y_coords * grid_size).astype(int)
    
    # 确保索引在有效范围内
    x_bins = np.clip(x_bins, 0, grid_size - 1)
    y_bins = np.clip(y_bins, 0, grid_size - 1)
    
    # 计算每个点的网格索引
    indices = y_bins * grid_size + x_bins
    
    # 统计每个网格中的点数
    for i in range(grid_size * grid_size):
        densities[i] = np.sum(indices == i) / len(x_coords)
    
    return densities

def prepare_dataset(features, labels, target_label):
    """准备用于训练SVM的数据集"""
    X = []
    y = []
    scene_names = []
    
    for scene, feat in features.items():
        if scene in labels and target_label in labels[scene]:
            X.append(list(feat.values()))
            y.append(labels[scene][target_label])
            scene_names.append(scene)
    
    return np.array(X), np.array(y), scene_names

def train_svm_model(X_train, y_train, X_test, y_test, target_label):
    """训练SVM模型并评估性能"""
    # 创建包含标准化和SVM的管道
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced'))
    ])
    
    # 参数网格搜索
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.1, 0.01]
    }
    
    # 使用网格搜索找到最佳参数
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    print(f"\n最佳参数 ({target_label}):", grid_search.best_params_)
    
    # 在测试集上评估
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{target_label} 分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
    plt.title(f'混淆矩阵 - {target_label}')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{target_label}.png')
    
    return best_model, accuracy, y_pred

def load_eye_tracking_data(file_dic):
    """从代码中提取的函数，用于加载并预处理眼动数据"""
    lod_coordinates = {}
    file_count = 0
    participant_count = {}
    
    for file_name in os.listdir(file_dic):
        if file_name.endswith('.csv'):
            file_path = os.path.join(file_dic, file_name)
            print(f"正在处理文件: {file_name}")
            data = pd.read_csv(file_path)
            filtered_data = data[['LOD', 'hitUV']].dropna(subset=['LOD', 'hitUV'])
            
            # 记录每个场景的参与者数量
            for lod in filtered_data['LOD'].unique():
                if lod not in participant_count:
                    participant_count[lod] = 0
                participant_count[lod] += 1
            
            # 创建LOD字典
            for lod, hitUV in filtered_data.itertuples(index=False):
                if lod not in lod_coordinates:
                    lod_coordinates[lod] = []
                lod_coordinates[lod].append(hitUV)
            file_count += 1
    
    # 转换坐标
    for lod, hitUV_list in lod_coordinates.items():
        coordinates = []
        for coord in hitUV_list:
            try:
                x, y = map(float, coord.strip('()').split(','))
                x = x + 0.025
                y = y - 0.16
                # 确保坐标在有效范围内
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                coordinates.append((x, y))
            except ValueError as e:
                print(f"错误转换坐标 {coord}: {e}")
        lod_coordinates[lod] = coordinates
    
    print(f"共处理了{file_count}个CSV文件")
    print(f"每个场景的参与者数量: {participant_count}")
    
    return lod_coordinates

def main_analysis(file_dic):
    """主分析流程"""
    # 加载眼动数据
    print("加载眼动数据...")
    lod_coordinates = load_eye_tracking_data(file_dic)
    
    # 从场景名称中提取标签
    print("提取场景标签...")
    scene_labels = extract_scene_labels(lod_coordinates.keys())
    print(f"共有{len(scene_labels)}个场景有有效标签")
    
    # 提取眼动特征
    print("提取眼动特征...")
    eye_features = extract_eye_tracking_features(lod_coordinates)
    print(f"共提取了{len(eye_features)}个场景的特征")
    
    # 为每个目标标签训练单独的SVM模型
    target_labels = ['feature_complexity', 'appearance', 'semantics']
    
    results = {}
    feature_importances = {}
    
    for target in target_labels:
        print(f"\n\n开始训练预测{target}的模型...")
        
        # 准备数据集
        X, y, scene_names = prepare_dataset(eye_features, scene_labels, target)
        print(f"数据集大小: {X.shape[0]}个样本, {X.shape[1]}个特征")
        print(f"标签分布: {np.unique(y, return_counts=True)}")
        
        # 划分训练集和测试集 (70% - 30%)
        X_train, X_test, y_train, y_test, train_scenes, test_scenes = train_test_split(
            X, y, scene_names, test_size=0.3, random_state=42, stratify=y
        )
        
        # 训练SVM模型
        model, accuracy, y_pred = train_svm_model(X_train, y_train, X_test, y_test, target)
        
        # 记录结果
        results[target] = {
            'accuracy': accuracy,
            'test_scenes': test_scenes,
            'true_labels': y_test,
            'predicted_labels': y_pred
        }
        
    # 打印最终结果摘要
    print("\n\n========== 最终结果摘要 ==========")
    for target, result in results.items():
        print(f"{target}: 准确率 = {result['accuracy']:.4f}")
    
    return results

# 主程序
if __name__ == "__main__":
    # 设置眼动数据文件夹路径
    file_dic = 'D:/Yunlei_Data/eye_data_analysis/real_experiment_data'  # 请修改为您的实际路径
    
    # 运行分析
    results = main_analysis(file_dic)