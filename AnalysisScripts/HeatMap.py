import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import time

def read_csv(file_path):
    """读取CSV文件并返回DataFrame"""
    return pd.read_csv(file_path)

def filter_data(data):
    """根据LOD字段的值筛选数据，返回包含LOD列和hitUV列的DataFrame"""
    return data[['LOD', 'hitUV']].dropna(subset=['LOD', 'hitUV'])

def create_lod_dict(filtered_data, lod_coordinates):
    """创建一个字典来存储每个LOD对应的hitUV列表"""
    for lod, hitUV in filtered_data.itertuples(index=False):
        if lod not in lod_coordinates:
            lod_coordinates[lod] = []
        lod_coordinates[lod].append(hitUV)
    return lod_coordinates

def convert_coordinates(lod_coordinates):
    """将每个hitUV转换为(x, y)坐标，并存储为一个字典"""
    for lod, hitUV_list in lod_coordinates.items():
        coordinates = []
        for coord in hitUV_list:
            try:
                x, y = map(float, coord.strip('()').split(','))
                x = x + 0.025
                y = y - 0.16
                # 移除固定偏移，改为确保坐标在有效范围内
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                coordinates.append((x, y))
            except ValueError as e:
                print(f"Error converting {coord}: {e}")
        lod_coordinates[lod] = coordinates
    return lod_coordinates

def load_eye_tracking_data(file_dic):
    """加载所有CSV文件的眼动数据并拼接"""
    lod_coordinates = {}
    file_count = 0
    participant_count = {}
    
    for file_name in os.listdir(file_dic):
        if file_name.endswith('.csv'):
            file_path = os.path.join(file_dic, file_name)
            print(f"正在处理文件: {file_name}")
            data = read_csv(file_path)
            filtered_data = filter_data(data)
            
            # 记录每个场景的参与者数量
            for lod in filtered_data['LOD'].unique():
                if lod not in participant_count:
                    participant_count[lod] = 0
                participant_count[lod] += 1
            
            lod_coordinates = create_lod_dict(filtered_data, lod_coordinates)
            file_count += 1
    
    lod_coordinates = convert_coordinates(lod_coordinates)
    print(f"共处理了{file_count}个CSV文件")
    print(f"每个场景的参与者数量统计: {participant_count}")
    
    return lod_coordinates

def uv_to_pixel(u, v, image_width, image_height):
    """将UV坐标转换为像素坐标"""
    x = int(u * image_width)
    y = int(v * image_height)
    return x, y

def create_anisotropic_kde(points_array, x_bandwidth=0.03, y_bandwidth=0.01):
    """创建各向异性KDE，对x和y方向使用不同的带宽"""
    # 大幅减小带宽，提高局部敏感度
    x_scale = 1.0 / x_bandwidth
    y_scale = 1.0 / y_bandwidth
    
    # 复制原始数据
    transformed_points = points_array.copy()
    
    # 分别缩放x和y坐标
    transformed_points[0, :] = transformed_points[0, :] * x_scale
    transformed_points[1, :] = transformed_points[1, :] * y_scale
    
    # 使用非常小的带宽创建KDE，进一步增加对局部特征的敏感度
    kde = gaussian_kde(transformed_points, bw_method=0.2)
    
    def evaluate(points):
        eval_points = points.copy()
        eval_points[0, :] = eval_points[0, :] * x_scale
        eval_points[1, :] = eval_points[1, :] * y_scale
        return kde(eval_points)
    
    return evaluate

def create_heatmap_anisotropic_kde(pixel_points, image_path, output_path, 
                                  x_bandwidth=0.03, y_bandwidth=0.01, 
                                  resolution_factor=0.25, alpha=0.7, intensity_factor=3.0):
    """使用各向异性KDE方法创建热力图并与原图叠加"""
    # 读取原始图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    height, width = image.shape[:2]
    
    # 确保有点
    if not pixel_points or len(pixel_points) < 2:
        print("警告：没有足够的眼动数据点用于KDE")
        cv2.imwrite(output_path, image)
        return
    
    # 转换为适合KDE的格式并筛选有效点
    valid_points = [(x, y) for x, y in pixel_points if 0 <= x < width and 0 <= y < height]
    print(f"有效坐标点数量: {len(valid_points)}/{len(pixel_points)}")
    
    if len(valid_points) < 2:
        print("警告：没有足够的有效点用于KDE")
        cv2.imwrite(output_path, image)
        return
    
    # 转为numpy数组
    points_array = np.array(valid_points).T
    
    # 计算有效点的标准差
    x_std = np.std(points_array[0, :])
    y_std = np.std(points_array[1, :])
    print(f"X方向标准差: {x_std:.4f}, Y方向标准差: {y_std:.4f}")
    
    # 使用更小的固定带宽，不进行动态调整
    adjusted_x_bandwidth = x_bandwidth
    adjusted_y_bandwidth = y_bandwidth
    
    print(f"带宽 - X: {adjusted_x_bandwidth:.4f}, Y: {adjusted_y_bandwidth:.4f}")
    
    # 提高分辨率因子，获得更精细的热力图
    reduced_width = int(width * resolution_factor)
    reduced_height = int(height * resolution_factor)
    
    # 创建网格
    x_grid = np.linspace(0, width-1, reduced_width)
    y_grid = np.linspace(0, height-1, reduced_height)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    
    try:
        start_time = time.time()
        
        print("开始各向异性KDE计算...")
        kde_func = create_anisotropic_kde(points_array, 
                                         x_bandwidth=adjusted_x_bandwidth, 
                                         y_bandwidth=adjusted_y_bandwidth)
        z = kde_func(grid_coords)
        
        heatmap_small = z.reshape(reduced_height, reduced_width)
        kde_time = time.time() - start_time
        print(f"KDE计算完成，耗时: {kde_time:.2f} 秒")
        
        heatmap = cv2.resize(heatmap_small, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # 强烈增强热点对比度
        heatmap = np.power(heatmap, 0.3) * intensity_factor
        
        # 归一化热力图
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-10)
        
        # 使用较高阈值消除弱信号区域
        threshold = 0.65
        heatmap = np.where(heatmap < threshold, heatmap * 0.05, heatmap)
        
        # 进一步增强高强度热点
        threshold_high = 0.85
        heatmap = np.where(heatmap > threshold_high, np.power(heatmap, 0.5) * 1.5, heatmap)
        
        # 进行中值滤波以减少噪声并保留热点
        heatmap = np.float32(heatmap)
        heatmap = cv2.medianBlur(heatmap, 3)
        
        heatmap = np.uint8(255 * heatmap)
        
        # 应用颜色映射，使用更突出热点的颜色映射
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 叠加热力图和原图
        overlay = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
        
        cv2.imwrite(output_path, overlay)
        cv2.imwrite(output_path.replace('.png', '_pure.png'), heatmap_colored)
        
        total_time = time.time() - start_time
        print(f"已生成热力图: {output_path}, 总耗时: {total_time:.2f} 秒")
        
    except Exception as e:
        print(f"生成热力图时出错: {e}")
        create_heatmap_traditional(pixel_points, image_path, output_path)

def create_heatmap_traditional(pixel_points, image_path, output_path, weight=30, sigma=15, alpha=0.7):
    """使用传统高斯模糊方法创建热力图"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    height, width = image.shape[:2]
    
    # 创建空白热力图
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    if not pixel_points:
        print("警告：没有眼动数据点")
        cv2.imwrite(output_path, image)
        return
    
    valid_points = [(x, y) for x, y in pixel_points if 0 <= x < width and 0 <= y < height]
    print(f"有效坐标点数量: {len(valid_points)}/{len(pixel_points)}")
    
    # 使用更小的权重和sigma使热点更加局部化
    for x, y in valid_points:
        if 0 <= x < width and 0 <= y < height:
            x, y = int(x), int(y)
            heatmap[y, x] += weight
    
    # 使用更小的sigma
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma)
    
    # 更强的非线性变换
    heatmap = np.power(heatmap, 0.3) * 2.5
    
    if np.max(heatmap) > 0:
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-10)
        
        # 使用高阈值
        threshold = 0.65
        heatmap = np.where(heatmap < threshold, heatmap * 0.05, heatmap)
        
        heatmap = np.uint8(255 * heatmap)
    else:
        print("警告：热力图全为零")
        heatmap = np.zeros((height, width), dtype=np.uint8)
    
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
    cv2.imwrite(output_path, overlay)
    print(f"已使用传统方法生成热力图: {output_path}")

def generate_statistics(lod_coordinates, output_folder):
    """生成各场景眼动数据的统计信息"""
    stats = {}
    for lod, coords in lod_coordinates.items():
        stats[lod] = {
            'data_points': len(coords),
            'x_mean': np.mean([c[0] for c in coords]) if coords else 0,
            'y_mean': np.mean([c[1] for c in coords]) if coords else 0,
            'x_std': np.std([c[0] for c in coords]) if coords else 0,
            'y_std': np.std([c[1] for c in coords]) if coords else 0
        }
    
    # 保存为CSV
    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    stats_path = os.path.join(output_folder, 'eye_tracking_stats.csv')
    stats_df.to_csv(stats_path)
    print(f"统计信息已保存至: {stats_path}")
    
    return stats

def visualize_eye_tracks(pixel_points, image_path, output_path, max_lines=200, alpha=0.7):
    """创建眼动轨迹可视化图"""
    # 读取原始图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    height, width = image.shape[:2]
    
    # 确保有点
    if not pixel_points or len(pixel_points) < 2:
        print("警告：没有足够的眼动数据点用于轨迹图")
        cv2.imwrite(output_path, image)
        return
    
    # 筛选有效点
    valid_points = [(x, y) for x, y in pixel_points if 0 <= x < width and 0 <= y < height]
    print(f"轨迹图有效坐标点数量: {len(valid_points)}/{len(pixel_points)}")
    
    # 如果点太多，随机采样以避免图像过于混乱
    if len(valid_points) > max_lines:
        import random
        random.seed(42)  # 使结果可重现
        sample_indices = random.sample(range(len(valid_points) - 1), max_lines)
        sampled_points = [valid_points[i] for i in sorted(sample_indices)]
        sampled_points_next = [valid_points[i + 1] for i in sorted(sample_indices)]
    else:
        sampled_points = valid_points[:-1]
        sampled_points_next = valid_points[1:]
    
    # 创建半透明叠加层
    overlay = image.copy()
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(sampled_points, sampled_points_next)):
        # 根据时间顺序改变颜色，从蓝色到红色
        color_val = int(255 * i / len(sampled_points))
        color = (255 - color_val, 0, color_val)  # BGR: 从蓝色到红色
        
        # 绘制线段
        cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
        
        # 在起点绘制小圆点
        if i == 0:
            cv2.circle(overlay, (int(x1), int(y1)), 5, (0, 255, 0), -1)  # 起点为绿色
    
    # 在终点绘制小圆点
    if sampled_points_next:
        last_point = sampled_points_next[-1]
        cv2.circle(overlay, (int(last_point[0]), int(last_point[1])), 5, (0, 0, 255), -1)  # 终点为红色
    
    # 将轨迹图与原图混合
    result = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
    
    # 添加图例
    legend_height = 30
    legend_img = np.ones((legend_height, width, 3), dtype=np.uint8) * 255
    
    # 绘制颜色渐变条
    grad_width = width - 200
    for i in range(grad_width):
        color_val = int(255 * i / grad_width)
        color = (255 - color_val, 0, color_val)  # BGR: 从蓝色到红色
        cv2.line(legend_img, (100 + i, 5), (100 + i, 25), color, 1)
    
    # 添加起点和终点标记
    cv2.circle(legend_img, (90, 15), 5, (0, 255, 0), -1)
    cv2.circle(legend_img, (100 + grad_width + 10, 15), 5, (0, 0, 255), -1)
    
    # 添加文字标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(legend_img, "起点", (60, 20), font, 0.5, (0, 0, 0), 1)
    cv2.putText(legend_img, "终点", (100 + grad_width + 20, 20), font, 0.5, (0, 0, 0), 1)
    cv2.putText(legend_img, "时间流向", (width//2 - 30, 20), font, 0.5, (0, 0, 0), 1)
    
    # 将图例添加到结果图像
    result_with_legend = np.vstack((result, legend_img))
    
    # 保存结果
    cv2.imwrite(output_path, result_with_legend)
    print(f"已生成眼动轨迹图: {output_path}")

def process_scenes(file_dic, scenes_folder, output_folder, use_kde=True, 
                   x_bandwidth=0.15, y_bandwidth=0.04, resolution_factor=0.25, 
                   create_tracks=True, alpha=0.7):
    """处理所有场景的热力图"""
    start_total = time.time()
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 加载所有CSV文件的眼动数据
    print("开始加载眼动数据...")
    load_start = time.time()
    lod_coordinates = load_eye_tracking_data(file_dic)
    load_time = time.time() - load_start
    print(f"数据加载完成，耗时: {load_time:.2f} 秒")
    
    # 生成统计信息
    stats = generate_statistics(lod_coordinates, output_folder)
    
    # 创建统计信息汇总表
    scene_stats = []
    for scene, stat in stats.items():
        scene_stats.append({
            'Scene': scene,
            'DataPoints': stat['data_points'],
            'X_Mean': stat['x_mean'],
            'Y_Mean': stat['y_mean'],
            'X_Std': stat['x_std'],
            'Y_Std': stat['y_std'],
            'Std_Ratio': stat['x_std'] / stat['y_std'] if stat['y_std'] > 0 else 0
        })
    
    # 将统计汇总保存为CSV
    summary_df = pd.DataFrame(scene_stats)
    summary_path = os.path.join(output_folder, 'scene_stats_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"场景统计汇总已保存至: {summary_path}")

    # 处理每个场景
    scene_count = 0
    for scene_image in os.listdir(scenes_folder):
        if scene_image.endswith(('.jpg', '.png')):
            scene_path = os.path.join(scenes_folder, scene_image)
            scene_name = os.path.splitext(scene_image)[0]
            
            scene_start = time.time()
            print(f"\n开始处理场景: {scene_name}")

            # 获取当前场景的数据
            if scene_name in lod_coordinates:
                scene_data = lod_coordinates[scene_name]
                print(f"场景{scene_name}找到{len(scene_data)}个眼动数据点")
            else:
                print(f"场景{scene_name}没有对应的眼动数据")
                continue

            # 获取图片尺寸
            img = cv2.imread(scene_path)
            if img is None:
                print(f"无法读取图片: {scene_path}")
                continue
                
            height, width = img.shape[:2]
            print(f"图片尺寸: {width}x{height}")

            # 转换UV坐标到像素坐标
            pixel_points = [
                uv_to_pixel(coord[0], coord[1], width, height)
                for coord in scene_data
            ]

            # 打印调试信息
            if pixel_points:
                print(f"场景: {scene_name}, 眼动数据点数: {len(pixel_points)}")
                if len(pixel_points) >= 5:
                    print(f"示例像素坐标: {pixel_points[:5]}")
            else:
                print(f"警告: 场景{scene_name}没有有效的像素坐标")
                continue

            # 生成热力图
            output_path = os.path.join(output_folder, f'heatmap_{scene_name}.png')
            
            if use_kde and len(pixel_points) >= 10:  # KDE需要足够的数据点
                create_heatmap_anisotropic_kde(pixel_points, scene_path, output_path, 
                                    x_bandwidth=x_bandwidth, y_bandwidth=y_bandwidth,
                                    resolution_factor=resolution_factor, alpha=alpha)
            else:
                create_heatmap_traditional(pixel_points, scene_path, output_path, 
                                          weight=100, sigma=50, alpha=alpha)
            
            # 可选生成眼动轨迹图
            if create_tracks:
                tracks_path = os.path.join(output_folder, f'tracks_{scene_name}.png')
                visualize_eye_tracks(pixel_points, scene_path, tracks_path, alpha=alpha)
            
            scene_time = time.time() - scene_start
            print(f"场景{scene_name}处理完成，耗时: {scene_time:.2f} 秒")
            scene_count += 1

    total_time = time.time() - start_total
    print(f"\n所有热力图处理完成! 共处理{scene_count}个场景，总耗时: {total_time:.2f} 秒")

def batch_process_with_parameters(file_dic, scenes_folder, output_base_folder):
    """使用多组参数批量处理热力图，便于对比不同参数效果"""
    # 定义不同的参数组合
    parameter_sets = [
        {
            'name': 'standard',
            'x_bandwidth': 0.25,
            'y_bandwidth': 0.08, 
            'alpha': 0.6
        },
        {
            'name': 'strong_anisotropic',
            'x_bandwidth': 0.3,
            'y_bandwidth': 0.05,
            'alpha': 0.6
        },
        {
            'name': 'smooth',
            'x_bandwidth': 0.3,
            'y_bandwidth': 0.15,
            'alpha': 0.7
        }
    ]
    
    # 为每组参数创建单独的输出文件夹并处理
    for params in parameter_sets:
        param_output_folder = os.path.join(output_base_folder, f"heatmaps_{params['name']}")
        print(f"\n\n开始使用参数集 '{params['name']}' 处理热力图")
        print(f"X带宽: {params['x_bandwidth']}, Y带宽: {params['y_bandwidth']}, 透明度: {params['alpha']}")
        
        process_scenes(
            file_dic, 
            scenes_folder, 
            param_output_folder,
            use_kde=True,
            x_bandwidth=params['x_bandwidth'],
            y_bandwidth=params['y_bandwidth'],
            resolution_factor=0.25,
            create_tracks=True,
            alpha=params['alpha']
        )

if __name__ == "__main__":
    # 设置路径
    file_dic = 'D:/Yunlei_Data/eye_data_analysis/real_experiment_data'
    scenes_folder = 'D:/Yunlei_Data/LOD_Scene/images'
    output_folder = 'D:/Yunlei_Data/eye_data_analysis/heatmaps'
    
    run_batch_processing = True
    
    if run_batch_processing:
        parameter_sets = [
            {
                'name': 'multi_hotspot',
                'x_bandwidth': 0.03,  # 大幅减小带宽
                'y_bandwidth': 0.01, 
                'alpha': 0.8
            },
            {
                'name': 'very_detailed',
                'x_bandwidth': 0.02,  # 极小带宽
                'y_bandwidth': 0.008,
                'alpha': 0.8
            },
            {
                'name': 'extreme_detail',
                'x_bandwidth': 0.01,  # 极端小带宽
                'y_bandwidth': 0.005,
                'alpha': 0.7
            }
        ]
        
        for params in parameter_sets:
            param_output_folder = os.path.join('D:/Yunlei_Data/eye_data_analysis', f"heatmaps_{params['name']}")
            print(f"\n\n开始使用参数集 '{params['name']}' 处理热力图")
            print(f"X带宽: {params['x_bandwidth']}, Y带宽: {params['y_bandwidth']}, 透明度: {params['alpha']}")
            
            process_scenes(
                file_dic, 
                scenes_folder, 
                param_output_folder,
                use_kde=True,
                x_bandwidth=params['x_bandwidth'],
                y_bandwidth=params['y_bandwidth'],
                resolution_factor=0.3,  # 略微提高分辨率
                create_tracks=True,
                alpha=params['alpha']
            )