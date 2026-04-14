import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import warnings

# Suppress specific warnings that might occur during optimization
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 1. 读取CSV数据
# 请将这里的路径修改为您的CSV文件路径
analysis_df = pd.read_csv("D:\\Yunlei_Data\\LOD_Scene\\real_experiment_data\\results\\aoi_pivot_summary.csv")

# 检查数据中是否有NaN值
if analysis_df.isna().any().any():
    print("警告：数据中含有NaN值，尝试填充...")
    # 对数值列用0填充NaN
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
    analysis_df[numeric_cols] = analysis_df[numeric_cols].fillna(0)

# 2. LOD编码处理
# 将LOD字符串转换为特征值（例如：H-H-H转为Feature_Complexity=3, Appearance=3, Semantics=3）
def parse_lod_code(lod_code):
    if not isinstance(lod_code, str):
        return 2, 2, 2  # 默认值，如果LOD不是字符串
    
    # 分割LOD编码
    parts = lod_code.split('-')
    if len(parts) != 3:
        return 2, 2, 2  # 如果格式不符，返回默认值
    
    # 转换为数值: H=3, M=2, L=1
    fc = 3 if parts[0] == 'H' else (2 if parts[0] == 'M' else 1)
    ap = 3 if parts[1] == 'H' else (2 if parts[1] == 'M' else 1)
    se = 3 if parts[2] == 'H' else (2 if parts[2] == 'M' else 1)
    
    return fc, ap, se

# 应用编码转换
analysis_df[['Feature_Complexity', 'Appearance', 'Semantics']] = pd.DataFrame(
    [parse_lod_code(lod) for lod in analysis_df['LOD']], 
    index=analysis_df.index
)

# 3. 定义不同的参数组合模型
def model_linear(params, FC, AP, SE):
    """简单线性加权模型"""
    w1, w2, w3 = params
    return w1*FC + w2*AP + w3*SE

def model_quadratic(params, FC, AP, SE):
    """二次项模型，增加FC的影响力"""
    w1, w2, w3, w4 = params
    return w1*FC + w2*FC**2 + w3*AP + w4*SE

def model_interaction(params, FC, AP, SE):
    """交互项模型，考虑参数间的交互作用"""
    w1, w2, w3, w4, w5, w6 = params
    return w1*FC + w2*AP + w3*SE + w4*FC*AP + w5*FC*SE + w6*AP*SE

def model_exponential(params, FC, AP, SE):
    """指数模型，FC产生指数效应"""
    w1, w2, w3 = params
    # 限制指数范围以避免溢出
    w1 = min(0.5, max(-0.5, w1))
    return np.exp(w1*FC) + w2*AP + w3*SE

def model_factorial(params, FC, AP, SE):
    """因子模型，使用乘法关系"""
    w1, w2, w3 = params
    # 限制指数以避免过大或过小的值
    w1 = min(3, max(0.1, w1))
    w2 = min(3, max(0.1, w2))
    w3 = min(3, max(0.1, w3))
    return (FC**w1) * (AP**w2) * (SE**w3)

def model_logarithmic(params, FC, AP, SE):
    """对数模型"""
    w1, w2, w3 = params
    return w1*np.log1p(FC) + w2*np.log1p(AP) + w3*np.log1p(SE)

# 定义优化目标函数
def objective_function(params, model_func, X, y):
    """优化目标：最小化预测与实际的差异"""
    FC, AP, SE = X
    try:
        y_pred = model_func(params, FC, AP, SE)
        # 检查输出是否含有无效值
        if np.isnan(y_pred).any() or np.isinf(y_pred).any():
            return 1e10  # 返回一个很大的错误值
        return np.mean((y_pred - y)**2)
    except:
        return 1e10  # 出错时返回一个很大的错误值

# 4. 评估模型函数
def evaluate_models(aoi_type):
    """评估不同模型对特定AOI注视的预测能力"""
    FC = analysis_df['Feature_Complexity'].values
    AP = analysis_df['Appearance'].values
    SE = analysis_df['Semantics'].values
    y = analysis_df[aoi_type].values
    
    # 检查目标值是否含有无效值
    if np.isnan(y).any() or np.isinf(y).any():
        print(f"警告：{aoi_type}列中含有无效值，尝试处理...")
        y = np.nan_to_num(y, nan=0.0, posinf=100.0, neginf=0.0)
    
    model_funcs = {
        'Linear': (model_linear, np.ones(3)),
        'Quadratic': (model_quadratic, np.ones(4)),
        'Interaction': (model_interaction, np.ones(6)),
        'Exponential': (model_exponential, np.array([0.1, 1.0, 1.0])),
        'Factorial': (model_factorial, np.array([1.0, 1.0, 1.0])),
        'Logarithmic': (model_logarithmic, np.ones(3))
    }
    
    results = {}
    for model_name, (model_func, initial_params) in model_funcs.items():
        print(f"  尝试{model_name}模型...")
        
        try:
            # 根据模型设置适当的约束和边界
            if model_name == 'Linear':
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = [(0.1, 0.8) for _ in range(len(initial_params))]
            elif model_name == 'Exponential':
                bounds = [(-0.5, 0.5), (-10, 10), (-10, 10)]
                constraints = None
            elif model_name == 'Factorial':
                bounds = [(0.1, 3), (0.1, 3), (0.1, 3)]
                constraints = None
            else:
                constraints = None
                bounds = None
                
            # 优化模型参数
            res = minimize(
                objective_function, 
                initial_params, 
                args=(model_func, (FC, AP, SE), y),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
            
            # 计算预测值
            y_pred = model_func(res.x, FC, AP, SE)
            
            # 检查预测值是否有效
            if np.isnan(y_pred).any() or np.isinf(y_pred).any():
                print(f"  - {model_name}模型产生了无效值，跳过...")
                continue
                
            # 计算相关系数和p值
            try:
                corr, p_value = pearsonr(y_pred, y)
                r2 = r2_score(y, y_pred)
                
                results[model_name] = {
                    'params': res.x,
                    'corr': corr,
                    'p_value': p_value,
                    'r2': r2,
                    'mse': res.fun,
                    'y_pred': y_pred
                }
                print(f"  - {model_name}模型: r = {corr:.3f}, p = {p_value:.4f}, R² = {r2:.3f}")
            except Exception as e:
                print(f"  - {model_name}模型统计计算错误: {str(e)}")
                
        except Exception as e:
            print(f"  - {model_name}模型优化失败: {str(e)}")
    
    if not results:
        print(f"  所有模型都失败，使用默认线性模型")
        # 创建一个简单的默认模型
        params = np.array([0.33, 0.33, 0.34])
        y_pred = model_linear(params, FC, AP, SE)
        results['Default_Linear'] = {
            'params': params,
            'corr': 0,
            'p_value': 1,
            'r2': 0,
            'mse': np.mean((y_pred - y)**2),
            'y_pred': y_pred
        }
    
    return results

# 5. 确定要分析的AOI列
# 获取除了File、LOD和LOD参数外的所有列名作为AOI类型
aoi_columns = [col for col in analysis_df.columns 
               if col not in ['File', 'LOD', 'Feature_Complexity', 'Appearance', 'Semantics']]

print(f"找到的AOI列: {aoi_columns}")

# 6. 对每个AOI运行模型评估
all_model_results = {}
for aoi in aoi_columns:
    print(f"正在分析 {aoi}...")
    all_model_results[aoi] = evaluate_models(aoi)

# 7. 找出每个AOI的最佳模型
best_models = {}
for aoi in aoi_columns:
    # 按相关系数、p值和R2排序
    models = all_model_results[aoi]
    # 优先考虑统计显著性
    significant_models = {name: data for name, data in models.items() 
                         if data['p_value'] < 0.05}
    
    if significant_models:
        # 在显著模型中选择R2最高的
        best_model = max(significant_models.items(), key=lambda x: x[1]['r2'])
    else:
        # 如果没有显著模型，选择相关系数最高的
        best_model = max(models.items(), key=lambda x: x[1]['corr'])
    
    best_models[aoi] = best_model

# 8. 可视化最佳模型结果
try:
    plt.figure(figsize=(20, 15))
    rows = (len(aoi_columns) + 2) // 3  # 每行3个图，计算需要的行数
    for i, aoi in enumerate(aoi_columns):
        plt.subplot(rows, 3, i+1)
        
        model_name, model_data = best_models[aoi]
        y_true = analysis_df[aoi]
        y_pred = model_data['y_pred']
        
        # 散点图：实际vs预测
        plt.scatter(y_true, y_pred, alpha=0.7)
        
        # 添加理想预测线
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        
        # 添加标签（如果数据点不多）
        if len(y_true) < 30:
            for j, txt in enumerate(analysis_df['LOD']):
                plt.annotate(txt, (y_true[j], y_pred[j]), fontsize=8)
        
        # 添加统计信息
        r2 = model_data['r2']
        corr = model_data['corr']
        p_val = model_data['p_value']
        sig_symbol = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")
        
        plt.title(f'Best Model for {aoi}: {model_name}\n' + 
                f'R² = {r2:.3f}, r = {corr:.3f}, p = {p_val:.4f}{sig_symbol}')
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('best_lod_models.png')
    print("图表已保存为 'best_lod_models.png'")
except Exception as e:
    print(f"绘图出错: {str(e)}")

# 9. 打印最佳模型参数和详细结果
print("\n最优LOD参数模型结果:")
for aoi in aoi_columns:
    model_name, model_data = best_models[aoi]
    params = model_data['params']
    corr = model_data['corr']
    p_val = model_data['p_value']
    r2 = model_data['r2']
    
    print(f"\n{aoi}:")
    print(f"  - 最佳模型: {model_name}")
    print(f"  - 相关系数: r = {corr:.3f} (p = {p_val:.4f})")
    print(f"  - 决定系数: R² = {r2:.3f}")
    
    # 根据模型类型打印参数解释
    if model_name == 'Linear' or model_name == 'Default_Linear':
        print(f"  - 参数: FC权重 = {params[0]:.3f}, AP权重 = {params[1]:.3f}, SE权重 = {params[2]:.3f}")
        print(f"  - 公式: {params[0]:.3f}*FC + {params[1]:.3f}*AP + {params[2]:.3f}*SE")
    elif model_name == 'Quadratic':
        print(f"  - 参数: FC = {params[0]:.3f}, FC² = {params[1]:.3f}, AP = {params[2]:.3f}, SE = {params[3]:.3f}")
        print(f"  - 公式: {params[0]:.3f}*FC + {params[1]:.3f}*FC² + {params[2]:.3f}*AP + {params[3]:.3f}*SE")
    elif model_name == 'Interaction':
        print(f"  - 参数: FC = {params[0]:.3f}, AP = {params[1]:.3f}, SE = {params[2]:.3f}, " +
              f"FC*AP = {params[3]:.3f}, FC*SE = {params[4]:.3f}, AP*SE = {params[5]:.3f}")
        print(f"  - 公式: {params[0]:.3f}*FC + {params[1]:.3f}*AP + {params[2]:.3f}*SE + " +
              f"{params[3]:.3f}*FC*AP + {params[4]:.3f}*FC*SE + {params[5]:.3f}*AP*SE")
    elif model_name == 'Exponential':
        print(f"  - 参数: FC指数 = {params[0]:.3f}, AP权重 = {params[1]:.3f}, SE权重 = {params[2]:.3f}")
        print(f"  - 公式: exp({params[0]:.3f}*FC) + {params[1]:.3f}*AP + {params[2]:.3f}*SE")
    elif model_name == 'Factorial':
        print(f"  - 参数: FC指数 = {params[0]:.3f}, AP指数 = {params[1]:.3f}, SE指数 = {params[2]:.3f}")
        print(f"  - 公式: FC^{params[0]:.3f} * AP^{params[1]:.3f} * SE^{params[2]:.3f}")
    elif model_name == 'Logarithmic':
        print(f"  - 参数: FC权重 = {params[0]:.3f}, AP权重 = {params[1]:.3f}, SE权重 = {params[2]:.3f}")
        print(f"  - 公式: {params[0]:.3f}*log(1+FC) + {params[1]:.3f}*log(1+AP) + {params[2]:.3f}*log(1+SE)")

# 10. 保存结果到CSV
try:
    model_results_df = pd.DataFrame({
        'AOI': [],
        'Best_Model': [],
        'Correlation': [],
        'P_Value': [],
        'R_Squared': [],
        'Formula': []
    })

    for aoi in aoi_columns:
        model_name, model_data = best_models[aoi]
        params = model_data['params']
        
        # 构建公式文本
        if model_name == 'Linear' or model_name == 'Default_Linear':
            formula = f"{params[0]:.3f}*FC + {params[1]:.3f}*AP + {params[2]:.3f}*SE"
        elif model_name == 'Quadratic':
            formula = f"{params[0]:.3f}*FC + {params[1]:.3f}*FC² + {params[2]:.3f}*AP + {params[3]:.3f}*SE"
        elif model_name == 'Interaction':
            formula = f"{params[0]:.3f}*FC + {params[1]:.3f}*AP + {params[2]:.3f}*SE + " + \
                    f"{params[3]:.3f}*FC*AP + {params[4]:.3f}*FC*SE + {params[5]:.3f}*AP*SE"
        elif model_name == 'Exponential':
            formula = f"exp({params[0]:.3f}*FC) + {params[1]:.3f}*AP + {params[2]:.3f}*SE"
        elif model_name == 'Factorial':
            formula = f"FC^{params[0]:.3f} * AP^{params[1]:.3f} * SE^{params[2]:.3f}"
        elif model_name == 'Logarithmic':
            formula = f"{params[0]:.3f}*log(1+FC) + {params[1]:.3f}*log(1+AP) + {params[2]:.3f}*log(1+SE)"
        
        # 添加到结果数据框
        new_row = pd.DataFrame({
            'AOI': [aoi],
            'Best_Model': [model_name],
            'Correlation': [model_data['corr']],
            'P_Value': [model_data['p_value']],
            'R_Squared': [model_data['r2']],
            'Formula': [formula]
        })
        
        model_results_df = pd.concat([model_results_df, new_row], ignore_index=True)

    # 保存到CSV
    model_results_df.to_csv('lod_best_models_results.csv', index=False)
    print("\n结果已保存到 'lod_best_models_results.csv'")
except Exception as e:
    print(f"保存CSV出错: {str(e)}")