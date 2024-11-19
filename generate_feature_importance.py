import pandas as pd
import xgboost as xgb
import numpy as np

def generate_xgboost_importance():
    # 假设我们有30个特征的重要性分数
    # 这里使用随机数据作为示例，您应该替换为实际的XGBoost模型的特征重要性
    np.random.seed(42)  # 设置随机种子以确保可重复性
    importance_scores = np.random.uniform(0, 1, 30)
    importance_scores = np.sort(importance_scores)[::-1]  # 降序排序
    
    # 创建DataFrame
    df = pd.DataFrame({
        'feature': range(30),
        'importance': importance_scores
    })
    
    # 保存到CSV文件
    df.to_csv('feature_importance_results/feature_importance_xgboost.csv', index=False)
    return df

if __name__ == "__main__":
    generate_xgboost_importance()
    print("Feature importance data has been generated!") 