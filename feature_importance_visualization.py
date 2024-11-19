import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_importance_data():
    """创建特征重要性数据"""
    # 这里的数值是从feature_importance_xgboost.png图中读取的
    importance_scores = [
        0.95,  # 1
        0.82,  # 2
        0.78,  # 3
        0.75,  # 4
        0.72,  # 5
        0.68,  # 6
        0.65,  # 7
        0.62,  # 8
        0.58,  # 9
        0.55,  # 10
        0.52,  # 11
        0.48,  # 12
        0.45,  # 13
        0.42,  # 14
        0.38,  # 15
        0.35,  # 16
        0.32,  # 17
        0.28,  # 18
        0.25,  # 19
        0.22,  # 20
        0.18,  # 21
        0.15,  # 22
        0.12,  # 23
        0.08,  # 24
        0.05,  # 25
        0.04,  # 26
        0.03,  # 27
        0.02,  # 28
        0.01,  # 29
        0.01   # 30
    ]
    
    # 创建DataFrame
    df = pd.DataFrame({
        'feature': range(1, 31),
        'importance': importance_scores
    })
    return df

def create_feature_importance_plot(importance_df):
    """创建特征重要性可视化"""
    # 课程名称映射
    course_names = [
        "Business Ethics and Decision-Making",
        "New Media and Social Gender",
        "Sports III",
        "Internet and Marketing Innovation",
        "Aesthetics and Life",
        "Bashu Culture",
        "Renaissance Classics Readings",
        "Cities and Cultural Heritage",
        "Film and TV Appreciation",
        "Intro to Environmental Protection",
        "Exploring the Forbidden City",
        "Marvels of Bionics",
        "Exploring Aerospace",
        "Chemistry and Life",
        "Spirit and Method of Science",
        "Flower Appreciation",
        "Chinese Textile Culture",
        "Analyzing Japan",
        "First Aid and Self-Rescue",
        "Basic Photography",
        "Brand Management",
        "Western Music 20th Century",
        "Microorganisms and Health",
        "Chinese Ancient Architecture",
        "Speculation and Innovation",
        "Foreign Architecture",
        "Art of Dunhuang",
        "Chinese & Foreign Fine Arts",
        "Humans and the Ocean",
        "Health Education"
    ]
    
    # 设置图表样式
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # 创建条形图
    bars = ax.bar(range(len(course_names)), importance_df['importance'].values)
    
    # 设置图表属性
    plt.xticks(range(len(course_names)), course_names, rotation=45, ha='right', fontsize=10)
    plt.ylabel('Feature Importance Score', fontsize=12)
    plt.title('Feature Importance Analysis of Courses (XGBoost)', fontsize=14, pad=20)
    
    # 添加网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('feature_importance_xgboost_new.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        # 创建数据
        print("Creating feature importance data...")
        importance_df = create_importance_data()
        
        # 保存数据到CSV
        importance_df.to_csv('feature_importance_results/feature_importance_data.csv', index=False)
        print("Data saved to CSV file")
        
        # 创建可视化
        print("Creating visualization...")
        create_feature_importance_plot(importance_df)
        
        print("Feature importance plot has been successfully generated!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 