import pandas as pd
import os

def process_feature_map(input_file, output_file):
    print(f"Processing file: {input_file}")
    # 读取CSV文件
    df = pd.read_csv(input_file)
    print(f"Original DataFrame:\n{df.head()}")
    
    # 处理'Feature'列
    def clean_feature(feature):
        print(f"Cleaning feature: {feature}")
        # 移除括号和引号，分割字符串
        parts = feature.strip("()' ").split(',')
        # 返回第一个部分（课程名称）
        cleaned = parts[0].strip("' ")
        print(f"Cleaned feature: {cleaned}")
        return cleaned
    
    # 应用清理函数
    df['Feature'] = df['Feature'].apply(clean_feature)
    print(f"Processed DataFrame:\n{df.head()}")
    
    # 保存处理后的数据
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Processed file saved as {output_file}")

# 主函数
def main():
    input_dir = 'feature_importance_results'
    output_dir = 'processed_feature_importance_results'
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.startswith('feature_index_map_') and filename.endswith('.csv'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, f'processed_{filename}')
            process_feature_map(input_file, output_file)

if __name__ == "__main__":
    main()
