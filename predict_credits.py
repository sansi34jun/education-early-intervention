import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import seaborn as sns
import matplotlib.font_manager as fm
import shap

# 在文件开头添加课程类型映射
COURSE_TYPE_MAP = {
    '通识必修课': 'GER',
    '学科基础必修课': 'DF',
    '专业必修课': 'MR',
    '集中实践教学环节': 'PT',
    '专业选修课': 'ME',
    '通识公共选修课': 'GEE'
}

# 更新课程名称映射字典，添加更多课程和更准确的英文名称
COURSE_NAME_MAP = {
    # 通识必修课 (GER)
    '中国近现代史纲要': 'Modern Chinese History',
    '习近平新时代中国特色社会主义思想概论': 'Xi Jinping Thought',
    '体育': 'PE',  # 简化体育课程名称
    '创新创业教育': 'Innovation',
    '大学军事理论': 'Military Theory',
    '大学日语': 'Japanese',
    '大学英语': 'English',
    '大学计算机基础及实验': 'Computer Basics',
    '形势与政策': 'Politics',
    '思想道德修养与法律基础': 'Ethics and Law',
    '毛泽东思想和中国特色社会主义理论体系概论': 'Mao Thought',
    '职业生涯规划与就业指导': 'Career Planning',
    '马克思主义基本原理概论': 'Marxism',
    
    # 学科基础必修课 (DF)
    '高等数学': 'Advanced Math',
    '线性代数': 'Linear Algebra',
    '概率论与数理统计': 'Probability Stats',
    '离散数学': 'Discrete Math',
    '数据结构': 'Data Structure',
    '计算机组成原理': 'Computer Org',
    '操作系统': 'OS',
    '计算机网络': 'Networks',
    '数据库原理': 'Database',
    '软件工程': 'Software Eng',
    '人工智能': 'AI',
    '编译原理': 'Compiler',
    '算法设计与分析': 'Algorithm',
    
    # 专业必修课 (MR)
    '程序设计基础': 'Programming',
    'Java程序设计': 'Java',
    'Python程序设计': 'Python',
    '微机原理与接口技术': 'Microcomputer',
    '计算机图形学': 'Graphics',
    
    # 选修课 (ME)
    '数字图像���理': 'Image Processing',
    '嵌入式系统': 'Embedded Sys',
    '云计算技术': 'Cloud Computing',
    '大数据技术': 'Big Data',
    '机器学习': 'ML',
    '深度学习': 'DL',
    '网络安全': 'Security',
    
    # 添加新的专业课程映射
    '工程项目管理': 'Engineering Project Management',
    '微机原理与应用': 'Microcomputer Principle',  # 略微缩短以适应显示
    'EDA技术及应用': 'EDA Technology',
    '嵌入式系统': 'Embedded Systems',
    '信息论': 'Information Theory',
    '专业英语': 'Professional English',
    '计算机网络': 'Networks',
    '工程制图': 'Engineering Drawing',
    '半导体材料': 'Semiconductor Materials',
    '电磁场与电磁波': 'Electromagnetic Fields',
    '电子测量技术': 'Electronic Measurement',
    '物理光学': 'Physical Optics',
    '传感器原理及应用': 'Sensor Principles',
    '可编程控制器原理及应用': 'PLC Principles',  # 使用缩写以适应显示
    '自动控制原理': 'Control Principles',
    '虚拟仪器系统设计': 'Virtual Instrument',
}

def load_data(file_path):
    try:
        data = {}
        print(f"尝试加载文件: {file_path}")
        xlsx = pd.ExcelFile(file_path)
        for sheet_name in xlsx.sheet_names:
            print(f"读取工作表: {sheet_name}")
            df = pd.read_excel(xlsx, sheet_name=sheet_name, header=[0, 1], index_col=0)
            # 打印每个工作表的唯一课程类型
            unique_course_types = df.columns.get_level_values(0).unique()
            print(f"工作表 {sheet_name} 的唯一课程类型: {unique_course_types}")
            data[sheet_name] = df
        if not data:
            raise ValueError(f"无法从 {file_path} 加载数据。请确保文件存在且不为空。")
        print(f"成功加载数据，共 {len(data)} 个工作表")
        return data
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        return None

def prepare_data(df, num_semesters):
    print(f"准备前 {num_semesters} 学期的数据")
    print(f"DataFrame 形状: {df.shape}")
    
    semester_order = [
        '2020-2021-1', '2020-2021-2',
        '2021-2022-1', '2021-2022-2',
        '2022-2023-1', '2022-2023-2'
    ]
    
    valid_semesters = semester_order[:num_semesters]
    
    feature_columns = []
    for col in df.columns:
        if isinstance(col, tuple) and len(col) == 2:
            course, attribute = col
            if attribute == '成绩':
                semester_col = (course, '学期')
                if semester_col in df.columns:
                    if df[semester_col].isin(valid_semesters).any():
                        feature_columns.append(col)
    
    print(f"找到的特征列: {feature_columns}")
    
    if not feature_columns:
        print(f"警告: 没有找到适合的特征列（前 {num_semesters} 学期）。")
        return None, None
    
    X = df[feature_columns].fillna(0)
    
    target_column = next((col for col in df.columns if '前六学期是否达标' in str(col)), None)
    if target_column is None:
        print(f"警告: 无法找到目标变量列 '前六学期是否达标'。可用的列: {df.columns}")
        return None, None
    
    y = df[target_column].astype(int)
    
    print(f"选择的特征列数量: {len(feature_columns)}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def get_english_course_name(chinese_name):
    """将中文课程名转换为英文名"""
    # 首先尝试直接匹配
    if chinese_name in COURSE_NAME_MAP:
        base_name = COURSE_NAME_MAP[chinese_name]
    else:
        # 尝试部分匹配
        for cn, en in COURSE_NAME_MAP.items():
            if cn in chinese_name:
                base_name = en
                break
        else:
            return chinese_name  # 如果没有找到匹配，返回原始名称
    
    # 处理课程编号
    number_map = {
        'Ⅰ': '1', 'Ⅱ': '2', 'Ⅲ': '3', 'Ⅳ': '4', 'Ⅴ': '5', 'Ⅵ': '6',
        '①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5', '⑥': '6',
        '⑴': '1', '⑵': '2', '⑶': '3', '⑷': '4', '⑸': '5', '⑹': '6',
        '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6'
    }
    
    for roman, arabic in number_map.items():
        if roman in chinese_name:
            return f"{base_name} {arabic}"
    
    return base_name

def train_and_predict(X, y, model_type='rf', course_type=None):
    if X is None or y is None or X.empty or y.empty:
        print("警告: 训练数据为空")
        return None, None, None, None, None, None, None
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'svm':
        model = SVC(probability=True, random_state=42)
    elif model_type == 'lr':
        model = LogisticRegression(random_state=42)
    elif model_type == 'dt':
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == 'adaboost':
        model = AdaBoostClassifier(n_estimators=100, random_state=42)
    elif model_type == 'xgboost':
        model = XGBClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("不支持的模型类型")
    
    model.fit(X_train, y_train)
    
    # 在测试集上评估模型
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    
    print(f"{model_type} 模型测试集率: {test_accuracy}")
    print(f"{model_type} 模型测试集召回率: {test_recall}")
    print(f"{model_type} 模型测试集F1分数: {test_f1}")
    
    # SHAP 分析
    supported_models = ['rf', 'xgboost']
    if model_type in supported_models:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # 准备特征名称（只使用英文名）
            feature_names = []
            for col in X.columns:
                if isinstance(col, tuple):
                    course_name = col[0]
                    english_name = get_english_course_name(course_name)
                    feature_names.append(english_name)
                else:
                    feature_names.append(str(col))
            
            # 确保 shap_values 是正确的形状
            if isinstance(shap_values, list):
                shap_vals = np.abs(shap_values[1]).mean(0)
            else:
                shap_vals = np.abs(shap_values).mean(0)
            
            # 创建更大的图形
            fig = plt.figure(figsize=(30, 20))
            ax = fig.add_subplot(111, projection='3d')
            
            # 准备数据
            x = np.arange(len(feature_names)) * 4  # 增加间距
            y = model.feature_importances_
            z = shap_vals
            
            # 设置柱的宽度
            dx = dy = 0.8
            
            # 创建更丰富的颜色映射
            colors = plt.cm.Set3(np.linspace(0, 1, len(x)))  # 使用Set3颜色方案，颜色更加丰富且易区分
            
            # 创建3D柱状图
            bars = ax.bar3d(x, y, np.zeros_like(y), 
                           dx * np.ones_like(y), 
                           dy * np.ones_like(y), 
                           z, 
                           color=colors,
                           alpha=0.9, shade=True)  # 增加alpha值使颜色更鲜明
            
            # 设置坐标轴标签（移除换行符）
            ax.set_xlabel('Course Names', fontsize=16, labelpad=120)  # 增加labelpad使标签下移
            ax.set_ylabel('Feature Importance', fontsize=16, labelpad=20)  # 移除换行符
            ax.set_zlabel('SHAP Values', fontsize=16, labelpad=20)  # 移除换行符
            
            # 设置刻度
            ax.set_xticks(x)
            ax.set_xticklabels(feature_names, rotation=80, ha='right', fontsize=12)
            
            # 调整y轴刻度（使用0.1的间隔，延伸到1.0）
            y_ticks = np.arange(0, 1.1, 0.1)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f'{val:.1f}' for val in y_ticks], fontsize=12)
            
            # 调整z轴刻度（使用0.5的间隔）
            z_max = max(z)
            z_ticks = np.arange(0, z_max + 0.5, 0.5)
            ax.set_zticks(z_ticks)
            ax.set_zticklabels([f'{val:.1f}' for val in z_ticks], fontsize=12)
            
            # 添加z轴标签（移到左边并靠近轴）
            ax.text2D(-0.15, 0.5, 'SHAP Values', fontsize=16, rotation=90,
                     transform=ax.transAxes, va='center')
            
            # 使用英文课程类型缩写
            course_type_en = COURSE_TYPE_MAP.get(course_type, course_type)
            ax.set_title(f'Course Impact Analysis - {model_type.upper()}\n{course_type_en}', 
                        fontsize=20, pad=40)
            
            # 调整视角以更好地显示标签
            ax.view_init(elev=20, azim=45)
            
            # 调整x轴标签位置和角度，避免重叠
            for idx, tick in enumerate(ax.get_xticklabels()):
                tick.set_rotation_mode("anchor")
                tick.set_rotation(80)
                tick.set_ha('right')
                tick.set_va('top')
                if idx % 2:
                    tick.set_y(tick.get_position()[1] - 0.02)
            
            # 添加颜色条（使用相同的颜色方案）
            norm = plt.Normalize(z.min(), z.max())
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Set3, norm=norm)  # 使用Set3颜色方案
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.1)
            cbar.set_label('SHAP Values', fontsize=16, labelpad=20)
            cbar.ax.tick_params(labelsize=12)
            
            # 调整布局，给标签留出更多空间
            plt.tight_layout(rect=[0.1, 0.05, 0.95, 0.95])
            
            # 保存图片
            importance_output_dir = 'feature_importance_results'
            os.makedirs(importance_output_dir, exist_ok=True)
            plt.savefig(os.path.join(importance_output_dir, 
                       f'shap_feature_importance_{model_type}_{course_type_en}.png'),
                       bbox_inches='tight', dpi=300)
            plt.close()
            
            # 保存特征重要性数据
            feature_importance = pd.DataFrame({
                'Course': feature_names,
                'Importance': model.feature_importances_,
                'SHAP': shap_vals
            })
            
            feature_importance = feature_importance.sort_values(
                by='Importance', ascending=False
            )
            
            feature_importance.to_csv(
                os.path.join(importance_output_dir, 
                            f'shap_feature_importance_{model_type}_{course_type_en}.csv'),
                index=False,
                encoding='utf-8-sig'
            )
            
            print(f"3D feature importance plot and CSV have been saved in the {importance_output_dir} folder.")
        except Exception as e:
            print(f"SHAP analysis error: {str(e)}")
            print("Skipping SHAP analysis, continuing with other steps")
            import traceback
            traceback.print_exc()
    
    return model, test_accuracy, test_recall, test_f1, imputer, X_test, y_test

# 添加这个函数来获取课程类型
def get_course_type(course_name):
    for course_type, code in COURSE_TYPE_MAP.items():
        if course_type in course_name:
            return course_type
    print(f"未能识别课程类型: {course_name}")  # 添加这行来帮助调试
    return '未知类'

def process_course_type(df, course_type):
    # 在函数开始时就转换课程类型为英文
    course_type_en = COURSE_TYPE_MAP.get(course_type, course_type)
    print(f"Processing course type: {course_type_en}")
    
    results = {}
    train_test_results = {}
    model_comparisons = {}
    
    for num_semesters in range(2, 6):
        X, y = prepare_data(df, num_semesters)
        if X is None or y is None:
            continue
        
        model_results = {}
        for model_type in ['rf', 'svm', 'lr', 'dt', 'adaboost', 'xgboost']:
            # 使用英文课程类型缩写
            model, accuracy, recall, f1, imputer, X_test, y_test = train_and_predict(
                X, y, model_type, course_type_en  # 传递英文缩写
            )
            if model is None:
                continue
            
            model_results[model_type] = {
                'Accuracy': accuracy,
                'Recall': recall,
                'F1 Score': f1
            }
        
        # 选择最佳模型时也使用英文缩写
        if model_results:
            best_model_type = max(model_results, key=lambda k: model_results[k]['F1 Score'])
            best_model, best_accuracy, best_recall, best_f1, best_imputer, X_test, y_test = train_and_predict(
                X, y, best_model_type, course_type_en  # 这里也使用英文缩写
            )
            
            X_all = best_imputer.transform(X)
            predictions = best_model.predict(X_all)
            probabilities = best_model.predict_proba(X_all)[:, 1]
            
            results[num_semesters] = {
                'Course Type': course_type_en,  # 使用英文缩写
                'Semesters Used': num_semesters,
                'Best Model': best_model_type,
                'Model Accuracy': best_accuracy,
                'Model Recall': best_recall,
                'Model F1 Score': best_f1,
                'Predictions': predictions,
                'Probabilities': probabilities
            }
            
            train_test_results[num_semesters] = {
                'Course Type': course_type_en,  # 使用英文缩写
                'Semesters Used': num_semesters,
                'Best Model': best_model_type,
                'Model Accuracy': best_accuracy,
                'Model Recall': best_recall,
                'Model F1 Score': best_f1,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': best_model.predict(X_test)
            }
            
            model_comparisons[num_semesters] = model_results
    
    return results, train_test_results, model_comparisons

def save_results(results, data, output_dir):
    print(f"开始保存结果到目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    for course_type, course_results in results.items():
        output_file = os.path.join(output_dir, f'prediction_results_{course_type}.xlsx')
        
        # 检查是否有结果要保存
        if not course_results:
            print(f"Warning: No results to save for {course_type}")
            continue
            
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            at_least_one_sheet = False
            
            for num_semesters, result in course_results.items():
                # 使用原始课程类型作为键来获取数据
                original_course_type = next(k for k, v in COURSE_TYPE_MAP.items() if v == course_type)
                df_source = data.get(original_course_type)
                
                if df_source is not None:
                    target_column = next((col for col in df_source.columns if '前六学期是否达标' in str(col)), None)
                    if target_column:
                        df_results = pd.DataFrame({
                            'Student': df_source.index,
                            'Model Accuracy': result['Model Accuracy'],
                            'Predictions': result['Predictions'],
                            'Probabilities': result['Probabilities'],
                            'Actual Qualification': df_source[target_column] if target_column else None
                        })
                        
                        sheet_name = f'S{num_semesters}'
                        df_results.to_excel(writer, sheet_name=sheet_name, index=False)
                        at_least_one_sheet = True
            
            # 如果没有添加任何工作表，添加一个默认工作表
            if not at_least_one_sheet:
                pd.DataFrame({'Info': ['No data available']}).to_excel(writer, sheet_name='Info', index=False)
        
        print(f"已将 {course_type} 的预测结果保存到 {output_file}")
    
    print("保存结果完成")

def save_train_test_results(train_test_results, output_dir):
    print(f"开始保存训练测试结果到目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    for course_type, course_results in train_test_results.items():
        output_file = os.path.join(output_dir, f'train_test_results_{course_type}.xlsx')
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for num_semesters, result in course_results.items():
                df_results = pd.DataFrame({
                    '实际值': result['y_test'],
                    '预测值': result['y_pred'],
                })
                
                sheet_name = f'前{num_semesters}学期'
                df_results.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # 添加分类报告
                report = classification_report(result['y_test'], result['y_pred'], output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                df_report.to_excel(writer, sheet_name=f'{sheet_name}_分类报')
            
            if len(course_results) == 0:
                pd.DataFrame().to_excel(writer, sheet_name='空白', index=False)
                
        print(f"已将 {course_type} 的训练测试结果保存到 {output_file}")
    print("存训练测试果完成")

def save_model_comparisons(model_comparisons, output_dir):
    print(f"开始保存模型比较结果到目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'model_comparisons.xlsx')
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for course_type, semesters in model_comparisons.items():
            for num_semesters, models in semesters.items():
                df = pd.DataFrame(models).T
                df.index.name = '模型'
                df.to_excel(writer, sheet_name=f'{course_type}_前{num_semesters}学')
    print(f"已将模型比较结果保存到 {output_file}")

def create_comprehensive_comparison_table(model_comparisons, output_dir):
    print("Creating comprehensive comparison table")
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    # 收集所有结果
    for course_type, semesters in model_comparisons.items():
        for num_semesters, models in semesters.items():
            for model, metrics in models.items():
                row = {
                    'Semesters': num_semesters,
                    'Model': model,
                    'Accuracy': metrics['Accuracy'],
                    'Recall': metrics['Recall'],
                    'F1 Score': metrics['F1 Score']
                }
                all_results.append(row)
    
    # 创建DataFrame并按学期和模型分组算平均值
    df = pd.DataFrame(all_results)
    df_avg = df.groupby(['Semesters', 'Model']).agg({
        'Accuracy': 'mean',
        'Recall': 'mean',
        'F1 Score': 'mean'
    }).reset_index()
    
    # 创建数据透视表
    df_pivot = df_avg.pivot_table(
        values=['Accuracy', 'Recall', 'F1 Score'],
        index=['Semesters'],
        columns=['Model'],
        aggfunc='first'
    )
    
    # 重新排列列以保持一致的顺序
    new_columns = []
    for model in ['rf', 'svm', 'lr', 'dt', 'adaboost', 'xgboost']:
        new_columns.extend([(metric, model) for metric in ['Accuracy', 'Recall', 'F1 Score']])
    df_pivot = df_pivot.reindex(columns=new_columns)
    
    # 找出每个学期的最佳模型（基于F1分数）
    best_models = df_avg.loc[df_avg.groupby('Semesters')['F1 Score'].idxmax()]
    
    styler = df_pivot.style
    
    def highlight_best(x):
        best = best_models[best_models['Semesters'] == x.name]
        if not best.empty:
            best_model = best.iloc[0]['Model']
            return ['font-weight: bold' if col[1] == best_model else '' for col in x.index]
        return ['' for _ in x.index]
    
    styler.apply(highlight_best, axis=1)
    styler.format("{:.4f}")
    
    # 保存结果
    output_file = os.path.join(output_dir, 'comprehensive_model_comparison.xlsx')
    styler.to_excel(output_file, engine='openpyxl')
    
    csv_output = os.path.join(output_dir, 'comprehensive_model_comparison.csv')
    df_pivot.to_csv(csv_output)
    
    print(f"Comprehensive comparison table saved to {output_file}")
    print(f"CSV version saved to {csv_output}")

def ensure_unicode(text):
    if isinstance(text, bytes):
        return text.decode('utf-8')
    return str(text)

def main():
    try:
        print("Starting main program")
        data = load_data('student_credits_analysis.xlsx')
        if data is None:
            print("Could not load data, program terminated")
            return
        
        results = {}
        train_test_results = {}
        all_model_comparisons = {}
        
        for course_type, df in data.items():
            # 在这里转换课程类型为英文缩写
            course_type_en = COURSE_TYPE_MAP.get(course_type, course_type)
            print(f"Processing {course_type_en} data")
            
            course_results, course_train_test_results, model_comparisons = process_course_type(df, course_type)
            if course_results:
                results[course_type_en] = course_results  # 使用英文缩写作为键
                train_test_results[course_type_en] = course_train_test_results
                all_model_comparisons[course_type_en] = model_comparisons
            else:
                print(f"Warning: No valid prediction results for {course_type_en}")
        
        if not results:
            print("Error: No valid prediction results generated")
            return
        
        output_dir = 'prediction_results'
        save_results(results, data, output_dir)
        
        train_test_output_dir = 'train_test_results'
        save_train_test_results(train_test_results, train_test_output_dir)
        
        model_comparison_output_dir = 'model_comparisons'
        save_model_comparisons(all_model_comparisons, model_comparison_output_dir)
        
        create_comprehensive_comparison_table(all_model_comparisons, model_comparison_output_dir)
        
        print(f"All prediction results saved to {output_dir} directory")
        print(f"All training test results saved to {train_test_output_dir} directory")
        print(f"All model comparisons, charts, and comprehensive comparison table saved to {model_comparison_output_dir} directory")
    except Exception as e:
        print(f"Error occurred during program execution: {str(e)}")
        print("Error details:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

