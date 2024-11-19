import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
import matplotlib as mpl
import numpy as np

# Set English font
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_train_test_results(input_dir):
    results = {}
    for filename in os.listdir(input_dir):
        if filename.startswith('train_test_results_') and filename.endswith('.xlsx'):
            course_type = filename[len('train_test_results_'):-5]
            file_path = os.path.join(input_dir, filename)
            df = pd.read_excel(file_path, sheet_name=None)
            results[course_type] = df
    return results

def plot_accuracy_comparison(accuracies, course_type):
    plt.figure(figsize=(10, 6))
    semesters = list(accuracies.keys())
    values = list(accuracies.values())
    plt.bar(semesters, values)
    plt.title(f'{course_type} - Accuracy Comparison for Different Semesters')
    plt.xlabel('Number of Semesters')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    plt.show()

def plot_accuracy_trend(accuracies, output_dir):
    plt.figure(figsize=(16, 10))
    course_type_mapping = {
        '通识必修课': 'General Education Required Courses',
        '学科基础必修课': 'Discipline Foundation Required Courses',
        '专业必修课': 'Major Required Courses',
        '集中实践教学环节': 'Concentrated Practical Teaching Segment',
        '专业选修课': 'Major Elective Courses',
        '通识公共选修课': 'General Education Public Elective Courses'
    }
    
    for course_type, acc in accuracies.items():
        semesters = list(acc.keys())
        values = list(acc.values())
        plt.plot(semesters, values, marker='o', label=course_type_mapping.get(course_type, course_type))
    
    plt.title('Accuracy Trend for Different Course Types', fontsize=24)
    plt.xlabel('Number of Semesters', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.ylim(0, 1)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig(os.path.join(output_dir, 'accuracy_trend.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_average_accuracy(accuracies, output_dir):
    course_type_mapping = {
        '通识必修课': 'General Education Required Courses',
        '学科基础必修课': 'Discipline Foundation Required Courses',
        '专业必修课': 'Major Required Courses',
        '集中实践教学环节': 'Concentrated Practical Teaching Segment',
        '专业选修课': 'Major Elective Courses',
        '通识公共选修课': 'General Education Public Elective Courses'
    }
    
    avg_accuracies = {course_type_mapping.get(course_type, course_type): sum(acc.values()) / len(acc) 
                      for course_type, acc in accuracies.items()}
    
    plt.figure(figsize=(16, 10))
    course_types = list(avg_accuracies.keys())
    values = list(avg_accuracies.values())
    
    plt.bar(course_types, values)
    plt.title('Average Accuracy for Different Course Types', fontsize=24)
    plt.xlabel('Course Type', fontsize=20)
    plt.ylabel('Average Accuracy', fontsize=20)
    plt.ylim(0, 1)
    
    for i, v in enumerate(values):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=16)
    
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, title, output_dir):
    course_type_mapping = {
        '通识必修课': 'General Education Required Courses',
        '学科基础必修课': 'Discipline Foundation Required Courses',
        '专业必修课': 'Major Required Courses',
        '集中实践教学环节': 'Concentrated Practical Teaching Segment',
        '专业选修课': 'Major Elective Courses',
        '通识公共选修课': 'General Education Public Elective Courses'
    }
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 30})
    
    # Translate the title to English
    for ch_name, en_name in course_type_mapping.items():
        if ch_name in title:
            title = title.replace(ch_name, en_name)
            break
    
    plt.title(title, fontsize=34, pad=20)  # Increased padding for more space
    plt.ylabel('Actual', fontsize=30)
    plt.xlabel('Predicted', fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.savefig(os.path.join(output_dir, f'{title}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def load_prediction_results(input_dir):
    results = {}
    for filename in os.listdir(input_dir):
        if filename.startswith('prediction_results_') and filename.endswith('.xlsx'):
            course_type = filename[len('prediction_results_'):-5]
            file_path = os.path.join(input_dir, filename)
            df = pd.read_excel(file_path, sheet_name=None)
            results[course_type] = df
    return results

def plot_course_type_accuracy(accuracies, output_dir):
    course_type_mapping = {
        '通识必修课': 'General Education Required Courses',
        '学科基础必修课': 'Discipline Foundation Required Courses',
        '专业必修课': 'Major Required Courses',
        '集中实践教学环节': 'Concentrated Practical Teaching Segment',
        '专业选修课': 'Major Elective Courses',
        '通识公共选修课': 'General Education Public Elective Courses'
    }
    
    for course_type, acc in accuracies.items():
        plt.figure(figsize=(14, 8))
        semesters = list(acc.keys())
        values = list(acc.values())
        plt.bar(semesters, values)
        plt.title(f'{course_type_mapping.get(course_type, course_type)}\nAccuracy Comparison for Different Semesters', fontsize=24)
        plt.xlabel('Number of Semesters', fontsize=20)
        plt.ylabel('Accuracy', fontsize=20)
        plt.ylim(0, 1)
        plt.tick_params(axis='both', which='major', labelsize=18)
        for i, v in enumerate(values):
            plt.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=16)
        plt.savefig(os.path.join(output_dir, f'{course_type}_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_combined_confusion_matrix(confusion_matrices, output_dir):
    # 课程类型映射
    course_type_mapping = {
        '通识必修课': 'GER',
        '学科基础必修课': 'DF',
        '专业必修课': 'MR',
        '集中实践教学环节': 'PT',
        '专业选修课': 'ME',
        '通识公共选修课': 'GEE'
    }
    
    # 标签映射
    labels = ['Not Meet', 'Meet']
    
    fig, axes = plt.subplots(2, 3, figsize=(40, 34))
    fig.suptitle('Confusion Matrices for Different Course Types', fontsize=56, y=0.98)
    
    # 使用一个集合来跟踪已绘制的课程类型
    plotted_course_types = set()
    
    for (course_type, cm) in confusion_matrices.items():
        # 检查课程类型是否已经绘制
        if course_type in plotted_course_types:
            continue
        
        # 确保不超过子图数量
        if len(plotted_course_types) >= len(axes.flatten()):
            print("Warning: More course types than available subplots.")
            break
        
        # 使用seaborn绘制热力图
        ax = axes.flatten()[len(plotted_course_types)]  # 获取下一个子图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                     xticklabels=labels, yticklabels=labels,
                     annot_kws={"size": 30})
        
        # 使用课程类型的英文缩写
        course_type_en = course_type_mapping.get(course_type, course_type)
        ax.set_title(course_type_en, fontsize=38, pad=20)
        
        ax.set_ylabel('Actual', fontsize=34)
        ax.set_xlabel('Predicted', fontsize=34)
        ax.tick_params(axis='both', which='major', labelsize=28)
        
        # 调整标签的旋转角度以提高可读性
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels, rotation=0)
        
        # 添加到已绘制集合中
        plotted_course_types.add(course_type)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # 生成新的图片文件名
    plt.savefig(os.path.join(output_dir, 'combined_confusion_matrices_new.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_semester_confusion_matrices(results, output_dir):
    """
    为每个课程类型生成 2 到 5 学期的混淆矩阵，并将它们汇总到一个大图中
    """
    # 课程类型映射
    course_type_mapping = {
        '通识必修课': 'GER',
        '学科基础必修课': 'DF',
        '专业必修课': 'MR',
        '集中实践教学环节': 'PT',
        '专业选修课': 'ME',
        '通识公共选修课': 'GEE'
    }
    
    # 标签映射
    labels = ['Not Meet', 'Meet']
    
    # 创建一个大图
    fig, axes = plt.subplots(6, 4, figsize=(40, 60))  # 6个课程类型，4个学期
    fig.suptitle('Confusion Matrices for Different Course Types (Semesters 2-5)', fontsize=64, y=0.98)
    
    for i, (course_type, data) in enumerate(results.items()):
        for semester in range(2, 6):  # 2-5学期
            sheet_name = f'前{semester}学期'
            if sheet_name in data:
                df = data[sheet_name]
                actual_column = '实际值' if '实际值' in df.columns else 'Actual'
                predicted_column = '预测值' if '预测值' in df.columns else 'Predicted'
                
                y_true = df[actual_column]
                y_pred = df[predicted_column]
                
                cm = confusion_matrix(y_true, y_pred)
                
                # 使用seaborn绘制热力图
                ax = axes[i, semester - 2]  # 选择对应的子图
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=labels, yticklabels=labels,
                           annot_kws={"size": 30})
                
                # 设置标题和标签
                course_type_en = course_type_mapping.get(course_type, course_type)
                ax.set_title(f'{course_type_en} - Semester {semester}', fontsize=32, pad=20)
                ax.set_ylabel('Actual', fontsize=28)
                ax.set_xlabel('Predicted', fontsize=28)
                ax.tick_params(axis='both', which='major', labelsize=24)
                
                # 调整标签的旋转角度
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=24)
                ax.set_yticklabels(labels, rotation=0, fontsize=24)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # 保存大图
    plt.savefig(os.path.join(output_dir, 'combined_confusion_matrices_all_courses.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_course_unqualified_distribution(data, output_dir):
    """绘制课程不及格分布图"""
    plt.figure(figsize=(12, 6))
    
    # 使用淡雅的配色方案
    colors = ['#E6E6FA', '#F0F8FF', '#F5F5F5', '#FFF0F5', '#F0FFF0', '#F8F8FF']
    
    # 绘制条形图
    bars = plt.bar(data.index, data.values, color=colors[:len(data)])
    
    plt.title('Course Unqualified Distribution', fontsize=14)
    plt.xlabel('Course Name', fontsize=12)
    plt.ylabel('Number of Unqualified Students', fontsize=12)
    
    # 旋转x轴标签以提高可读性
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'course_unqualified_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def visualize_results(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results = load_train_test_results(input_dir)
    
    # 添加处理不及格分布的代码
    unqualified_counts = {}
    for course_type, data in results.items():
        for sheet_name, df in data.items():
            if not sheet_name.endswith('分类报告'):
                if '实际值' in df.columns:
                    unqualified_count = (df['实际值'] == 0).sum()
                elif 'Actual' in df.columns:
                    unqualified_count = (df['Actual'] == 0).sum()
                else:
                    continue
                
                if course_type not in unqualified_counts:
                    unqualified_counts[course_type] = unqualified_count
                else:
                    unqualified_counts[course_type] += unqualified_count
    
    # 创建不及格分布的Series并绘图
    unqualified_series = pd.Series(unqualified_counts)
    plot_course_unqualified_distribution(unqualified_series, output_dir)
    
    accuracies = {course_type: {} for course_type in results.keys()}
    confusion_matrices = {}

    for course_type, data in results.items():
        for sheet_name, df in data.items():
            if sheet_name.endswith('分类报告'):
                continue
            
            try:
                num_semesters = int(sheet_name[1]) if sheet_name.startswith('S') else int(sheet_name[1])
                
                # 检查列名并使用正确的列名
                if '实际值' in df.columns:
                    actual_column = '实际值'
                    predicted_column = '预测值'
                elif 'Actual' in df.columns:
                    actual_column = 'Actual'
                    predicted_column = 'Predicted'
                else:
                    print(f"Warning: Cannot find actual/predicted columns in sheet {sheet_name}")
                    print(f"Available columns: {df.columns.tolist()}")
                    continue
                
                y_true = df[actual_column]
                y_pred = df[predicted_column]
                
                accuracy = accuracy_score(y_true, y_pred)
                accuracies[course_type][num_semesters] = accuracy
                
                cm = confusion_matrix(y_true, y_pred)
                # 确保每个课程类型只添加一次混淆矩阵
                if course_type not in confusion_matrices:
                    confusion_matrices[course_type] = cm
                else:
                    confusion_matrices[course_type] += cm
                
                print(f"{course_type} - First {num_semesters} semesters accuracy: {accuracy:.4f}")
                print(f"Actual qualified: {sum(y_true)}, Predicted qualified: {sum(y_pred)}")
                print(f"Total samples: {len(y_true)}")
                print("Confusion matrix:")
                print(cm)
                print("\n")
                
            except Exception as e:
                print(f"Error processing sheet {sheet_name} for {course_type}: {str(e)}")
                print(f"Sheet columns: {df.columns.tolist()}")
                continue
    
    # 确保所有课程类型都被处理
    all_course_types = ['GER', 'DF', 'MR', 'PT', 'ME', 'GEE']
    for course_type in all_course_types:
        if course_type not in confusion_matrices:
            confusion_matrices[course_type] = np.zeros((2, 2))  # 创建一个空的混淆矩阵
    
    if accuracies and any(accuracies.values()):
        plot_accuracy_trend(accuracies, output_dir)
        plot_course_type_accuracy(accuracies, output_dir)
        plot_combined_confusion_matrix(confusion_matrices, output_dir)
        plot_average_accuracy(accuracies, output_dir)
        plot_semester_confusion_matrices(results, output_dir)
        
        # Print final accuracy data
        print("\nFinal Accuracy Data:")
        for course_type, acc in accuracies.items():
            print(f"{course_type}:")
            for semester, accuracy in acc.items():
                print(f"  First {semester} semesters: {accuracy:.4f}")
    else:
        print("No valid accuracy data to plot")

if __name__ == "__main__":
    input_dir = 'train_test_results'
    output_dir = 'visualization_results'
    visualize_results(input_dir, output_dir)
