import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def plot_course_type_overlap(input_file, output_dir):
    course_types = {
        '通识必修课': 'GER',
        '学科基础必修课': 'DF',
        '专业必修课': 'MR',
        '集中实践教学环节': 'PT',
        '专业选修课': 'ME',
        '通识公共选修课': 'GEE'
    }

    not_qualified = {}
    total_students = 0

    for ch_name, en_name in course_types.items():
        try:
            df = pd.read_excel(input_file, sheet_name=ch_name)
            student_id_col = df.columns[0]
            qualification_col = '前六学期是否达标'
            
            if student_id_col in df.columns and qualification_col in df.columns:
                not_qualified[en_name] = set(df[df[qualification_col] == 0][student_id_col])
                print(f"Found {len(not_qualified[en_name])} unqualified students in {ch_name}")
                total_students = max(total_students, len(df))
            else:
                print(f"Warning: Required columns not found in sheet {ch_name}")
        except Exception as e:
            print(f"Error reading sheet {ch_name}: {str(e)}")

    if not not_qualified:
        print("No data available for any course type.")
        return

    # 计算每个学生未达标的课程数量
    student_unqualified_count = Counter()
    for course, students in not_qualified.items():
        for student in students:
            student_unqualified_count[student] += 1

    # 为每个课程类型计算未达标课程数量的分布
    distribution = {course: [0] * 6 for course in course_types.values()}
    for course, students in not_qualified.items():
        for student in students:
            count = student_unqualified_count[student] - 1  # 索引从0开始
            distribution[course][count] += 1

    # 创建堆叠条形图
    fig, ax = plt.subplots(figsize=(18, 14))  # 进一步增加图表大小

    courses = list(course_types.values())
    bottom = np.zeros(len(courses))

    # 使用更协调的颜色方案
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#FFFF99']
    for i in range(6):
        values = [distribution[course][i] for course in courses]
        bars = ax.bar(courses, values, bottom=bottom, label=f'{i+1} courses', color=colors[i])
        
        # 在每个颜色块上添加数据标签
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bottom[j] + height/2,
                        f'{int(height)}',
                        ha='center', va='center', fontsize=14, color='black', fontweight='bold')
        
        bottom += values

    ax.set_ylabel('Number of Unqualified Students', fontsize=20)
    ax.set_xlabel('Course Types', fontsize=20)
    ax.set_title('Distribution of Unqualified Courses by Course Type', fontsize=24)
    ax.legend(title='Number of Unqualified Courses', fontsize=16, title_fontsize=18)
    
    # 增大刻度标签字体
    ax.tick_params(axis='both', which='major', labelsize=18)

    # 添加网格线以便于读取数据
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 在柱子顶部添加总数标签
    for i, course in enumerate(courses):
        total = sum(distribution[course])
        ax.text(i, total, f'{total}', ha='center', va='bottom', fontsize=18, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 保存图表
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "course_unqualified_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Stacked bar chart has been saved to {output_path}")

    # 生成分析文字
    analysis = generate_analysis(course_types.values(), not_qualified, total_students, distribution)
    print("\nAnalysis of Student Qualification Status by Course Type:")
    print(analysis)

def generate_analysis(course_types, not_qualified, total_students, distribution):
    analysis = []
    
    # 计算总体不合格率
    total_unqualified = len(set.union(*not_qualified.values()))
    overall_unqualification_rate = (total_unqualified / total_students) * 100
    
    analysis.append(f"Overall, {total_unqualified} out of {total_students} students ({overall_unqualification_rate:.1f}%) did not meet the requirements in at least one course type.")
    
    # 分析每种课程类型
    for course in course_types:
        unqualified_count = len(not_qualified[course])
        rate = unqualified_count / total_students * 100
        analysis.append(f"\nFor {course}:")
        analysis.append(f"- {unqualified_count} students ({rate:.1f}%) did not meet the requirements.")
        
        # 添加未达标课程数量的分布信息
        for i, count in enumerate(distribution[course]):
            if count > 0:
                percentage = count / unqualified_count * 100
                analysis.append(f"  - {count} students ({percentage:.1f}%) were unqualified in {i+1} course types.")

    # 总结
    analysis.append("\nIn conclusion:")
    if overall_unqualification_rate > 40:
        analysis.append("There is a concerning trend of high unqualification rates across course types. Immediate action may be needed to address this issue and provide additional support to students.")
    elif overall_unqualification_rate > 20:
        analysis.append("While many students are meeting the requirements, there is significant room for improvement across all course types.")
    else:
        analysis.append("Overall, the majority of students are meeting the requirements, but there is still room for improvement, especially in courses with higher unqualification rates.")
    
    return "\n".join(analysis)

# 使用示例
if __name__ == "__main__":
    input_file = 'student_credits_analysis.xlsx'  # 请确保这是正确的文件名
    output_dir = 'visualization_results'
    plot_course_type_overlap(input_file, output_dir)
