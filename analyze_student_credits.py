import sys
print(sys.executable)
print(sys.path)

import os
import pandas as pd
from zipfile import BadZipFile
from collections import defaultdict

# 检查pandas是否正确安装
try:
    pd.__version__
except AttributeError:
    print("pandas 未正确安装，请尝试重新安装：")
    print("pip install --upgrade pandas")
    exit(1)

# 定义课程性质及其学分要求
CREDIT_REQUIREMENTS = {
    '通识必修课': 39.5,
    '学科基础必修课': 45.0,
    '专业必修课': 23.5,
    '集中实践教学环节': 13.0,
    '专业选修课': 18.5,
    '通识公共选修课': 10.0
}

def process_student_data(directory_path):
    # 存储学生的各类课程信息
    student_courses = defaultdict(lambda: defaultdict(dict))
    student_credits = defaultdict(lambda: defaultdict(float))
    student_credits_first_six = defaultdict(lambda: defaultdict(float))
    
    # 定义有效的学期（8个学期）
    valid_semesters = [
        '2020-2021-1', '2020-2021-2',
        '2021-2022-1', '2021-2022-2',
        '2022-2023-1', '2022-2023-2',
        '2023-2024-1', '2023-2024-2'
    ]
    
    first_six_semesters = valid_semesters[:6]
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            file_path = os.path.join(directory_path, filename)
            engine = 'openpyxl' if filename.endswith('.xlsx') else 'xlrd'
            try:
                df = pd.read_excel(file_path, engine=engine)
            except BadZipFile:
                print(f"警告: 文件 {filename} 可能已损坏或不是有效的Excel文件。跳过此文件。")
                continue
            except Exception as e:
                print(f"警告: 读取文件 {filename} 时发生错误: {str(e)}。跳过此文件。")
                continue
            
            # 筛选有效的学期
            df = df[df['开课学期'].isin(valid_semesters)]
            
            # 计算每个学生的学分和课程信息
            for _, row in df.iterrows():
                course_type = row['课程性质']
                student_name = row['姓名']
                course_name = row['课程名称']
                semester = row['开课学期']
                
                # 如果课程已存在，只保留成绩最高的一次
                if course_name in student_courses[student_name][course_type]:
                    if row['总成绩'] <= student_courses[student_name][course_type][course_name]['成绩']:
                        continue
                
                student_courses[student_name][course_type][course_name] = {
                    '学期': semester,
                    '成绩': row['总成绩'],
                    '学分': row['学分'],
                    '计入总分': False
                }
                
                # 只有成绩达到60分以上才计入总学分，且每门课只计算一次
                if row['总成绩'] >= 60 and not student_courses[student_name][course_type][course_name]['计入总分']:
                    student_credits[student_name][course_type] += row['学分']
                    student_courses[student_name][course_type][course_name]['计入总分'] = True
                    
                    # 计算前六学期的学分
                    if semester in first_six_semesters:
                        student_credits_first_six[student_name][course_type] += row['学分']

    # 创建结果DataFrame
    result_dfs = {}
    for course_type in CREDIT_REQUIREMENTS.keys():
        # 获取所有课程名称
        all_courses = set()
        for student in student_courses.values():
            all_courses.update(student[course_type].keys())
        all_courses = sorted(all_courses)

        # 修改创建列的部分
        columns = pd.MultiIndex.from_product([all_courses, ['学期', '成绩', '学分']])
        additional_columns = [('前六学期总学分', ''), ('前六学期是否达标', ''), ('总学分', ''), ('是否达标', '')]
        columns = columns.append(pd.MultiIndex.from_tuples(additional_columns))

        # 打印列数
        print(f"Total number of columns: {len(columns)}")

        # 创建多级列索引
        columns = pd.MultiIndex.from_product([all_courses, ['学期', '成绩', '学分']])
        columns = columns.append(pd.MultiIndex.from_tuples([
            ('前六学期总学分', ''),
            ('前六学期是否达标', ''),
            ('总学分', ''),
            ('是否达标', '')
        ]))
        
        # 创建DataFrame
        data = []
        for name, courses in student_courses.items():
            row = []
            for course in all_courses:
                if course in courses[course_type]:
                    course_info = courses[course_type][course]
                    row.extend([course_info['学期'], course_info['成绩'], course_info['学分']])
                else:
                    row.extend(['', '', ''])
            
            first_six_credits = student_credits_first_six[name][course_type]
            total_credits = student_credits[name][course_type]
            
            is_qualified_first_six = 1 if first_six_credits >= CREDIT_REQUIREMENTS[course_type] else 0
            is_qualified_total = 1 if total_credits >= CREDIT_REQUIREMENTS[course_type] else 0
            
            row.extend([first_six_credits, is_qualified_first_six, total_credits, is_qualified_total])
            data.append(row)

        result_df = pd.DataFrame(data, index=student_courses.keys(), columns=columns)
        
        # 在创建DataFrame之前添加这些打印语句
        print(f"Number of columns: {len(columns)}")
        print(f"Number of data columns: {len(data[0]) if data else 0}")
        print("Columns:")
        print(columns)
        print("\nFirst row of data:")
        print(data[0] if data else "No data")
        
        result_dfs[course_type] = result_df

    # 保存结果
    with pd.ExcelWriter('student_credits_analysis.xlsx') as writer:
        for course_type, df in result_dfs.items():
            df.to_excel(writer, sheet_name=course_type)
    print(f"已将学生各类课程学分信息保存到 student_credits_analysis.xlsx")

if __name__ == "__main__":
    directory_path = '/Users/crj/learn1/data/学院学生数据'
    process_student_data(directory_path)
