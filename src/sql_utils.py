import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))])

import sqlite3
import sqlparse

from func_timeout import func_timeout, FunctionTimedOut

# 美化表格输出，将列名和数值对齐输出
def normalize_perform_table(column_names: list, values: list):
    rows = []
    # 计算每一列的最大宽度，以确保表格对齐
    widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]
    # 生成表头，列名左对齐并根据最大宽度调整列宽
    header = '  '.join(f'{column.ljust(width)}' for column, width in zip(column_names, widths))
    # 生成每一行的数据，值左对齐并根据最大宽度调整列宽
    for value in values:
        row = '  '.join(f'{str(v).ljust(width)}' for v, width in zip(value, widths))
        rows.append(row)
    # 返回表头和所有行的数据
    return header + '\n' + '\n'.join(rows)

# 规范化 CREATE TABLE 语句，使用 sqlparse 进行格式化
def normalize_create_table(create_table_statement):
    # 使用 sqlparse 对 SQL 语句进行格式化，并将其转换为单行
    formatted_statement = sqlparse.format(create_table_statement,
                                          reindent=False,
                                          keyword_case='upper',
                                          identifier_case=None)
    # 返回格式化后的 CREATE TABLE 语句
    return formatted_statement

# 生成数据库的表结构提示信息，包含表的创建语句和示例数据
def generate_schema_prompt(db_path, num_rows=3, normalization=True):
    # 连接到 SQLite 数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # 查询数据库中的所有表及其创建语句
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    full_schema_prompt_list = []

    # 遍历所有表
    for table_name, create_statement in tables:
        # 跳过 SQLite 的内部表
        if table_name == 'sqlite_sequence':
            continue
        # 如果需要，规范化 CREATE TABLE 语句
        if normalization:
            create_statement = normalize_create_table(create_statement)
            
        schema_prompt = create_statement
        # 如果需要示例数据，查询表中的前 num_rows 行
        if num_rows > 0:
            table_name = '`{}`'.format(table_name) if table_name in['order', 'group'] or table_name.upper() in sqlparse.keywords.KEYWORDS else table_name
            cursor.execute("SELECT * FROM {} LIMIT {}".format(table_name, num_rows))
            column_names = [description[0] for description in cursor.description]
            values = cursor.fetchall()
            # 使用美化函数生成示例数据表格
            rows_prompt = normalize_perform_table(column_names=column_names, values=values)
            # 将创建语句和示例数据组合在一起
            schema_prompt += f"\n/* \n{num_rows} example rows: \nSELECT * FROM {table_name} LIMIT {num_rows}; \n{rows_prompt}\n */"
        # 将表的提示信息添加到列表中
        full_schema_prompt_list.append(schema_prompt)

    # 返回所有表的提示信息，以换行分隔
    return "\n\n".join(full_schema_prompt_list)

def execute_sql(predicted_sql,ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    result = "failed"
    passed = False
    if set(predicted_res) == set(ground_truth_res):
        result = "passed"
        passed = True
    return result, passed


def execute_model(predicted_sql,ground_truth, db_place, meta_time_out):
    try:
        result, passed = func_timeout(meta_time_out, execute_sql,
                                  args=(predicted_sql, ground_truth, db_place))
    except FunctionTimedOut:
        result = "timeout"
        passed = False
    except Exception as e:
        result = f"error:{e}"
        passed = False

    return result, passed