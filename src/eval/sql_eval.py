import sqlite3

from func_timeout import func_timeout, FunctionTimedOut

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