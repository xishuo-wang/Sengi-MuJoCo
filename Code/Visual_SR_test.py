import pandas as pd
import sqlite3
import os

db_path = r"D:\Software\测力台软件\Data\pi_10_260522_14.db"
conn = sqlite3.connect(db_path)

# 获取数据库文件名（不含扩展名）
db_name = os.path.splitext(os.path.basename(db_path))[0]

# 获取所有表
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)

# 导出每个表
for i, table_name in enumerate(tables['name']):
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    
    # 生成CSV文件名
    if len(tables) > 1:
        csv_name = f"{db_name}_{table_name}.csv"
    else:
        csv_name = f"{db_name}.csv"
    
    csv_path = os.path.join(os.path.dirname(db_path), csv_name)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"已导出：{csv_path}")

conn.close()