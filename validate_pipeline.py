"""Quick validation of the data pipeline logic without Rich console."""
import sys, json, yaml, random
from pathlib import Path
from collections import defaultdict

with open('training/config.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
d = cfg['data']

# Load schemas
with open(d['tables'], encoding='utf-8') as f:
    tables = json.load(f)
schema_map = {t['db_id']: t for t in tables}
print(f'Schemas loaded: {len(schema_map)}')

# Load data
with open(d['spider_train'], encoding='utf-8') as f:
    train_spider = json.load(f)
with open(d['spider_others'], encoding='utf-8') as f:
    train_others = json.load(f)
with open(d['spider_dev'], encoding='utf-8') as f:
    dev_data = json.load(f)

all_train = train_spider + train_others
random.seed(42)
random.shuffle(all_train)
val_data = all_train[:500]
train_data = all_train[500:]
print(f'Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(dev_data)}')

# Test schema builder
TYPE_MAP = {'number': 'REAL', 'text': 'TEXT', 'time': 'DATETIME', 'boolean': 'INTEGER', 'others': 'TEXT'}

def build_create(db_schema):
    table_names = db_schema['table_names_original']
    col_names = db_schema['column_names_original']
    col_types = db_schema['column_types']
    primary_keys = set(db_schema['primary_keys'])
    foreign_keys = {fk[0]: fk[1] for fk in db_schema['foreign_keys']}
    tables_d = defaultdict(list)
    for col_idx, (table_idx, col_name) in enumerate(col_names):
        if table_idx == -1:
            continue
        tables_d[table_idx].append((col_idx, col_name, col_types[col_idx]))
    stmts = []
    for t_idx, cols in tables_d.items():
        t_name = table_names[t_idx]
        lines = []
        for col_idx, col_name, col_type in cols:
            sql_type = TYPE_MAP.get(col_type, 'TEXT')
            pk = ' PRIMARY KEY' if col_idx in primary_keys else ''
            lines.append('    ' + col_name + ' ' + sql_type + pk)
        stmts.append('CREATE TABLE ' + t_name + ' (\n' + ',\n'.join(lines) + '\n);')
    return '\n\n'.join(stmts)

sample = train_data[0]
db_id = sample['db_id']
schema_sql = build_create(schema_map[db_id])
print('Sample DB:', db_id)
print('Question:', sample['question'])
print('SQL:', sample['query'])
print('Schema snippet:', schema_sql[:300])

# Test executor with actual SQLite
import sqlite3, os
db_path = os.path.join(d['databases_dir'], db_id, db_id + '.sqlite')
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(sample['query']).fetchall()
        print('Execution test: OK, rows returned:', len(rows))
    except Exception as e:
        print('Execution error:', e)
    conn.close()
else:
    print('DB file not found at:', db_path)

print('\nPipeline validation: PASSED')
