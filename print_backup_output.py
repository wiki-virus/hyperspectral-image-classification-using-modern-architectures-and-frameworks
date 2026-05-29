import json
import sys

# Set standard output encoding to utf-8 for Windows console
sys.stdout.reconfigure(encoding='utf-8')

with open('few shot_backup.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][6]
for out in cell.get('outputs', []):
    if 'text' in out:
        print("".join(out['text']))
