import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('few shot_backup.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in [11, 12]:
    if i < len(nb['cells']):
        cell = nb['cells'][i]
        print(f"=== CELL {i} ===")
        for out in cell.get('outputs', []):
            if 'text' in out:
                print("".join(out['text']))
