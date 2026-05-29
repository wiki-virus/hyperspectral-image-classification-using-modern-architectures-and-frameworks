import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('few shot_backup.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        outputs = cell.get('outputs', [])
        for out in outputs:
            if 'text' in out:
                text = "".join(out['text'])
                for line in text.split('\n'):
                    if 'holdout' in line.lower() or 'accuracy' in line.lower():
                        print(f"Cell {i} | {line}")
