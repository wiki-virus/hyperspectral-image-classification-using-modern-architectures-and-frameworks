import json

def extract_code(notebook_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = []
    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            code_cells.append(f"# === CELL {i} ===\n{source}\n")
            
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(code_cells))
    print(f"Extracted {len(code_cells)} code cells from {notebook_path} to {output_path}")

extract_code('few shot_backup.ipynb', 'few_shot_backup_code.py')
