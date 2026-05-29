import json

def check_outputs(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"=== Outputs for {path} ===")
    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', [])).strip()
            # print first line of code
            first_line = source.split('\n')[0] if source else ''
            outputs = cell.get('outputs', [])
            has_output = len(outputs) > 0
            text_outputs = []
            for out in outputs:
                if 'text' in out:
                    text_outputs.append("".join(out['text']))
                elif 'data' in out and 'text/plain' in out['data']:
                    text_outputs.append("".join(out['data']['text/plain']))
            
            output_summary = "\n".join(text_outputs)
            # Find lines containing "accuracy" or "HOLDOUT"
            lines = [line for line in output_summary.split('\n') if 'accuracy' in line.lower() or 'holdout' in line.lower() or 'epoch' in line.lower()]
            if lines or has_output:
                print(f"Cell {i}: {first_line}")
                for line in lines[:5]:
                    print(f"  [OUT] {line}")

check_outputs('few shot.ipynb')
check_outputs('few shot_backup.ipynb')
