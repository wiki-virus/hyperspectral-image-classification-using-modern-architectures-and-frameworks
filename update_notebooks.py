import json
import glob

replacement_source = [
    "# --- 6. Plotting Cell ---\n",
    "import pandas as pd\n",
    "# Plotting training curves and evaluation results\n",
    "histories = {\n",
    "    'CNN': (loss_cnn, acc_cnn) if 'loss_cnn' in locals() else None,\n",
    "    'CNN+Attention': (loss_attention, acc_attention) if 'loss_attention' in locals() else None,\n",
    "    'CNN+Mamba': (loss_mamba, acc_mamba) if 'loss_mamba' in locals() else None,\n",
    "    'CNN+Attention+Mamba': (loss_attn_mamba, acc_attn_mamba) if 'loss_attn_mamba' in locals() else None\n",
    "}\n",
    "\n",
    "def smooth(y, window=50):\n",
    "    return pd.Series(y).rolling(window, min_periods=1).mean()\n",
    "\n",
    "# 1. Loss vs Episode\n",
    "plt.figure(figsize=(12, 5))\n",
    "plt.subplot(1, 2, 1)\n",
    "for name, hist in histories.items():\n",
    "    if hist:\n",
    "        plt.plot(hist[0], alpha=0.3)\n",
    "        plt.plot(smooth(hist[0]), label=f'{name}')\n",
    "plt.title('Loss vs Episodes')\n",
    "plt.xlabel('Episode')\n",
    "plt.ylabel('Loss')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "\n",
    "# 2. Accuracy vs Episode\n",
    "plt.subplot(1, 2, 2)\n",
    "for name, hist in histories.items():\n",
    "    if hist:\n",
    "        plt.plot(hist[1], alpha=0.3)\n",
    "        plt.plot(smooth(hist[1]), label=f'{name}')\n",
    "plt.title('Accuracy vs Episodes')\n",
    "plt.xlabel('Episode')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# 3. Holdout Evaluation Distributions\n",
    "if eval_results:\n",
    "    plt.figure(figsize=(10, 6))\n",
    "    data_to_plot = [eval_results[name] for name in eval_results]\n",
    "    plt.boxplot(data_to_plot, tick_labels=list(eval_results.keys()))\n",
    "    plt.title('5-Shot Holdout Test Accuracy Distribution (100 Trials)')\n",
    "    plt.ylabel('Accuracy (%)')\n",
    "    plt.grid(True)\n",
    "    plt.show()\n"
]

notebooks = ['fsl_maml.ipynb', 'fsl_protonet.ipynb', 'fsl_relationnet.ipynb', 'fsl_siemese.ipynb']

for nb in notebooks:
    print(f"Processing {nb}...")
    with open(nb, 'r', encoding='utf-8') as f:
        d = json.load(f)
    
    # Find the plotting cell
    found = False
    for cell in reversed(d['cells']):
        if cell['cell_type'] == 'code':
            source_text = ''.join(cell['source'])
            if 'Plotting training curves' in source_text or 'Histories' in source_text or 'histories = {' in source_text or 'Loss vs Episodes' in source_text:
                cell['source'] = replacement_source
                found = True
                break
    
    if found:
        with open(nb, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=1)
        print(f"Updated {nb}")
    else:
        print(f"Could not find plotting cell in {nb}")

