import json
with open(r'c:\Users\Kat\Documents\AICB_lab\Day22-Track3-DPO-Alignment-Lab\colab\Lab22_DPO_T4.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

import sys
with open(r'c:\Users\Kat\Documents\AICB_lab\Day22-Track3-DPO-Alignment-Lab\scratch_output.txt', 'w', encoding='utf-8') as f:
    for cell in nb.get('cells', []):
        outputs = cell.get('outputs', [])
        source = "".join(cell.get('source', []))
        
        stdout_texts = []
        for o in outputs:
            if o.get('name') == 'stdout':
                text = "".join(o.get('text', []))
                stdout_texts.append(text)
            elif o.get('data') and o['data'].get('text/plain'):
                text = "".join(o['data']['text/plain'])
                stdout_texts.append(text)
        
        if stdout_texts:
            f.write("=== SOURCE ===\n")
            f.write(source[:500] + "...\n")
            f.write("=== STDOUT ===\n")
            f.write("\n".join(stdout_texts) + "\n")
            f.write("="*40 + "\n")

