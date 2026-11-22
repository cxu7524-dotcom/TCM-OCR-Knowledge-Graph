"""Inspect each missed row alongside the corresponding transcription output."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
score=json.loads((ROOT/'output/formula_stage6/run-v6az4xah/正文方剂表_步骤6逐行评测.json').read_text(encoding='utf8'))
for page, data in score['pages'].items():
    for row in data['manual']['ledger']:
        if row['matched']:
            continue
        print('\nPAGE',page,'GT',row['id'],row['name'],row['herb'],row['dose'])
        for pred in data['manual']['predictions']:
            if pred['name']['value'] == row['name']:
                print(' PRED',pred['herb']['value'],pred['dose']['value'],
                      'HERB_SRC', (pred['herb'].get('source') or {}).get('text'),
                      'DOSE_SRC', (pred['dose'].get('source') or {}).get('text'))
