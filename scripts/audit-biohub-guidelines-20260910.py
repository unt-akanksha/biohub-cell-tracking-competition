"""Read-only live CLI audit; cache official page text and newest topic inventory."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / '.biohub/cache/guidelines-audit-20260910-1925'
COMP = 'biohub-cell-tracking-during-development'


def query(arguments):
    result = subprocess.run(['kaggle', 'competitions', *arguments, '--format', 'json'],
                            capture_output=True, text=True, encoding='utf-8', check=True,
                            timeout=90, env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8'))
    return json.loads(result.stdout)


def main():
    TARGET.mkdir(exist_ok=False)
    pages = query(['pages', COMP, '--content'])
    topics = query(['topics', 'list', COMP, '--sort-by', 'recent'])
    (TARGET / 'pages.json').write_text(json.dumps(pages, indent=2, ensure_ascii=False), encoding='utf-8')
    (TARGET / 'topics.json').write_text(json.dumps(topics, indent=2, ensure_ascii=False), encoding='utf-8')
    rows = []
    for page in pages:
        text = page.get('content', '')
        name = page['name'].lower().replace(' ', '-')
        if not name.replace('-', '').isalnum():
            raise ValueError('Unexpected official page name')
        (TARGET / (name + '.md')).write_text(text, encoding='utf-8')
        rows.append(dict(name=page['name'], characters=len(text),
                         content_sha256=hashlib.sha256(text.encode()).hexdigest()))
    receipt = dict(utc=datetime.now(timezone.utc).isoformat(), pages=rows,
                   topic_inventory_count=len(topics), external_mutations=False,
                   discussions_fully_read=False, submission_performed=False)
    (TARGET / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
