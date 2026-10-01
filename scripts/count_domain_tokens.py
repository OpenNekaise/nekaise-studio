"""CPU-only native-tokenizer accounting for a prepared core; never loads weights."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from collections import Counter


def count(out, tokenizer_path):
    # Bound this offline job; do not import torch/transformers or download anything.
    os.environ.setdefault('RAYON_NUM_THREADS', '4')
    from tokenizers import Tokenizer

    data = tokenizer_path.read_bytes()
    pinned = out / 'accounting' / 'tokenizer.json'
    pinned.parent.mkdir(exist_ok=True)
    pinned.write_bytes(data)
    tokenizer = Tokenizer.from_file(str(pinned))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    manifest = out / 'dataset/core.manifest.jsonl'
    manifest_bytes = manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    summary_path = out / 'accounting/summary.json'
    summary_path.write_text(json.dumps({'status': 'counting', 'manifest_sha256': manifest_hash}) + '\n')
    by_source, by_category = Counter(), Counter()
    total, documents, batch, chars = 0, 0, [], 0
    with io.StringIO(manifest_bytes.decode()) as rows, (out / 'accounting/core.tokens.jsonl').open('w') as results:
        def flush():
            nonlocal total, documents, batch, chars
            if not batch:
                return
            texts = []
            for row in batch:
                path = (out / row['object_path']).resolve()
                if not path.is_relative_to(out.resolve()):
                    raise ValueError('Object escapes bundle')
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != row['text_sha256']:
                    raise ValueError('Object changed during token accounting')
                texts.append(raw.decode('utf-8-sig'))
            encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
            for row, encoding in zip(batch, encodings, strict=True):
                n = len(encoding.ids)
                total += n
                documents += 1
                by_source[row['source']] += n
                for category in row.get('categories', []):
                    by_category[category] += n
                results.write(json.dumps({'id': row['id'], 'text_sha256': row['text_sha256'], 'tokens': n}) + '\n')
            batch, chars = [], 0
            if documents % 1000 < 100:
                print(json.dumps({'documents': documents, 'native_text_tokens': total}), flush=True)
        for line in rows:
            row = json.loads(line)
            batch.append(row)
            chars += row['chars']
            if chars >= 250000 or len(batch) >= 64:
                flush()
        flush()
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != manifest_hash:
        raise RuntimeError('Core export changed during accounting; rerun after finalization finishes')
    report = {'status': 'complete', 'documents': documents, 'native_text_tokens': total,
        'tokens_by_source': by_source, 'tokens_by_category_overlapping': by_category,
        'tokenizer_sha256': hashlib.sha256(data).hexdigest(), 'tokenizer_source': str(tokenizer_path.resolve()),
        'manifest_sha256': manifest_hash,
        'special_tokens': False, 'truncation': False, 'model_weights_loaded': False,
        'meaning': 'Exact native text token count, one copy of each core document. Not trained exposure, chat targets, packing, BOS/EOS or loss-mask accounting.'}
    summary_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    args = parser.parse_args()
    count(args.out.resolve(), args.tokenizer)
