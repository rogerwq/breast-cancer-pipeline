"""
Job 00 — Verify CNN model files are present and loadable.
Writes outputs/model_status.json for downstream jobs.
"""
import argparse, json, sys
from pathlib import Path

REQUIRED = [
    'binary_model.h5',
    'benign_subtype_model.h5',
    'malignant_subtype_model.h5',
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', required=True)
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    status = {'models_dir': str(models_dir), 'files': {}, 'ok': True}

    for fname in REQUIRED:
        fpath = models_dir / fname
        exists = fpath.exists()
        size_mb = round(fpath.stat().st_size / 1e6, 1) if exists else 0
        status['files'][fname] = {'exists': exists, 'size_mb': size_mb}
        if not exists:
            status['ok'] = False
            print(f'  ✗ MISSING: {fpath}', file=sys.stderr)
        else:
            print(f'  ✓ {fname}  ({size_mb} MB)')

    # Quick load test
    if status['ok']:
        print('Running quick load test...')
        import tensorflow as tf
        for fname in REQUIRED:
            tf.keras.models.load_model(str(models_dir / fname))
            print(f'  ✓ Loaded {fname}')
        status['load_test'] = 'passed'
    else:
        print('ERROR: Missing model files. Stopping.', file=sys.stderr)
        sys.exit(1)

    out = Path('outputs/model_status.json')
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(status, indent=2))
    print(f'\nStatus written to {out}')
    print('Job 00 complete ✓')

if __name__ == '__main__':
    main()
