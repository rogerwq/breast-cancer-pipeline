"""
Job 01 — DenseNet121 two-stage inference pipeline.
Runs on CPU (no NVIDIA required).
Outputs: outputs/cnn_prediction.json, outputs/confidence_chart.png
"""
import argparse, json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import tensorflow as tf

IMG_SIZE = 224

BENIGN_SUBTYPES = {
    'adenosis': 0, 'fibroadenoma': 1,
    'phyllodes_tumor': 2, 'tubular_adenoma': 3,
}
MALIGNANT_SUBTYPES = {
    'ductal_carcinoma': 0, 'lobular_carcinoma': 1,
    'mucinous_carcinoma': 2, 'papillary_carcinoma': 3,
}
BENIGN_IDX    = {v: k for k, v in BENIGN_SUBTYPES.items()}
MALIGNANT_IDX = {v: k for k, v in MALIGNANT_SUBTYPES.items()}

# ── Preprocessing ──────────────────────────────────────────────────────────
def apply_clahe(img_bgr):
    lab   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def preprocess(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {img_path}')
    img = apply_clahe(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img / 255.0

# ── Uncertainty ────────────────────────────────────────────────────────────
def uncertainty_level(probs):
    margin = float(np.max(probs)) - float(np.sort(probs)[-2])
    if margin >= 0.40: return 'LOW'
    if margin >= 0.20: return 'MODERATE'
    return 'HIGH'

# ── Confidence chart ───────────────────────────────────────────────────────
def save_confidence_chart(result, out_path):
    labels = [k.replace('_', ' ').title() for k in result['differential']]
    values = [v * 100 for v in result['differential'].values()]
    is_mal = result['binary'] == 'malignant'
    colors = ['#d32f2f' if is_mal else '#388e3c'] + ['#90a4ae'] * (len(labels) - 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4),
                             gridspec_kw={'width_ratios': [1, 2]})
    fig.patch.set_facecolor('#1a1a2e')

    # Donut — binary
    ax0 = axes[0]
    ax0.set_facecolor('#1a1a2e')
    mal_p = result['mal_prob'] * 100
    ben_p = result['ben_prob'] * 100
    ax0.pie([mal_p, ben_p], colors=['#d32f2f', '#388e3c'],
            startangle=90, counterclock=False,
            wedgeprops=dict(width=0.45, edgecolor='#1a1a2e', linewidth=2))
    ax0.text(0, 0.08, f"{result['binary_conf']*100:.1f}%",
             ha='center', va='center', fontsize=22,
             fontweight='bold', color='white')
    verdict = '🔴 MALIGNANT' if is_mal else '🟢 BENIGN'
    ax0.text(0, -0.22, verdict, ha='center', va='center',
             fontsize=12, fontweight='bold', color='white')
    ax0.text(0, -0.42, result['display_label'].split(': ')[1],
             ha='center', fontsize=9, color='#b0bec5')
    ax0.legend(handles=[
        mpatches.Patch(color='#d32f2f', label=f'Malignant {mal_p:.1f}%'),
        mpatches.Patch(color='#388e3c', label=f'Benign {ben_p:.1f}%'),
    ], loc='lower center', fontsize=8, facecolor='#1a1a2e',
       labelcolor='white', framealpha=0.4, bbox_to_anchor=(0.5, -0.15))
    ax0.set_title('Binary Classification', color='white',
                  fontsize=11, fontweight='bold', pad=8)

    # Horizontal bars — subtypes
    ax1 = axes[1]
    ax1.set_facecolor('#16213e')
    bars = ax1.barh(labels[::-1], values[::-1], color=colors[::-1],
                    edgecolor='#1a1a2e', height=0.5)
    for bar, val in zip(bars, values[::-1]):
        ax1.text(min(val + 1, 98), bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}%', va='center', ha='left',
                 color='white', fontsize=10, fontweight='bold')
    ax1.set_xlim(0, 110)
    ax1.set_xlabel('Probability (%)', color='#b0bec5', fontsize=9)
    ax1.set_title('Subtype Differential', color='white',
                  fontsize=11, fontweight='bold')
    ax1.tick_params(colors='white', labelsize=9)
    ax1.spines[:].set_color('#37474f')
    unc_color = {'LOW': '#4caf50', 'MODERATE': '#ff9800', 'HIGH': '#f44336'}
    ax1.text(0.98, -0.12,
             f"AI Uncertainty: {result['uncertainty']}",
             transform=ax1.transAxes, ha='right', fontsize=9,
             color=unc_color[result['uncertainty']], fontweight='bold')

    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print(f'  Confidence chart saved → {out_path}')

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',  required=True)
    parser.add_argument('--models_dir',  required=True)
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    out_dir    = Path('outputs')
    out_dir.mkdir(exist_ok=True)

    # Load models
    print('Loading CNN models on CPU...')
    binary_model    = tf.keras.models.load_model(str(models_dir / 'binary_model.h5'))
    benign_model    = tf.keras.models.load_model(str(models_dir / 'benign_subtype_model.h5'))
    malignant_model = tf.keras.models.load_model(str(models_dir / 'malignant_subtype_model.h5'))
    print('  ✓ All models loaded')

    # Preprocess
    print(f'Processing image: {args.image_path}')
    batch = np.expand_dims(preprocess(args.image_path), 0)

    # Stage 1 — binary
    mal   = float(binary_model.predict(batch, verbose=0)[0][0])
    ben   = round(1 - mal, 4); mal = round(mal, 4)
    pred  = 'malignant' if mal > 0.5 else 'benign'
    conf  = mal if pred == 'malignant' else ben

    # Stage 2 — subtype
    if pred == 'benign':
        raw     = benign_model.predict(batch, verbose=0)[0]
        idx_map = BENIGN_IDX
    else:
        raw     = malignant_model.predict(batch, verbose=0)[0]
        idx_map = MALIGNANT_IDX

    top  = int(np.argmax(raw))
    diff = {idx_map[i]: round(float(p), 4)
            for i, p in sorted(enumerate(raw), key=lambda x: -x[1])}

    result = {
        'binary':        pred,
        'binary_conf':   round(conf, 4),
        'ben_prob':      ben,
        'mal_prob':      mal,
        'subtype':       idx_map[top],
        'subtype_conf':  round(float(raw[top]), 4),
        'uncertainty':   uncertainty_level(raw),
        'differential':  diff,
        'display_label': f"{pred.capitalize()}: {idx_map[top].replace('_',' ').title()}",
        'image_path':    str(args.image_path),
    }

    # Save outputs
    pred_path = out_dir / 'cnn_prediction.json'
    pred_path.write_text(json.dumps(result, indent=2))
    save_confidence_chart(result, str(out_dir / 'confidence_chart.png'))

    # Print summary
    print(f'\n  Result       : {result["display_label"]}')
    print(f'  Binary conf  : {result["binary_conf"]*100:.1f}%')
    print(f'  Subtype conf : {result["subtype_conf"]*100:.1f}%')
    print(f'  Uncertainty  : {result["uncertainty"]}')
    print('\n  Subtype Differential:')
    for name, prob in diff.items():
        bar = '█' * int(prob * 30) + '░' * (30 - int(prob * 30))
        print(f'    {name.replace("_"," ").title():<28}: [{bar}] {prob*100:.1f}%')
    print('\nJob 01 complete ✓')

if __name__ == '__main__':
    main()
