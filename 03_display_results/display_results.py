"""Job 03 — Print the final report and evaluation scores to console."""
import argparse, json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', default='patient_report.txt')
    parser.add_argument('--scores', default='evaluation_scores.json')
    args = parser.parse_args()

    report = Path(args.report).read_text(encoding='utf-8')
    scores = json.loads(Path(args.scores).read_text())

    print('\n' + '━'*65)
    print(report)
    print('━'*65)

    if scores.get('scores'):
        s = scores['scores']
        r = s['rouge']
        b = s['bertscore']
        print('\n📐 REPORT QUALITY vs. MEDICAL GROUND TRUTH')
        print('─'*45)
        print(f'  ROUGE-1  : {r["rouge1"]:.4f}')
        print(f'  ROUGE-2  : {r["rouge2"]:.4f}')
        print(f'  ROUGE-L  : {r["rougeL"]:.4f}')
        if b.get('skipped') or b['f1'] is None:
            print(f'  BERT-F1  : Skipped (requires torch>=2.4)')
        else:
            print(f'  BERT-F1  : {b["f1"]:.4f}', end='  ')
            if b['f1'] > 0.85:   print('✅ Strong — report aligns with medical literature')
            elif b['f1'] > 0.80: print('⚠️  Moderate — review carefully')
            else:                 print('❌ Weak — may contain hallucinations')
        print(f'\n  Model: {scores["model"]}')

    # Write summary file
    out = Path('outputs/final_summary.txt')
    out.parent.mkdir(exist_ok=True)
    out.write_text(report, encoding='utf-8')
    print(f'\n  Summary saved → {out}')
    print('\n✅ Breast Cancer AI Pipeline complete.')

if __name__ == '__main__':
    main()
