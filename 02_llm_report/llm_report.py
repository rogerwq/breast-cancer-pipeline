"""
Job 02 — Patient report generation using Mistral-7B via Ollama.

Ollama runs Mistral-7B-Instruct (Q4_K_M quantized) locally on CPU.
No NVIDIA GPU required — works on Intel Iris Xe.

Pre-requisite (run once on your PC):
    ollama pull mistral
"""

import argparse, json, sys, time
import requests
from pathlib import Path

# ── Ground truth references for ROUGE/BERTScore evaluation ────────────────
REFERENCE_REPORTS = {
    'Benign: Fibroadenoma': (
        "Fibroadenoma is a benign breast tumour. Danger level LOW. "
        "No chemotherapy needed. Urgency ROUTINE. Monitor with ultrasound annually. "
        "No malignant potential. Surgery only if growing rapidly or >3 cm."
    ),
    'Malignant: Ductal Carcinoma': (
        "Invasive ductal carcinoma is the most common breast cancer. Danger HIGH but treatable. "
        "Urgency URGENT — see oncologist within days. Treatment includes surgery, radiation, "
        "hormone therapy. Chemotherapy depends on receptor status and tumour grade. "
        "5-year survival >99% if localised."
    ),
    'Benign: Adenosis': (
        "Adenosis is a benign proliferative breast condition. Danger LOW. "
        "No chemotherapy. Urgency ROUTINE. Annual imaging surveillance. "
        "No significant increase in cancer risk."
    ),
    'Malignant: Lobular Carcinoma': (
        "Invasive lobular carcinoma is the second most common breast cancer. Danger HIGH. "
        "Urgency URGENT. Hormone therapy almost always needed. "
        "MRI essential. Chemotherapy sometimes needed for high-risk cases."
    ),
    'Benign: Phyllodes Tumor': (
        "Benign phyllodes tumour is a rare fibroepithelial breast lesion. Danger LOW to MODERATE. "
        "Urgency SOON. Wide local excision with clear margins required. "
        "No chemotherapy. 10-17% local recurrence risk if incompletely excised."
    ),
    'Malignant: Mucinous Carcinoma': (
        "Mucinous carcinoma is a special type of breast cancer with favourable prognosis. "
        "Danger MODERATE. Urgency SOON. Hormone therapy usually sufficient. "
        "Chemotherapy often not needed. 10-year survival >80-90%."
    ),
    'Benign: Tubular Adenoma': (
        "Tubular adenoma is a rare benign breast tumour. Danger LOW. "
        "No chemotherapy. Urgency ROUTINE. No treatment needed if concordant with imaging. "
        "Annual ultrasound surveillance."
    ),
    'Malignant: Papillary Carcinoma': (
        "Papillary carcinoma is a rare breast cancer with favourable prognosis. "
        "Danger MODERATE. Urgency SOON. Hormone therapy main treatment. "
        "Chemotherapy usually not required. Lumpectomy with clear margins standard."
    ),
}

# ── System prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a compassionate expert AI cancer advisor. A patient has received \
an AI-assisted breast cancer classification from a histopathology image analysis. \
Your job is to explain the result clearly and caringly — like a knowledgeable friend \
who is also a medical expert.

Generate a detailed patient report with EXACTLY these 7 sections and headings:

1. WHAT WE FOUND
   Explain the diagnosis (benign/malignant and specific subtype) in simple plain language.

2. IS THIS DANGEROUS?
   Rate danger: LOW / MODERATE / HIGH with explanation and survival rates where relevant.

3. DO YOU NEED TO SEE A DOCTOR?
   Give urgency: ROUTINE / SOON (within weeks) / URGENT (within days). Name the specialist.

4. WHAT TREATMENT MIGHT BE NEEDED?
   Cover surgery, chemotherapy, radiation, hormone therapy, targeted therapy for this subtype.
   State clearly which are typically needed, sometimes needed, and usually NOT needed.

5. IS CHEMOTHERAPY NECESSARY?
   Answer directly for this specific subtype. Explain what factors determine whether chemo is needed.

6. WARNING SIGNS — WHEN TO SEEK IMMEDIATE HELP
   List specific red-flag symptoms for this cancer type that require immediate medical attention.

7. IMPORTANT DISCLAIMER
   Remind the patient this is an AI tool, not a replacement for a qualified pathologist or oncologist.

Rules:
- Use plain language. Avoid unexplained medical jargon.
- Be honest but compassionate — not falsely reassuring or unnecessarily alarming.
- Be specific to this exact subtype, not generic breast cancer advice."""


def build_prompt(pred: dict) -> str:
    """Build the Mistral [INST] prompt from CNN prediction dict."""
    diff_lines = '\n'.join(
        f"  {i+1}. {k.replace('_', ' ').title():<28} {v*100:.1f}%"
        for i, (k, v) in enumerate(pred['differential'].items())
    )
    unc_note = {
        'LOW'     : 'The AI model is highly confident in this result.',
        'MODERATE': 'The AI has moderate confidence — pathologist review is especially important.',
        'HIGH'    : 'WARNING: AI confidence is LOW. Professional pathologist review is essential.',
    }[pred['uncertainty']]

    user_msg = (
        f"PATIENT AI CANCER SCREENING RESULT\n"
        f"{'─'*50}\n"
        f"  Classification : {pred['binary'].upper()}\n"
        f"  Subtype        : {pred['subtype'].replace('_', ' ').title()}\n"
        f"  Binary confidence  : {pred['binary_conf']*100:.1f}%  "
        f"(Malignant: {pred['mal_prob']*100:.1f}%  |  Benign: {pred['ben_prob']*100:.1f}%)\n"
        f"  Subtype confidence : {pred['subtype_conf']*100:.1f}%\n"
        f"  AI uncertainty     : {pred['uncertainty']}\n"
        f"  Confidence note    : {unc_note}\n\n"
        f"Subtype probability breakdown (all considered subtypes):\n"
        f"{diff_lines}\n\n"
        f"Please generate the full 7-section patient advisory report for this finding."
    )
    # Mistral instruction format
    return f"[INST] {SYSTEM_PROMPT}\n\n{user_msg} [/INST]"


def call_mistral_ollama(prompt: str, host: str, model: str, max_tokens: int) -> str:
    """Call Mistral-7B via Ollama REST API (CPU-compatible, no GPU needed)."""
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model"  : model,
        "prompt" : prompt,
        "stream" : False,
        "options": {
            "num_predict"   : max_tokens,
            "temperature"   : 0.3,
            "top_p"         : 0.85,
            "repeat_penalty": 1.15,
        },
    }
    print(f'  Calling Ollama at {url} with model={model}...')
    print(f'  (Mistral-7B on CPU may take 3-8 minutes — please wait)')
    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=900)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print('\n  ERROR: Cannot connect to Ollama.', file=sys.stderr)
        print('  Make sure Ollama is running: open a terminal and run: ollama serve', file=sys.stderr)
        print('  Then pull Mistral if not done: ollama pull mistral', file=sys.stderr)
        sys.exit(1)
    elapsed = round(time.time() - t0, 1)
    print(f'  Done in {elapsed}s')
    return resp.json()['response'].strip()


def compute_rouge(generated: str, reference: str) -> dict:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    s = scorer.score(reference, generated)
    return {
        'rouge1': round(s['rouge1'].fmeasure, 4),
        'rouge2': round(s['rouge2'].fmeasure, 4),
        'rougeL': round(s['rougeL'].fmeasure, 4),
    }


def compute_bertscore(generated: str, reference: str) -> dict:
    try:
        from bert_score import score as bscore
        P, R, F1 = bscore([generated], [reference], lang='en', verbose=False)
        return {
            'precision': round(float(P[0]), 4),
            'recall'   : round(float(R[0]), 4),
            'f1'       : round(float(F1[0]), 4),
        }
    except (ImportError, Exception) as e:
        print(f'  ⚠️  BERTScore skipped: {e}')
        print('  (BERTScore requires torch>=2.4 — ROUGE scores are still valid)')
        return {'precision': None, 'recall': None, 'f1': None, 'skipped': True}


def interpret(scores: dict) -> dict:
    bert_f1 = scores['bertscore']['f1']
    bert_rating = (
        'Strong ✅'   if bert_f1 and bert_f1 > 0.85 else
        'Moderate ⚠️' if bert_f1 and bert_f1 > 0.80 else
        'Skipped ⚠️'  if bert_f1 is None else 'Weak ❌'
    )
    return {
        'ROUGE-1': 'Good ✅' if scores['rouge']['rouge1'] > 0.20 else
                   'Fair ⚠️'  if scores['rouge']['rouge1'] > 0.10 else 'Low ❌',
        'ROUGE-L': 'Good ✅' if scores['rouge']['rougeL'] > 0.15 else
                   'Fair ⚠️'  if scores['rouge']['rougeL'] > 0.08 else 'Low ❌',
        'BERT-F1': bert_rating,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction',   default='cnn_prediction.json')
    parser.add_argument('--ollama_model', default='mistral')
    parser.add_argument('--ollama_host',  default='http://host.docker.internal:11434')
    parser.add_argument('--max_tokens',   type=int, default=450)
    args = parser.parse_args()

    # Load CNN prediction from Job 01
    pred_path = Path(args.prediction)
    if not pred_path.exists():
        print(f'ERROR: {pred_path} not found. Run Job 01 first.', file=sys.stderr)
        sys.exit(1)
    pred = json.loads(pred_path.read_text())
    print(f'\nCNN result: {pred["display_label"]}')
    print(f'  Binary  : {pred["binary_conf"]*100:.1f}%  (Malignant: {pred["mal_prob"]*100:.1f}% | Benign: {pred["ben_prob"]*100:.1f}%)')
    print(f'  Subtype : {pred["subtype_conf"]*100:.1f}%')
    print(f'  Uncertainty: {pred["uncertainty"]}')

    # Build prompt and generate report
    prompt      = build_prompt(pred)
    report_text = call_mistral_ollama(
        prompt, args.ollama_host, args.ollama_model, args.max_tokens)

    # Evaluate vs ground truth
    label = pred['display_label']
    ref   = REFERENCE_REPORTS.get(label)
    eval_scores = None
    if ref:
        print('\nComputing evaluation scores vs ground truth...')
        rouge_scores = compute_rouge(report_text, ref)
        bert_scores  = compute_bertscore(report_text, ref)
        eval_scores  = {'rouge': rouge_scores, 'bertscore': bert_scores}
        ratings      = interpret(eval_scores)
        print(f'  ROUGE-1 : {rouge_scores["rouge1"]:.4f}  {ratings["ROUGE-1"]}')
        print(f'  ROUGE-2 : {rouge_scores["rouge2"]:.4f}')
        print(f'  ROUGE-L : {rouge_scores["rougeL"]:.4f}  {ratings["ROUGE-L"]}')
        if bert_scores.get('skipped'):
            print(f'  BERT-F1 : Skipped (torch version incompatible — rebuild image to fix)')
        else:
            print(f'  BERT-F1 : {bert_scores["f1"]:.4f}  {ratings["BERT-F1"]}')
            if bert_scores['f1'] > 0.84:
                print('\n  ✅ Report is semantically aligned with medical ground truth.')
            else:
                print('\n  ⚠️  Low BERT-F1 — review report carefully before clinical use.')
    else:
        print(f'\n  No ground truth reference for: {label}')

    # Save outputs
    out_dir = Path('outputs')
    out_dir.mkdir(exist_ok=True)

    report_path = out_dir / 'patient_report.txt'
    report_path.write_text(
        f"BREAST CANCER AI ADVISORY REPORT\n"
        f"{'='*60}\n"
        f"Diagnosis   : {label}\n"
        f"Binary conf : {pred['binary_conf']*100:.1f}%  "
        f"(Mal: {pred['mal_prob']*100:.1f}% | Ben: {pred['ben_prob']*100:.1f}%)\n"
        f"Subtype conf: {pred['subtype_conf']*100:.1f}%\n"
        f"Uncertainty : {pred['uncertainty']}\n"
        f"\nSubtype Differential:\n" +
        '\n'.join(
            f"  {k.replace('_',' ').title():<28}: {v*100:.1f}%"
            for k, v in pred['differential'].items()
        ) +
        f"\n{'='*60}\n\n{report_text}\n",
        encoding='utf-8'
    )

    scores_path = out_dir / 'evaluation_scores.json'
    scores_path.write_text(json.dumps({
        'diagnosis' : label,
        'model'     : f'Mistral-7B via Ollama ({args.ollama_model})',
        'scores'    : eval_scores,
        'report_length_chars': len(report_text),
    }, indent=2))

    print(f'\nOutputs saved:')
    print(f'  {report_path}')
    print(f'  {scores_path}')
    print('\nJob 02 complete ✓')


if __name__ == '__main__':
    main()
