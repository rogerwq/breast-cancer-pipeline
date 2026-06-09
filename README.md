# Breast Cancer Diagnosis — CNN + LLM Pipeline

**Author:** Maryam Rafaqat  
**GitHub:** [mary270/breast-cancer-pipeline](https://github.com/mary270/breast-cancer-pipeline)  
**Dataset:** [BreakHis on Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis)  
**Models:** [HuggingFace — breast-cancer-densenet121](https://huggingface.co/MaryamRafaqat/breast-cancer-densenet121)

---

## What This Project Does

This pipeline takes a breast histopathology image as input and produces a full patient advisory report in two stages:

1. **CNN Stage (DenseNet121)** — classifies the image as benign or malignant, then identifies the specific subtype (4 benign + 4 malignant subtypes from the BreakHis dataset) with confidence percentages
2. **LLM Stage (Mistral-7B)** — generates a 7-section patient-friendly report covering: what was found, danger level, doctor urgency, treatment options, chemotherapy guidance, warning signs, and disclaimer

---

## Repository Structure

```
breast-cancer-pipeline/
├── .chiral/
│   └── workflow.toml          ← Silva workflow definition
├── 00_setup_models/           ← Job 00: verify model files
├── 01_cnn_inference/          ← Job 01: DenseNet121 inference
├── 02_llm_report/             ← Job 02: Mistral-7B report generation
├── 03_display_results/        ← Job 03: display + save results
├── notebooks/                 ← Kaggle training notebooks
│   ├── 01_breast_cancer_classification.ipynb
│   ├── 02_llm_medical_report_generation.ipynb
│   └── 03_gradio_ui.ipynb
├── input_files/
│   └── sample_image.png       ← sample BreakHis test image
├── Dockerfile
├── requirements_silva.txt
└── global_params.json
```

---

## How to Run (Silva)

### Prerequisites

| Requirement | Install |
|---|---|
| Docker Desktop | https://www.docker.com/products/docker-desktop |
| Ollama | https://ollama.com/download |
| Mistral-7B model | `ollama pull mistral` |
| Silva | https://github.com/chiral-data/silva |

### Step 1 — Download model files

Download the 3 model files from HuggingFace:  
👉 https://huggingface.co/MaryamRafaqat/breast-cancer-densenet121

Place them in a `models/` folder inside the repo:
```
breast-cancer-pipeline/
└── models/
    ├── binary_model.h5
    ├── benign_subtype_model.h5
    └── malignant_subtype_model.h5
```

### Step 2 — Add a test image

Place any BreakHis `.png` histopathology image in `input_files/` and name it `sample_image.png`.  
A sample image is already included in this repo.

### Step 3 — Build the Docker image

```bash
docker build -t breast-cancer-pipeline:latest .
```

### Step 4 — Run with Silva

```bash
export SILVA_WORKFLOW_HOME=/path/to/breast-cancer-pipeline
silva
```

Select the `breast-cancer-diagnosis-cnn-llm` workflow and press Enter.

### Alternative — Run without Silva (Docker directly)

```bash
# Job 00 — verify models
docker run --rm \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/00_setup_models:/workspace/job \
  -w /workspace/job \
  -e PARAM_MODELS_DIR=/workspace/models \
  breast-cancer-pipeline:latest bash run.sh

# Job 01 — CNN inference
docker run --rm \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/input_files:/workspace/input_files \
  -v $(pwd)/01_cnn_inference:/workspace/job \
  -w /workspace/job \
  -e PARAM_MODELS_DIR=/workspace/models \
  -e PARAM_IMAGE_PATH=/workspace/input_files/sample_image.png \
  breast-cancer-pipeline:latest bash run.sh

# Job 02 — Mistral-7B report (requires Ollama running)
docker run --rm \
  -v $(pwd)/01_cnn_inference/outputs:/workspace/inputs \
  -v $(pwd)/02_llm_report:/workspace/job \
  -w /workspace/job \
  --add-host=host.docker.internal:host-gateway \
  -e PARAM_OLLAMA_MODEL=mistral \
  -e PARAM_OLLAMA_HOST=http://host.docker.internal:11434 \
  breast-cancer-pipeline:latest \
  bash -c "cp /workspace/inputs/cnn_prediction.json . && bash run.sh"

# Job 03 — display results
docker run --rm \
  -v $(pwd)/02_llm_report/outputs:/workspace/inputs \
  -v $(pwd)/03_display_results:/workspace/job \
  -w /workspace/job \
  breast-cancer-pipeline:latest \
  bash -c "cp /workspace/inputs/patient_report.txt . && cp /workspace/inputs/evaluation_scores.json . && bash run.sh"
```

---

## Sample Output

```
🔴 MALIGNANT — Ductal Carcinoma
Binary confidence : 100.0%  (Malignant: 100.0% | Benign: 0.0%)
Subtype confidence: 80.7%
Uncertainty       : LOW

Subtype Differential:
  Ductal Carcinoma      : ████████████████████ 80.7%
  Lobular Carcinoma     : ████░░░░░░░░░░░░░░░░ 12.0%
  Mucinous Carcinoma    : █░░░░░░░░░░░░░░░░░░░  6.4%
  Papillary Carcinoma   : ░░░░░░░░░░░░░░░░░░░░  1.0%

1. WHAT WE FOUND
   Your breast tissue sample shows Invasive Ductal Carcinoma (IDC)...

2. IS THIS DANGEROUS?
   DANGER LEVEL: HIGH — but highly treatable when caught early...

3. DO YOU NEED TO SEE A DOCTOR?
   URGENCY: URGENT — please book within the next few days...

[... full 7-section report ...]

ROUGE-1 : 0.1156
ROUGE-L : 0.0889
```

---

## Training (Kaggle)

The models were trained on the BreakHis dataset using Kaggle GPU notebooks:
- `notebooks/01_breast_cancer_classification.ipynb` — DenseNet121 training
- Binary accuracy: **97.1%**
- Benign subtype accuracy: **89.2%**
- Malignant subtype accuracy: **75.4%**

---

## Notes

- The pipeline runs fully on **CPU — no NVIDIA GPU required**
- Mistral-7B runs locally via Ollama (Q4 quantized, ~4.1GB)
- Report generation takes ~2–4 minutes on CPU
- BERTScore evaluation requires torch>=2.4 (rebuild image after first run)
