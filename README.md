# LingoAI — Multilingual Review Intelligence

**LingoAI** analyses customer reviews written in **Urdu script, Roman Urdu, English and code-switched (mixed) text**.
A fine-tuned multilingual Transformer predicts positive / neutral / negative sentiment with calibrated confidence and word-level explanations,
and the workspace adds bulk CSV analysis, session analytics and downloadable reports. No login.

```
Review Input Studio → Language & Script Intelligence → Text Normalisation → Transformer Inference
                    → Sentiment Intelligence → Review Analytics → Reports
```

## What is inside

| Area | Where | What it does |
|---|---|---|
| Data acquisition | `nlp/data_acquisition/` | Downloads 8 public datasets pinned to exact Git commits, records SHA-256 hashes |
| Data preparation | `nlp/data_preparation/build.py` | Label mapping, cleaning, conflict removal, de-duplication, stratified 80/10/10 split by label × language |
| Language & script | `nlp/language_detection/` | Urdu-script share + character n-gram model + learned word lists → `urdu`, `roman_urdu`, `english`, `mixed_roman_english`, `mixed_urdu_english`, `uncertain` |
| Normalisation | `nlp/preprocessing/normalize.py` | NFKC, Arabic→Urdu letter forms, diacritics/tatweel, URLs, mentions, elongation |
| Baseline | `nlp/training/baseline.py` | TF-IDF (char + word n-grams) + logistic regression |
| Model selection | `nlp/training/compare.py` | Multilingual MiniLM vs DistilBERT-multilingual on the same subset |
| Fine-tuning | `nlp/training/finetune.py` | Full fine-tune of `microsoft/Multilingual-MiniLM-L12-H384` on CPU, early stopping on validation macro-F1 |
| Export | `nlp/training/export.py` | Temperature calibration, vocabulary trimming, ONNX + int8 quantisation, test evaluation of the exact served file, error analysis, model card |
| Ensemble | `nlp/training/export_ensemble.py`, `nlp/inference/ensemble.py` | Served model = 35% Transformer + 65% TF-IDF character model (weight and calibration chosen on validation) |
| Inference | `nlp/inference/engine.py` | onnxruntime + tokenizers (no PyTorch in production), sliding windows for long reviews |
| Explainability | `nlp/explainability/occlusion.py` | Leave-one-word-out influence per word |
| Analytics | `nlp/analytics/text_insights.py` | Distinctive words/phrases per sentiment (weighted log-odds), function words learned from data |
| API | `backend/app/` | FastAPI under `/api` (docs at `/api/docs`) |
| Frontend | `frontend/` | React + Vite, 3D landing (react-three-fiber), Plotly charts |

## Datasets

All are public and downloaded from their original repositories at fixed commits (`nlp/data_acquisition/sources.py`):

- Roman Urdu Data Set — Sharf (UCI, CC BY 4.0)
- Daraz product reviews (Roman Urdu) and Urdu product sentiment — `mirfan899/Urdu` (MIT)
- Urdu Sentiment Corpus — Khan et al.
- English: fine-grained sentence sentiment (Täckström & McDonald) and UCI Sentiment Labelled Sentences (Amazon, Yelp, IMDb; CC BY 4.0)

Mixed-language reviews come from inside these datasets and are found by the language detector; no separate code-switched labelled corpus of sufficient size was available.
Final numbers (rows, per-language test metrics, baseline comparison) are in `reports/final_metrics.json` and on the **Model Performance** page.

## Run locally

```bash
# 1. Python (3.12+)
pip install -r requirements-train.txt

# 2. Reproduce data and model (CPU only; full fine-tune takes about an hour on 2 cores)
python -m nlp.data_acquisition.download
python -m nlp.data_preparation.build
python -m nlp.training.baseline
python -m nlp.training.finetune --model <path-or-hub-id-of-Multilingual-MiniLM> \
       --tokenizer xlm-roberta-base --epochs 4 --lr 8e-5 --out models/checkpoints/final
python -m nlp.training.export --ckpt models/checkpoints/final --out models/sentiment
python -m nlp.training.export_ensemble   # TF-IDF model, blend weight on validation, final test metrics

# 3. API
cd backend && uvicorn app.main:app --reload --port 8000

# 4. Frontend (proxies /api to :8000)
cd frontend && npm ci && npm run dev
```

The exported model (`models/sentiment/`, ~45 MB) is committed, so steps 2 can be skipped to just run the app.

## Tests

```bash
cd backend && python -m pytest -q          # NLP unit tests + API tests on the real model
cd frontend && npx playwright test         # browser end-to-end tests (needs API + preview running)
```

## Deploy (Vercel)

One Vercel project serves both parts: the static Vite build and a Python function (`api/index.py`) for `/api/*` (see `vercel.json`).
Only `requirements.txt` (FastAPI, onnxruntime, tokenizers, numpy) is installed in production; training data and pretrained weights are excluded from the function.

## Limits

- Sarcasm and irony are frequently misread.
- Very short reviews carry little signal and are usually flagged as low confidence.
- Training data is product reviews and social media posts; other domains may score lower.
- Mixed Urdu-script + English reviews are rare in the data, so their test estimate is unreliable.
- Reviews are processed in memory; nothing is stored on the server. Session history lives in the browser tab only.

Built by Haniya Imran.
