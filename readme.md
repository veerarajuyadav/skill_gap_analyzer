# AcadLink Skill Gap Analyzer

Compare a resume with a job description or a supported role title. The analyzer
reports Present, Partial, and Missing skills, coverage, and priority gaps. It
accepts resume text, TXT, DOCX, and text-based PDF files.

## Architecture and technologies

- Python, pandas/NumPy and Parquet data; ESCO/O*NET skill ontology and aliases.
- Rule-based matching with RapidFuzz and explicit resume evidence.
- Optional PyTorch neural classifier: frozen SentenceTransformer MiniLM encoder,
  pooled resume chunks, skill embeddings, lexical features, and trainable fusion
  and classification layers. Embeddings are cached during training; inference
  loads the checkpoint once per analyzer and batches skill predictions.
- `pdfminer.six` reads PDFs; `python-docx` reads Word files. Scanned PDFs require
  external OCR; OCR is not implemented here.

The analyzer distinguishes C, C++, and C#. It uses the existing `k8s -> kubernetes`
abbreviation when Kubernetes is present in the ontology, including normalization
before neural inference. Other abbreviations and model weights are unchanged.

## Dataset and training approach

Dataset C pairs existing resume text with job skill requirements. The improved
builder records evidence/provenance and separates uncertain cases for review.
The original datasets are preserved. Resume-grouped splits contain 92,164 training,
19,766 validation, and 19,732 test rows; labels are Missing (0), Partial (1), and
Present (2). These are constructed, weakly supervised labels.

The experimental baseline freezes MiniLM, caches embeddings, trains the heads
with square-root inverse-frequency class weights, and selects a checkpoint by
validation macro-F1 with early stopping. The selected run used epoch 2. There
are only 28/6/7 Partial examples in train/validation/test. No synthetic Partial
examples were added. The optional severity head is disabled because the improved
splits do not contain severity targets. No retraining is needed to run the app.

## Install

Run from the project root in PowerShell. Python 3.12 was used for validation.
Install the runtime and training dependencies for the implemented paths:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "torch==2.7.1" "sentence-transformers==5.0.0" "transformers==4.54.1" numpy pandas scikit-learn pyarrow rapidfuzz pdfminer.six python-docx
```

`requirements.txt` also lists optional experimentation packages. Existing local
data/model artifacts are required; installing packages does not create them.
The datasets and model directories are Git-ignored. Keep the supplied
`acadlink_skill_gap/data/processed/canonical_skills.parquet` and, for role lookup,
`dataset_B.parquet` there. Neural mode needs both:

```text
acadlink_skill_gap/models/neural_baseline/full_v1/best_model.pt
acadlink_skill_gap/models/neural_baseline/full_v1/frozen_encoder/
```

Keep the checkpoint and its encoder directory together for offline loading.
`training_results.json` beside them records the training configuration and results.

### Large artifacts and Git checkpoints

Git tracks the reviewed source and tests, including the improved Dataset C builder;
generated datasets, caches, weights, and exported encoder files remain ignored.
The current checkpoint is approximately 91.8 MB and its encoder weights 90.9 MB.
A source commit or fresh clone does not contain these artifacts. Preserve the
entire local `full_v1/` directory, including the tokenizer, pooling configuration,
and `training_results.json`. Do not commit datasets or force-add model directories.
For transfer or backup, copy the bundle separately and keep its directory layout.

The supplied model files can be identified with these SHA-256 hashes:

```text
best_model.pt:  e01c0d457dc9ad50c6ab7492cf1e4cbdc7da6b1cdf8d312b320d95beb1d7289f
frozen_encoder/model.safetensors:  1377e9af0ca0b016a9f2aa584d6fc71ab3ea6804fae21ef9fb1416e2944057ac
```

## Rule-based analysis (default)

```powershell
python -B skill_gap_analyzer.py --resume "Python developer deploying services to K8s." --role "API Developer"
python -B skill_gap_analyzer.py --resume-file resume.docx --role "Platform Engineer" --job-description-file job.txt --output rule_result.json
```

`--resume-file` also accepts `.pdf` and `.txt`. Supply either `--resume` or
`--resume-file`. Use `--job-description "Required skills: Python, SQL, Kubernetes"`
or a UTF-8 `--job-description-file`; otherwise role lookup requires an exact
supported title. With a supplied JD, `--role` is its display title. Explicit
comma/semicolon skill lists preserve unknown names; prose matching uses complete
ontology phrases. Default JSON output is `skill_gap_result.json`.

## Neural analysis (experimental)

```powershell
python -B skill_gap_analyzer.py --resume-file resume.docx --role "Platform Engineer" --job-description-file job.txt --neural --device cpu --output neural_result.json
```

Omit `--device cpu` to select CUDA only when available. Use `--checkpoint PATH`
for another compatible checkpoint. A load or inference failure logs the reason
and falls back to rules. Existing output fields remain available; neural mode
adds probabilities, rule comparisons, and matching/fallback metadata. Partial
skills remain included in the legacy gap lists because they are not fully Present.

## Validation and limitations

```powershell
python -B -m unittest discover -s acadlink_skill_gap/tests -p "test*.py" -v
```

Tests cover existing behavior, symbol boundaries, aliases, file parsing, neural
CPU inference, checkpoint reuse, and failure fallback. Real-checkpoint tests are
skipped when model artifacts are absent. The dataset builder also has embedded
policy tests runnable with `--self-test`; the command above does not rebuild data.

Weak-label test scores measure agreement with labeling rules, not validated
real-world accuracy or candidate competence. Partial support is too small for
reliable three-class conclusions, and class weights do not supply missing
diversity. Probabilities are uncalibrated. Missing means no sufficient evidence
in the supplied resume, not proof that a person lacks a skill. Legacy extraction
can emit noisy extra candidates; aliases, paraphrases, and lost PDF text can
still cause errors. Review requirements and predictions before using them.
