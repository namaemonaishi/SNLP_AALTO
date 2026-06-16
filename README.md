# Aalto SNLP Multilingual Toxicity Detection

Course project for **ELEC-E5550 Statistical Natural Language Processing D** at Aalto University. The project studies binary toxicity detection across English, German, and Finnish under cross-lingual data scarcity and class imbalance.

The final system is built around the Detoxify multilingual model with an XLM-RoBERTa backbone and LoRA-based parameter-efficient fine-tuning. The best overall model, **Shared LoRA trained on machine-translated data**, achieved **0.8229 multilingual F1** and **0.8522 accuracy** on the final test set.

## Results at a Glance

| Best achievement | Model | Score |
| --- | --- | --- |
| Best overall multilingual F1 | Shared LoRA on machine-translated data | 0.8229 |
| Best overall accuracy | Shared LoRA on machine-translated data | 0.8522 |
| Best German F1 | Language-specific adapters on machine-translated data | 0.5549 |
| Best Finnish F1 | Language-specific adapters on real multilingual data | 0.9017 |
| Best validation macro F1 | Two-stage fine-tuning | 0.7996 |
| Best validation macro AUC | Two-stage fine-tuning | 0.8696 |
| Best weakest-language validation AUC | Language-specific adapters on machine-translated data | 0.8024 |

Final test-set comparison from the project report:

| Experiment | Multi. F1 | Multi. P | Multi. R | Acc | EN F1 | FI F1 | DE F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear SVM | 0.7486 | 0.7949 | 0.7074 | 0.8093 | 0.9109 | 0.5595 | 0.4812 |
| Zero-Shot | 0.7499 | 0.6510 | 0.8843 | 0.7633 | 0.9662 | 0.8879 | 0.4216 |
| Shared LoRA on MTD | **0.8229** | 0.7925 | **0.8556** | **0.8522** | 0.9619 | 0.8706 | 0.5510 |
| Shared LoRA on RMD | 0.7597 | 0.7007 | 0.8295 | 0.7894 | 0.9599 | 0.8592 | 0.4274 |
| Two-Stage Fine-Tuning | 0.8094 | 0.7779 | 0.8436 | 0.8406 | 0.9596 | 0.8134 | 0.5543 |
| LSA on RMD | 0.7680 | 0.6996 | 0.8512 | 0.7936 | 0.9598 | **0.9017** | 0.4394 |
| LSA on MTD | 0.8164 | **0.7935** | 0.8406 | 0.8483 | **0.9622** | 0.8149 | **0.5549** |

Abbreviations:

- **MTD**: machine-translated data.
- **RMD**: real multilingual data.
- **LSA**: language-specific adapters.

## What This Project Built

### 1. Multilingual data pipelines

The project includes reusable preprocessing scripts for building several multilingual training sets:

| File | Output | Purpose |
| --- | --- | --- |
| `process_datasets_v3.py` | `final_trilingual_dataset.tsv` | Builds a balanced English-German-Finnish dataset with 17,491 examples per language and a 36:64 toxic/non-toxic ratio. |
| `build_full_real_only.py` | `full_train_real_only.tsv` | Builds a larger real/mixed multilingual corpus with about 50k examples per language, preserving more natural toxicity distributions. |
| `merge_round2.py` | `round2_trilingual_dataset.tsv` | Adds GermEval21 German data and Suomi24 Finnish data for the second fine-tuning stage. |

The `clean_text` v3 pipeline removes or normalizes URLs, emails, HTML tags, wiki markup, user mentions, hashtags, timestamps, multilingual date formats, extremely short texts, duplicates, and machine-translation repetition artifacts.

### 2. LoRA fine-tuning experiments

The notebooks implement multiple model strategies on Kaggle:

| Notebook | Main idea |
| --- | --- |
| `notebook-zeroshot-fixed-threshold-patience10.ipynb` | English-only fine-tuning and zero-shot transfer to German and Finnish. |
| `notebook-shared-lora-single-gpu-trans (1).ipynb` | Shared LoRA model trained on machine-translated multilingual data. |
| `notebook-shared-lora-single-gpu-real-fixed-threshold.ipynb` | Shared LoRA model trained on real multilingual data. |
| `notebook-two-stage-fixed-threshold.ipynb` | Stage 1 on translated data, then Stage 2 on real multilingual data. |
| `notebook-three-model-translated-fixed-threshold10p.ipynb` | Three independent language-specific LoRA models trained on translated data. |
| `notebook-three-model-real-fixed-threshold.ipynb` | Three independent language-specific LoRA models trained on real multilingual data. |

Core modeling choices:

- Base model: Detoxify `multilingual`, built on XLM-RoBERTa.
- Fine-tuning: LoRA with rank `r=32`, `lora_alpha=64`, `lora_dropout=0.1`.
- Trainable head: custom binary classifier over the `[CLS]` representation with dropout `0.3`.
- Input length: `256` tokens with head-tail truncation for long comments.
- Loss: focal loss, including per-language alpha values for imbalanced real-data settings.
- Evaluation: per-language validation AUC, macro AUC, macro-optimal F1, weakest-language AUC, and per-language thresholding.

### 3. Main findings

1. Machine-translated data was the strongest overall training signal. Shared LoRA on MTD gave the best multilingual F1 and accuracy.
2. German was the hardest language and the main bottleneck for multilingual generalization.
3. Real multilingual data helped Finnish, especially with language-specific adapters, but did not solve the German bottleneck by itself.
4. Language-specific adapters improved targeted language performance, but shared LoRA remained better for overall multilingual balance.
5. Validation ranking and final test ranking did not always match: two-stage fine-tuning was strongest on validation macro metrics, while Shared LoRA on MTD was strongest on the final test set.

## Repository Structure

```text
.
|-- build_full_real_only.py
|-- merge_round2.py
|-- process_datasets_v3.py
|-- notebook-shared-lora-single-gpu-real-fixed-threshold.ipynb
|-- notebook-shared-lora-single-gpu-trans (1).ipynb
|-- notebook-three-model-real-fixed-threshold.ipynb
|-- notebook-three-model-translated-fixed-threshold10p.ipynb
|-- notebook-two-stage-fixed-threshold.ipynb
`-- notebook-zeroshot-fixed-threshold-patience10.ipynb
```

Raw datasets, trained checkpoints, and Kaggle output files are not committed to this repository.

## Data Requirements

The preprocessing scripts expect the relevant source files in the working directory. Depending on the pipeline, these include:

- `train.tsv`
- `train_fi_deepl.jsonl`
- `de_hf_112024.csv`
- `all_annotations.tsv`
- `GermEval21_TrainData.csv`
- `all_annotations__1_.tsv`
- `final_trilingual_dataset.tsv`

The Kaggle notebooks expect prepared data under paths like:

```text
/kaggle/input/datasets/lyushiro/snlp22/train_cleaned.tsv
/kaggle/input/datasets/lyushiro/snlp22/translated_cleaned.tsv
/kaggle/input/datasets/lyushiro/snlp22/full_train_real_only.tsv
/kaggle/input/datasets/lyushiro/snlp22/round2_trilingual_dataset.tsv
/kaggle/input/datasets/lyushiro/snlp22/dev_cleaned.tsv
/kaggle/input/datasets/lyushiro/snlp22/test_cleaned.tsv
```

## Running the Project

Install the Python dependencies used in the notebooks:

```bash
pip install detoxify peft pytorch-lightning torchmetrics wandb datasets transformers -U
```

Build datasets locally:

```bash
python process_datasets_v3.py
python build_full_real_only.py
python merge_round2.py
```

Train and predict on Kaggle:

1. Upload or attach the prepared TSV/CSV/JSONL files as a Kaggle dataset.
2. Open one of the experiment notebooks.
3. Check that the `/kaggle/input/datasets/lyushiro/snlp22/` paths match your attached dataset.
4. Run all cells. Each notebook writes `train_model.py`, trains the model, saves `result.ckpt` and `threshold.json`, then creates `submission_results.tsv`.

For Weights & Biases logging, configure `WANDB_API_KEY` through Kaggle secrets or environment variables before running experiments.

## Project Contribution Summary

According to the final report division of labor, Shilong Liu led the model and implementation work, including:

- multilingual data preparation;
- machine-translated data construction;
- XLM-RoBERTa and Detoxify fine-tuning;
- LoRA-based experiment design;
- language-specific adapter experiments;
- hyperparameter tuning and threshold calibration;
- final result organization and analysis.

Heini Koivu contributed to the literature review, project planning, evaluation discussion, and final presentation work.

## Report

The results in this README are summarized from the final project report, **"Aalto SNLP Course Competition: Multilingual Toxicity Detection Full Project Report"** by Shilong Liu and Heini Koivu.
