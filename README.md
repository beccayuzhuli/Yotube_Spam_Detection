# Hybrid LLM-ML Architecture for YouTube Spam Detection

This repository recreates the RSM8421 final project workflow for YouTube comment spam detection. The project combines lexical TF-IDF features with frozen RoBERTa sentence embeddings, then trains lightweight classical machine learning classifiers for efficient binary spam classification.

The target task is:

- `CLASS = 1`: spam comment
- `CLASS = 0`: non-spam comment

## Project Summary

YouTube comment spam often uses repeated promotional phrases, URLs, altered spelling, and self-promotion patterns such as "check my channel" or "subscribe". A pure rule-based detector is fast but brittle, while full LLM inference or transformer fine-tuning can be expensive for a small labeled dataset.

This project uses RoBERTa as a frozen feature extractor instead of a fine-tuned model. The semantic embeddings are concatenated with word-level and character-level TF-IDF features, then passed into classical classifiers.

## Problem Statement

YouTube is one of the largest video platforms in the world, with hundreds of hours of video uploaded every minute and massive daily comment volume. Its open comment system makes it a frequent target for spam.

The project focuses on three common spam patterns from the presentation:

| Spam type | Example pattern | Why it matters |
| --- | --- | --- |
| Self-promotion | "check my channel" | Promotes unrelated channels or content |
| Phishing or links | "free at bit.ly/..." | Sends users to external URLs |
| Evasion | "ch3ck 0ut" | Alters spelling to avoid simple filters |

The central challenge is balancing accuracy and efficiency. Rule-based systems are fast but weak, while fine-tuned BERT/RoBERTa or GPT-style inference can be accurate but costly. This project tests whether hybrid features can provide LLM-level signal with traditional ML efficiency.

## Research Questions

1. Can hybrid TF-IDF and RoBERTa features outperform either feature type alone?
2. Can a frozen transformer feature extractor achieve strong accuracy without expensive fine-tuning?
3. Are lexical spam indicators and semantic embeddings complementary rather than redundant?

## Hypothesis

Lexical features and semantic features capture different signals:

```text
TF-IDF: exact spam phrases, URLs, repeated words, misspellings
RoBERTa: meaning, context, paraphrases, semantic similarity
Hybrid: both lexical precision and semantic understanding
```

The expected result is that concatenating TF-IDF and RoBERTa embeddings improves discrimination over either feature family alone.

## Reported Results

The class presentation reported the following final hybrid ensemble performance on an 80/20 stratified split of 1,956 YouTube comments:

| Metric | Score |
| --- | ---: |
| Accuracy | 97.2% |
| Precision | 98.0% |
| Recall | 96.5% |
| F1 score | 97.2% |
| ROC-AUC | 99.5% |

Feature ablation from the presentation:

| Feature set | Accuracy | F1 | ROC-AUC |
| --- | ---: | ---: | ---: |
| TF-IDF only | 95.47% | 95.53% | 98.66% |
| RoBERTa only | 94.83% | 94.85% | 98.64% |
| Hybrid | 96.88% | 96.91% | 99.31% |

## Reproduced KaggleHub Run

Using `ahsenwaheed/youtube-comments-spam-dataset`, the prepared dataset contains 1,956 comments:

| Class | Count |
| --- | ---: |
| Non-spam (`0`) | 951 |
| Spam (`1`) | 1,005 |

Live results from the repository code with an 80/20 stratified split:

| Feature set / classifier | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF / Logistic Regression | 94.90% | 96.41% | 93.53% | 94.95% | 98.77% |
| TF-IDF / Random Forest | 96.17% | 97.45% | 95.02% | 96.22% | 98.90% |
| TF-IDF / XGBoost | 94.39% | 94.97% | 94.03% | 94.50% | 98.22% |
| TF-IDF / LightGBM | 95.15% | 95.96% | 94.53% | 95.24% | 98.32% |
| TF-IDF / Soft-vote ensemble | 94.90% | 95.02% | 95.02% | 95.02% | 98.97% |
| RoBERTa / Logistic Regression | 95.41% | 95.98% | 95.02% | 95.50% | 99.27% |
| Hybrid / Soft-vote ensemble | 96.17% | 97.45% | 95.02% | 96.22% | 99.64% |

## Architecture

```text
comment text
  -> preprocessing
  -> word TF-IDF n-grams, 1-4, 5,000 features
  -> char TF-IDF n-grams, 2-6, 5,000 features
  -> frozen roberta-base mean-pooled embeddings, 768 features
  -> concatenate into 10,768-dimensional hybrid feature vector
  -> Logistic Regression / Random Forest / XGBoost / LightGBM
  -> optional soft-voting ensemble
```

The implementation keeps RoBERTa optional so the TF-IDF baseline can run on machines without PyTorch or HuggingFace Transformers.

## Dataset Profile

The PPT uses the YouTube Spam Collection dataset from Kaggle, covering comments from five music videos:

| Source video |
| --- |
| PSY - Gangnam Style |
| Katy Perry - Roar |
| Eminem - Rap God |
| Shakira - Waka Waka |
| LMFAO - Party Rock |

The dataset has a nearly balanced target distribution:

| Class | Count | Share |
| --- | ---: | ---: |
| Non-spam | 951 | 48.6% |
| Spam | 1,005 | 51.4% |
| Total | 1,956 | 100.0% |

Descriptive patterns from the PPT:

| Metric | Non-spam | Spam | Difference |
| --- | ---: | ---: | ---: |
| Average character length | 75.2 | 94.3 | +25% |
| Average word count | 14.1 | 17.8 | +26% |
| Average URL count | 0.014 | 0.238 | +17x |
| Contains "subscribe" | 2.1% | 18.4% | +9x |
| Contains "check out" | 1.5% | 22.7% | +15x |

These patterns explain why TF-IDF is a strong baseline: spam has distinctive surface-level vocabulary and phrase structure.

## Exploratory Text Patterns

The PPT vocabulary analysis showed moderate overlap between spam and non-spam words:

| Vocabulary group | Count |
| --- | ---: |
| Non-spam unique words | 2,847 |
| Spam unique words | 3,124 |
| Shared words | 1,456 |
| Jaccard similarity | 0.32 |

Typical non-spam words are content-focused:

```text
love, song, amazing, great, music, best
```

Typical spam words are promotional:

```text
check, my, subscribe, channel, out, please
```

This supports the hybrid design. TF-IDF detects the repeated promotional vocabulary, while RoBERTa helps detect semantic variants that do not use the exact same words.

## Technical Background

### TF-IDF

TF-IDF stands for term frequency-inverse document frequency. It gives more weight to terms that are frequent in one comment but uncommon across the full dataset.

This project uses two TF-IDF branches:

| Branch | N-gram range | Max features | Purpose |
| --- | ---: | ---: | --- |
| Word TF-IDF | 1-4 | 5,000 | Captures phrases like "check out my channel" |
| Character TF-IDF | 2-6 | 5,000 | Captures misspellings like "ch3ck" or "subscr1be" |

TF-IDF strengths:

- Fast and lightweight
- Strong for repeated spam vocabulary
- Interpretable
- Effective for URLs, keywords, and spelling evasion

TF-IDF limitations:

- Does not understand semantic similarity
- Treats many paraphrases as unrelated
- Uses bag-of-words style assumptions
- Produces high-dimensional sparse features

### RoBERTa

RoBERTa is a transformer encoder model based on BERT-style self-attention. The PPT highlights these RoBERTa properties:

| Property | Value |
| --- | ---: |
| Layers | 12 |
| Attention heads | 12 |
| Hidden dimension | 768 |
| Parameters | 125M |

RoBERTa improves on BERT through design choices such as dynamic masking, removing the next sentence prediction objective, larger batches, and more pretraining data.

In this project, RoBERTa is not fine-tuned. It is used as a frozen embedding model:

```text
comment -> roberta-base -> mean pooling -> 768-dimensional vector
```

This gives the classifier semantic information without requiring GPU training.

## Methodology

The full hybrid feature vector is:

```text
X_hybrid = [X_TF-IDF || X_RoBERTa]
```

Dimension breakdown:

| Feature block | Dimensions | Share |
| --- | ---: | ---: |
| Word TF-IDF | 5,000 | 46.4% |
| Character TF-IDF | 5,000 | 46.4% |
| RoBERTa embedding | 768 | 7.1% |
| Total hybrid features | 10,768 | 100.0% |

The downstream classifiers are:

| Classifier | Role |
| --- | --- |
| Logistic Regression | Linear baseline for high-dimensional text features |
| Random Forest | Bagged tree model for nonlinear interactions |
| XGBoost | Gradient boosting model |
| LightGBM | Efficient histogram-based gradient boosting |
| Soft-vote ensemble | Averages predicted probabilities from all classifiers |

Soft voting uses predicted probabilities rather than hard class labels:

```text
P(spam | x) = average model probability
prediction = spam if P(spam | x) > 0.5
```

## Why Not Fine-Tune RoBERTa?

The project intentionally avoids fine-tuning RoBERTa because:

- The dataset is small, with about 2,000 comments.
- Fine-tuning a 125M parameter model could overfit.
- Frozen embeddings are easier to combine with TF-IDF features.
- The approach can run locally without GPU training.
- Classical ML inference is faster and cheaper for deployment.

## Complementary Feature Strengths

| Capability | TF-IDF | RoBERTa | Hybrid |
| --- | ---: | ---: | ---: |
| Exact keyword matching | Yes | No | Yes |
| URL and phrase detection | Yes | Limited | Yes |
| Misspelling detection | Yes | Limited | Yes |
| Semantic similarity | No | Yes | Yes |
| Paraphrase detection | No | Yes | Yes |
| Context understanding | No | Yes | Yes |
| Interpretability | High | Low | Medium |

The PPT's key message is that the two feature types are additive, not redundant.

## Error Analysis

The presentation reported a small number of classification errors for the final model:

| Error type | Count | Example pattern | Likely cause |
| --- | ---: | --- | --- |
| False positive | 4 | "Love this! Check out my reaction video" | Genuine fan comment contains spam-like phrase |
| False negative | 7 | "Amazing! More quality content on my profile" | Subtle self-promotion without obvious spam keywords |

Main error drivers:

- The boundary between engagement and self-promotion is ambiguous.
- TF-IDF can overweight words like "check", "my", or "channel".
- Frozen RoBERTa embeddings may miss domain-specific spam cues.

## Repository Layout

```text
.
|-- README.md
|-- requirements.txt
|-- requirements-optional.txt
|-- scripts/
|   `-- download_dataset.py
|-- src/
|   `-- youtube_spam_detection/
|       |-- __init__.py
|       |-- features.py
|       |-- train.py
|       `-- utils.py
`-- tests/
    `-- test_utils.py
```

## Data

The project uses the KaggleHub dataset:

- Extended public dataset: https://www.kaggle.com/datasets/ahsenwaheed/youtube-comments-spam-dataset

Expected columns:

| Column | Description |
| --- | --- |
| `COMMENT_ID` | unique comment id |
| `AUTHOR` | commenter username |
| `DATE` | timestamp |
| `CONTENT` | comment text |
| `CLASS` | binary spam label |

The training CLI only requires `CONTENT` and `CLASS`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

For the TF-IDF baseline only, install:

```bash
python -m pip install pandas numpy scipy scikit-learn joblib
```

For hybrid RoBERTa features, also install PyTorch and Transformers:

```bash
python -m pip install -r requirements-optional.txt
```

## Download Data

```bash
python scripts/download_dataset.py
```

This uses:

```python
import kagglehub

path = kagglehub.dataset_download("ahsenwaheed/youtube-comments-spam-dataset")
```

The script combines all matching CSV files into:

```text
data/youtube_comments_spam.csv
```

## Usage

Run the TF-IDF baseline:

```bash
python -m youtube_spam_detection.train --data data/youtube_comments_spam.csv --feature-set tfidf
```

Run RoBERTa-only features:

```bash
python -m youtube_spam_detection.train --data data/youtube_comments_spam.csv --feature-set roberta
```

Run the hybrid feature model:

```bash
python -m youtube_spam_detection.train --data data/youtube_comments_spam.csv --feature-set hybrid
```

Save a fitted pipeline:

```bash
python -m youtube_spam_detection.train --data data/youtube_comments_spam.csv --feature-set tfidf --model-out artifacts/tfidf_model.joblib
```

## Model Choices

- TF-IDF captures exact spam vocabulary, URLs, n-grams, and misspellings.
- Character n-grams help detect spelling evasion such as `ch3ck`, `subscr1be`, or obfuscated links.
- RoBERTa embeddings capture semantic similarity and paraphrases that lexical features may miss.
- Classical classifiers keep inference efficient and interpretable compared with full transformer inference.

## Key Takeaways

1. TF-IDF remains very strong for YouTube spam because spam contains repeated lexical signatures.
2. RoBERTa embeddings add semantic information that TF-IDF cannot capture.
3. Hybrid TF-IDF plus RoBERTa features improve ROC-AUC in the reproduced run.
4. Frozen transformer embeddings are a practical alternative to fine-tuning on small datasets.
5. Soft-voting ensembles can improve robustness by combining different model assumptions.

## Limitations

- The reported results are based on a relatively small YouTube spam collection centered on music video comments.
- Frozen RoBERTa embeddings are not adapted to the spam domain.
- Text-only classification ignores user history, posting frequency, video context, and network behavior.
- Spam tactics evolve, so production use would require monitoring and periodic retraining.

## Future Work

- Add user and channel behavior features.
- Fine-tune RoBERTa on a larger labeled spam corpus.
- Calibrate thresholds for moderation workflows with different false-positive costs.
- Add drift monitoring for new spam campaigns.
