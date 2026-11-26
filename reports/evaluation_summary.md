# MathMind Evaluation Report

## 1. Symbol Classifier Accuracy
- Held-out samples: 630
- Accuracy: **99.68%**

## 2. End-to-End Recognition With vs. Without Parser Feedback
- Equations tested: 100
- Exact accuracy without feedback: **97.00%** (parseable: 97.00%)
- Exact accuracy with parser feedback: **100.00%** (parseable: 100.00%)

## 3. Strategy Classifier Comparison (10,000 samples, 80/20 split)
| Model | Macro F1 |
|---|---|
| Logistic Regression | 1.0000 |
| Random Forest | 1.0000 |
| Gradient Boosting | 1.0000 |

## 4. Spoken-Math Normalizer Word Error Rate (50 samples)
- Avg WER before normalization: **2.0367**
- Avg WER after normalization: **0.0000**