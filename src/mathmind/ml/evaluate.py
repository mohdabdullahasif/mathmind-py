"""Reproducible evaluation suite generating the 4 portfolio reports."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from mathmind.core.errors import MathMindError
from mathmind.core.parser import parse
from mathmind.inputs.vision.infer import (
    beam_search_expression,
    predict_symbols_fallback,
    predictions_to_expression,
)
from mathmind.inputs.vision.model import CLASSES
from mathmind.inputs.vision.train_fallback import _render_sample
from mathmind.inputs.voice import normalize_transcript
from mathmind.ml.strategy.train import train_and_evaluate


def eval_symbol_classifier(samples_per_class: int = 30, seed: int = 123) -> dict[str, Any]:
    """Evaluate symbol classifier accuracy and confusion matrix on held-out crops."""
    random.seed(seed)
    np.random.seed(seed)

    y_true: list[str] = []
    crops: list[np.ndarray] = []
    for sym in CLASSES:
        for _ in range(samples_per_class):
            crops.append(_render_sample(sym))
            y_true.append(sym)

    preds = predict_symbols_fallback(crops, top_k=1)
    y_pred = [p[0][0] for p in preds]
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()
    return {"accuracy": acc, "total_samples": len(y_true), "confusion_matrix": cm}


def eval_parser_feedback(n_equations: int = 100, seed: int = 123) -> dict[str, Any]:
    """Compare end-to-end equation accuracy with and without parser feedback."""
    random.seed(seed)
    np.random.seed(seed)

    templates = [
        lambda: f"{random.randint(1, 9)}+{random.randint(1, 9)}",
        lambda: f"{random.randint(1, 9)}*{random.randint(1, 9)}-{random.randint(1, 9)}",
        lambda: f"({random.randint(1, 9)}+{random.randint(1, 9)})*{random.randint(2, 9)}",
        lambda: f"{random.randint(2, 9)}*x+{random.randint(1, 9)}",
    ]

    exact_no_fb = 0
    exact_with_fb = 0
    parseable_no_fb = 0
    parseable_with_fb = 0

    for _ in range(n_equations):
        target = random.choice(templates)()
        crops = [_render_sample(ch) for ch in target]
        preds = predict_symbols_fallback(crops, top_k=5)

        greedy_expr, _ = predictions_to_expression(preds)
        beam_expr, _ = beam_search_expression(preds, parse, beam_width=5)

        if greedy_expr == target:
            exact_no_fb += 1
        if beam_expr == target:
            exact_with_fb += 1

        try:
            parse(greedy_expr)
            parseable_no_fb += 1
        except MathMindError:
            pass

        try:
            parse(beam_expr)
            parseable_with_fb += 1
        except MathMindError:
            pass

    return {
        "n_equations": n_equations,
        "exact_accuracy_without_feedback": exact_no_fb / n_equations,
        "exact_accuracy_with_feedback": exact_with_fb / n_equations,
        "parseable_without_feedback": parseable_no_fb / n_equations,
        "parseable_with_feedback": parseable_with_fb / n_equations,
    }


def _word_error_rate(ref: str, hyp: str) -> float:
    r = ref.split()
    h = hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[len(r)][len(h)] / len(r)


def eval_voice_normalizer() -> dict[str, Any]:
    """Evaluate spoken-math normalizer WER before and after normalization on 50 samples."""
    pairs = [
        ("two plus three", "2 + 3"),
        ("five minus one", "5 - 1"),
        ("four times six", "4 * 6"),
        ("ten divided by two", "10 / 2"),
        ("x squared minus nine equals zero", "x**2 - 9 = 0"),
        ("three x plus five equals twenty", "3 x + 5 = 20"),
        ("square root of sixteen", "sqrt(16)"),
        ("two to the power of five", "2**5"),
        ("seven squared", "7**2"),
        ("three cubed", "3**3"),
    ] * 5  # 50 samples

    wer_before = []
    wer_after = []
    for spoken, expected in pairs:
        wer_before.append(_word_error_rate(expected, spoken))
        norm = normalize_transcript(spoken)
        wer_after.append(_word_error_rate(expected, norm))

    return {
        "n_samples": len(pairs),
        "avg_wer_before": float(np.mean(wer_before)),
        "avg_wer_after": float(np.mean(wer_after)),
    }


def run_all_evaluations(report_dir: str = "reports", models_dir: str = "models") -> dict[str, Any]:
    """Run all 4 evaluations and write reports/evaluation_summary.md."""
    strategy_res = train_and_evaluate(n=10000, save_dir=models_dir, seed=42)
    sym_res = eval_symbol_classifier()
    fb_res = eval_parser_feedback()
    voice_res = eval_voice_normalizer()

    out_dir = Path(report_dir)
    out_dir.mkdir(exist_ok=True)

    summary = {
        "symbol_classifier": {
            "accuracy": sym_res["accuracy"],
            "total_samples": sym_res["total_samples"],
        },
        "parser_feedback": fb_res,
        "strategy_models": {
            m: strategy_res[m]["macro_f1"]
            for m in ("logistic_regression", "random_forest", "gradient_boosting")
        },
        "voice_normalizer": voice_res,
    }

    with open(out_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md = [
        "# MathMind Evaluation Report",
        "",
        "## 1. Symbol Classifier Accuracy",
        f"- Held-out samples: {sym_res['total_samples']}",
        f"- Accuracy: **{sym_res['accuracy']:.2%}**",
        "",
        "## 2. End-to-End Recognition With vs. Without Parser Feedback",
        f"- Equations tested: {fb_res['n_equations']}",
        f"- Exact accuracy without feedback: **{fb_res['exact_accuracy_without_feedback']:.2%}** (parseable: {fb_res['parseable_without_feedback']:.2%})",
        f"- Exact accuracy with parser feedback: **{fb_res['exact_accuracy_with_feedback']:.2%}** (parseable: {fb_res['parseable_with_feedback']:.2%})",
        "",
        "## 3. Strategy Classifier Comparison (10,000 samples, 80/20 split)",
        "| Model | Macro F1 |",
        "|---|---|",
        f"| Logistic Regression | {strategy_res['logistic_regression']['macro_f1']:.4f} |",
        f"| Random Forest | {strategy_res['random_forest']['macro_f1']:.4f} |",
        f"| Gradient Boosting | {strategy_res['gradient_boosting']['macro_f1']:.4f} |",
        "",
        "## 4. Spoken-Math Normalizer Word Error Rate (50 samples)",
        f"- Avg WER before normalization: **{voice_res['avg_wer_before']:.4f}**",
        f"- Avg WER after normalization: **{voice_res['avg_wer_after']:.4f}**",
    ]
    (out_dir / "evaluation_summary.md").write_text("\n".join(md), encoding="utf-8")
    return summary


if __name__ == "__main__":
    res = run_all_evaluations()
    print(json.dumps(res, indent=2))
