"""Strategy dataset generator and model trainer."""
from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mathmind.ml.strategy.features import STRATEGY_LABELS, extract_features


def _arithmetic() -> tuple[str, str]:
    a = random.randint(1, 1000)
    b = random.randint(1, 1000)
    op = random.choice(["+", "-", "*", "/"])
    return f"{a} {op} {b}", "direct-arithmetic"


def _linear() -> tuple[str, str]:
    a = random.randint(1, 20)
    b = random.randint(-50, 50)
    c = random.randint(-50, 50)
    return f"{a}*x + {b} = {c}", "linear-isolate"


def _quadratic_formula() -> tuple[str, str]:
    a = random.choice([1, 2, 3])
    b = random.randint(-10, 10)
    c = random.randint(-10, 10)
    return f"{a}*x**2 + {b}*x + {c} = 0", "quadratic-formula"


def _factoring() -> tuple[str, str]:
    r1, r2 = random.randint(-5, 5), random.randint(-5, 5)
    b = -(r1 + r2)
    c = r1 * r2
    return f"x**2 + {b}*x + {c}", "factoring"


def _substitution() -> tuple[str, str]:
    m = random.randint(1, 5)
    k = random.randint(-10, 10)
    a = random.randint(1, 5)
    b = random.randint(1, 5)
    c = random.randint(5, 40)
    return f"y = {m}*x + {k}; {a}*x + {b}*y = {c}", "substitution"


def _elimination() -> tuple[str, str]:
    a1, b1, c1 = random.randint(1, 5), random.randint(1, 5), random.randint(5, 30)
    a2, b2, c2 = random.randint(1, 5), random.randint(-5, -1), random.randint(5, 30)
    return f"{a1}*x + {b1}*y = {c1}; {a2}*x + {b2}*y = {c2}", "elimination"


def _matrix() -> tuple[str, str]:
    a, b, c, d = (
        random.randint(1, 9),
        random.randint(1, 9),
        random.randint(1, 9),
        random.randint(1, 9),
    )
    return f"[[{a},{b}],[{c},{d}]]", "matrix-inverse"


def _differentiate() -> tuple[str, str]:
    n = random.randint(2, 6)
    c = random.randint(1, 9)
    return f"differentiate {c}*x**{n}", "differentiate"


def _integrate() -> tuple[str, str]:
    n = random.randint(1, 5)
    c = random.randint(1, 9)
    return f"integrate {c}*x**{n}", "integrate"


_GENERATORS = [
    _arithmetic,
    _linear,
    _quadratic_formula,
    _factoring,
    _substitution,
    _elimination,
    _matrix,
    _differentiate,
    _integrate,
]


def generate_dataset(n: int = 10000, seed: int = 42) -> tuple[np.ndarray, list[str]]:
    """Generate *n* labeled problems with fixed seed. Returns (X, y)."""
    random.seed(seed)
    np.random.seed(seed)

    pairs: list[tuple[str, str]] = []
    per_class = n // len(_GENERATORS)
    for gen in _GENERATORS:
        for _ in range(per_class):
            pairs.append(gen())
    random.shuffle(pairs)

    problems = [p for p, _ in pairs]
    labels = [lbl for _, lbl in pairs]
    X = np.array([extract_features(p) for p in problems], dtype=float)
    return X, labels


def train_and_evaluate(
    n: int = 10000, save_dir: str = "models", seed: int = 42
) -> dict[str, Any]:
    """Train 3 classifiers, evaluate on held-out split, and save best model."""
    X, y = generate_dataset(n, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    models: dict[str, Any] = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, random_state=seed)),
            ]
        ),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=seed
        ),
    }

    results: dict[str, Any] = {}
    best_name = ""
    best_f1 = 0.0
    best_model = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = float(f1_score(y_test, y_pred, average="macro"))
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=STRATEGY_LABELS).tolist()
        results[name] = {
            "macro_f1": f1,
            "report": report,
            "confusion_matrix": cm,
        }
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    with open(save_path / "strategy_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(save_path / "strategy_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"}
                for k, v in results.items()
            },
            f,
            indent=2,
        )

    results["best_model"] = best_name
    results["best_f1"] = best_f1
    return results
