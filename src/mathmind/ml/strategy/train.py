"""Strategy dataset generator and model trainer."""
from __future__ import annotations

import json
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

random.seed(42)
np.random.seed(42)


# ------------------------------------------------------------------ generator
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
    a = random.randint(1, 5)
    b = random.randint(1, 10)
    return f"{a}*x + {b}*y = {random.randint(5, 50)}", "substitution"


def _elimination() -> tuple[str, str]:
    return (
        f"{random.randint(1,3)}*x + {random.randint(1,3)}*y = {random.randint(5,20)}",
        "elimination",
    )


def _matrix() -> tuple[str, str]:
    return "[[1,2],[3,4]]", "matrix-inverse"


def _differentiate() -> tuple[str, str]:
    n = random.randint(2, 5)
    return f"differentiate x**{n}", "differentiate"


def _integrate() -> tuple[str, str]:
    n = random.randint(1, 4)
    return f"integrate x**{n}", "integrate"


_GENERATORS = [
    _arithmetic, _linear, _quadratic_formula, _factoring,
    _substitution, _elimination, _matrix, _differentiate, _integrate,
]


def generate_dataset(n: int = 10000) -> tuple[np.ndarray, list[str]]:
    """Generate *n* labeled problems. Returns (X, y)."""
    problems = []
    labels = []
    per_class = n // len(_GENERATORS)
    for gen in _GENERATORS:
        for _ in range(per_class):
            prob, label = gen()
            problems.append(prob)
            labels.append(label)
    # shuffle
    combined = list(zip(problems, labels))
    random.shuffle(combined)
    problems, labels = zip(*combined)

    X = np.array([extract_features(p) for p in problems])
    return X, list(labels)


# ------------------------------------------------------------------ training
def train_and_evaluate(n: int = 10000, save_dir: str = "models") -> dict[str, Any]:
    X, y = generate_dataset(n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models: dict[str, Any] = {
        "logistic_regression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500, random_state=42))]),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results: dict[str, Any] = {}
    best_name = ""
    best_f1 = 0.0
    best_model = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=STRATEGY_LABELS).tolist()
        results[name] = {"macro_f1": f1, "report": report, "confusion_matrix": cm}
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    # save best
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    import pickle
    with open(save_path / "strategy_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(save_path / "strategy_results.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"} for k, v in results.items()}, f, indent=2)

    results["best_model"] = best_name
    results["best_f1"] = best_f1
    return results
