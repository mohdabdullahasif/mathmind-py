# MathMind 🧮

**An AI-powered math assistant** — solve expressions, do symbolic math, recognize handwriting, process voice input, and export beautiful PDF worksheets.

## Features

| MVP | Feature |
|-----|---------|
| 1 | Custom expression engine (lexer, parser, AST evaluator) — no `eval()` |
| 2 | Variables, user functions, REPL with tab completion |
| 3 | Symbolic math: solve, diff, integrate, simplify, factor (SymPy) |
| 4 | Matrix ops (det, inverse, eigenvalues) & statistics |
| 5 | Natural language translator ("20% off 2400" → answer) |
| 6 | Handwritten expression recognition (CNN + OpenCV) |
| 7 | Beam-search error correction for OCR |
| 8 | Voice input via Whisper + spoken math normalizer |
| 9 | ML strategy predictor (Logistic Regression / Random Forest / GBM) |
| 10 | LaTeX & PDF worksheet export (ReportLab) |
| 11 | Streamlit web app with 5 tabs + SQLite history |

## Install

```bash
git clone https://github.com/mohdabdullahasif/mathmind-py.git
cd mathmind-py
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -e ".[dev]"
```

For optional features:
```bash
pip install -e ".[vision]"  # OpenCV + PyTorch CNN
pip install -e ".[voice]"   # Whisper + sounddevice
pip install -e ".[app]"     # Streamlit UI
```

## Usage

```bash
# Solve an expression
mathmind solve "x**2 - 5*x + 6 = 0"

# Symbolic differentiation
mathmind symbolic "x**3 + 2*x" --op diff

# Natural language
mathmind nl "20% off 2400"

# Matrix
mathmind matrix "[[1,2],[3,4]]" --op det

# Statistics
mathmind stats "2,4,6,8,10"

# Interactive REPL
mathmind repl

# Export PDF
mathmind export "x**2 - 5*x + 6 = 0" --output solution.pdf

# Train strategy model
mathmind train --n 10000

# View history
mathmind history
```

## Streamlit App

```bash
streamlit run src/mathmind/app.py
```

Then open `http://localhost:8501` — 5 tabs: Type, Speak, Draw/Upload, Matrix & Stats, History.

## Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
mathmind/
├── pyproject.toml
├── src/mathmind/
│   ├── core/       lexer, parser, AST, evaluator, simplifier, symbols
│   ├── math/       symbolic.py, matrix.py, steps.py
│   ├── inputs/     nl.py, voice.py, vision/
│   ├── ml/         strategy/ (features, train, predict)
│   ├── export/     latex.py, pdf.py
│   ├── store.py    SQLite persistence
│   ├── cli.py      Typer CLI
│   └── app.py      Streamlit web app
└── tests/
```

## Architecture

```
Voice  → Whisper → normalize_transcript ──┐
Image  → CNN     → beam_search           ──┤
NL     → regex   → translate             ──┼→ expression → Lexer → Parser → AST → Evaluator → Result
Typed  ─────────────────────────────────── ┘                              ↑
                                                                   Simplifier
                                                             SymPy (symbolic ops)
                                                        Strategy ML (predict method)
```

## License

MIT © 2026 mohdabdullaha
