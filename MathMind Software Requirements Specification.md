# MathMind: Software Requirements Specification

*Version 1.0 · Author: Abdullah · November 2026 · Status: Draft for implementation*

## 1. Introduction

### 1.1 Purpose

This document specifies the requirements for **MathMind**, an AI-powered math assistant written in Python. It is the single reference for building the system in 11 incremental MVPs.

### 1.2 Scope

MathMind accepts math input in four ways (typed expressions, plain English, handwritten images, voice), evaluates or solves it with a custom compiler-style engine, explains the solution step by step, and exports results as LaTeX/PDF worksheets. It also provides matrix and statistics tools and an ML model that predicts which solving strategy fits a problem.

**In scope:** single-user desktop/CLI/web app, local processing, offline-capable core. **Out of scope:** multi-user accounts, cloud hosting, mobile apps, proofs and advanced symbolic integration beyond SymPy's capability.

### 1.3 Definitions

| Term | Meaning |
| --- | --- |
| AST | Abstract Syntax Tree built by the parser |
| REPL | Read-Eval-Print Loop (interactive shell) |
| CNN | Convolutional Neural Network used for symbol recognition |
| NL | Natural language input such as "15% of 240" |
| STT | Speech-to-text (Whisper) |
| Strategy | Solving method label, e.g. *linear*, *quadratic-formula*, *factoring*, *substitution*, *matrix-inverse* |

### 1.4 References

Python 3.11+ docs, SymPy, NumPy, PyTorch, OpenCV, OpenAI Whisper (`openai-whisper` or `faster-whisper`), Typer, Streamlit, ReportLab/LaTeX.

## 2. Overall Description

### 2.1 Product perspective

A standalone Python package (`mathmind`) with a shared core engine and four front-ends: CLI/REPL, Streamlit web UI, importable library, and batch exporter. Every input channel (typed, NL, image, voice) converges to a **canonical expression string**, which goes through the same lexer, parser, and evaluator.

```
Voice -> Whisper --\
Image -> CNN ------>  Canonical expression -> Lexer -> Parser(AST) -> Evaluator / Symbolic -> Steps -> UI / Export
NL ---> NL parser -/
Typed ------------/
```

### 2.2 Major functions

1. Parse and evaluate arithmetic and algebraic expressions
2. Variables, user-defined functions, REPL
3. Symbolic simplify, differentiate, integrate, solve with steps
4. Matrix and statistics operations
5. Natural-language to expression translation
6. Handwritten expression recognition
7. Voice input
8. Solving-strategy prediction
9. LaTeX and PDF worksheet export
10. History, persistence, and a web UI

### 2.3 User classes

| User | Needs |
| --- | --- |
| Student | Solve problems and see the steps, input by hand or voice |
| Teacher | Generate worksheets as PDF/LaTeX |
| Developer (you) | Clean modules, tests, evaluation reports for the portfolio |

### 2.4 Operating environment

Windows/Linux/macOS, Python 3.11+, 8 GB RAM minimum, optional CUDA GPU for training (CPU is sufficient for inference). Microphone and camera/file upload optional.

### 2.5 Constraints

- The core engine (lexer, parser, evaluator) must be written by hand, not delegated to `eval()` or SymPy's parser.
- `eval()` and `exec()` are forbidden anywhere on user input.
- All ML inference must run locally by default.
- Datasets must be publicly available or self-generated.

### 2.4 Assumptions and dependencies

Public datasets (MNIST, Kaggle handwritten math symbols) remain available. Whisper `base` or `small` model is downloaded once. LaTeX output is mandatory; PDF compilation uses ReportLab (no TeX install required) with optional `pdflatex`.

## 3. System Features and Functional Requirements

Priority: **M** = must, **S** = should, **C** = could.

### 3.1 Expression engine (MVP 1)

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-1.1 | The lexer shall tokenize numbers (int, float, scientific), operators `+ - * / % ** ^`, parentheses, identifiers, `=`, commas | M |
| FR-1.2 | The parser shall build an AST using recursive descent or Pratt parsing with correct precedence; `**` is right-associative; unary minus binds correctly (`-2**2 = -4`) | M |
| FR-1.3 | The evaluator shall walk the AST and return a numeric result | M |
| FR-1.4 | Implicit multiplication shall be supported (`2x`, `3(4+1)`) | S |
| FR-1.5 | Errors shall report type, message and column position, e.g. `UnexpectedToken ')' at col 7` | M |
| FR-1.6 | Division by zero, overflow, and domain errors shall raise typed exceptions, never crash | M |

**Grammar (EBNF):**

```
expr    = term { ("+" | "-") term } ;
term    = unary { ("*" | "/" | "%") unary } ;
unary   = ["-" | "+"] power ;
power   = primary [ "**" unary ] ;
primary = NUMBER | IDENT | IDENT "(" [args] ")" | "(" expr ")" ;
args    = expr { "," expr } ;
stmt    = IDENT "=" expr | IDENT "(" params ")" "=" expr | expr ;
```

### 3.2 Variables, functions, REPL (MVP 2)

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-2.1 | Variable assignment `x = 5` with a symbol table | M |
| FR-2.2 | Built-ins: `sin cos tan asin acos atan log ln sqrt abs factorial floor ceil round min max` | M |
| FR-2.3 | User functions `f(x) = x**2 + 1` and calls `f(3)` including nested calls | M |
| FR-2.4 | Constants `pi`, `e`, and `ans` (last result) | M |
| FR-2.5 | REPL with history (up/down), syntax highlighting, and tab completion (`prompt_toolkit`) | S |
| FR-2.6 | Angle mode toggle (degrees/radians) | S |

### 3.3 Symbolic math with steps (MVP 3)

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-3.1 | Simplify, expand, factor expressions | M |
| FR-3.2 | Differentiate and integrate (definite and indefinite) | M |
| FR-3.3 | Solve linear, quadratic, and polynomial equations, and 2x2/3x3 linear systems | M |
| FR-3.4 | Own rule-based simplifier on the AST: constant folding, `x+0`, `x*1`, `x*0`, `x-x` | M |
| FR-3.5 | Step generator producing ordered, human-readable steps (e.g. "Subtract 3 from both sides") | M |
| FR-3.6 | Each step shall contain: description, expression before, expression after | M |
| FR-3.7 | Plot a function over a range (matplotlib) | S |

### 3.4 Matrix and statistics mode (MVP 4)

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-4.1 | Matrix literal syntax `[[1,2],[3,4]]` with `+ - *`, scalar multiply, transpose, determinant, inverse, rank, trace | M |
| FR-4.2 | Solve `Ax = b`; eigenvalues and eigenvectors | S |
| FR-4.3 | Statistics on lists: mean, median, mode, variance, std dev, quartiles, min/max, correlation | M |
| FR-4.4 | Linear regression with R-squared and a plot | S |
| FR-4.5 | Distributions: normal and binomial probability functions | C |
| FR-4.6 | Singular/non-square errors shall give clear messages ("matrix is singular") | M |
| FR-4.7 | Where practical, show steps for 2x2 determinant, inverse, and Gaussian elimination | S |

### 3.5 Natural language to math (MVP 5)

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-5.1 | Rule/regex translator for common phrases: "15% of 240", "square root of 81", "5 more than 3 times 7" | M |
| FR-5.2 | Word-problem handler extracting quantities and operations (discounts, tax, averages, simple interest) | S |
| FR-5.3 | ML translator: fine-tuned T5-small trained on template-generated text/expression pairs (min. 20,000 pairs) | S |
| FR-5.4 | The generated expression shall always be displayed for user verification before/with the answer | M |
| FR-5.5 | Translation output is evaluated only by the MathMind engine, never by `eval` | M |
| FR-5.6 | Optional pluggable LLM backend (off by default) | C |

### 3.6 Handwritten expression recognition (MVP 6)

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-6.1 | Accept PNG/JPG upload and canvas drawing | M |
| FR-6.2 | Preprocess: grayscale, denoise, adaptive threshold, deskew | M |
| FR-6.3 | Segment symbols with OpenCV contours, ordered left to right; merge multi-stroke symbols (`=`, `÷`) | M |
| FR-6.4 | CNN classifies 28x28 or 32x32 crops into: digits 0-9, `+ - × ÷ = ( )`, letters `x y z`, `.`, `/` | M |
| FR-6.5 | Output the top-3 predictions per symbol with confidence | M |
| FR-6.6 | Support simple fractions and exponents via vertical-position analysis | C |
| FR-6.7 | Allow the user to correct a misread symbol before solving | S |

### 3.7 Error correction and evaluation (MVP 7)

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-7.1 | If the recognized string fails parsing, try the next-best combinations (beam search over top-k) until the parser accepts one | M |
| FR-7.2 | Rank candidates by joint confidence times syntactic validity | M |
| FR-7.3 | Evaluation script reports symbol accuracy, end-to-end accuracy, and a confusion matrix | M |
| FR-7.4 | Ablation: with versus without parser feedback; custom CNN versus transfer learning | S |

### 3.8 Voice input (MVP 8)

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-8.1 | Record from microphone (push-to-talk) or accept an audio file (wav/mp3) | M |
| FR-8.2 | Transcribe with Whisper (`base` default, configurable) | M |
| FR-8.3 | Normalize spoken math: "x squared plus three x minus five equals zero" becomes `x**2 + 3x - 5 = 0`; handles number words, "over", "times", "to the power of", "point" | M |
| FR-8.4 | Show the transcript and normalized expression for confirmation before solving | M |
| FR-8.5 | Fallback: if normalization fails, pass the transcript to the NL translator (FR-5.1) | S |
| FR-8.6 | Optional text-to-speech readout of the answer | C |

### 3.9 Solving-strategy prediction (MVP 9)

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-9.1 | Define strategy labels (min. 8), e.g. `direct-arithmetic`, `linear-isolate`, `quadratic-formula`, `factoring`, `substitution`, `elimination`, `matrix-inverse`, `differentiate`, `integrate` | M |
| FR-9.2 | Build a labeled dataset: generate 10,000+ problems programmatically, label by the rule that solves them | M |
| FR-9.3 | Features: AST-derived (max degree, number of variables, presence of functions, equation vs expression, matrix present, node counts) plus optional TF-IDF of the text | M |
| FR-9.4 | Train and compare Logistic Regression, Random Forest, and a small gradient-boosting model; report accuracy, macro F1, confusion matrix | M |
| FR-9.5 | Show the predicted strategy with confidence and the top-2 alternatives in the UI | M |
| FR-9.6 | The solver uses the predicted strategy as the default method, falling back to the rule-based choice when confidence is below 0.6 | S |
| FR-9.7 | Log user overrides to improve retraining | C |

### 3.10 Export (MVP 10)

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-10.1 | Export a solution (problem, steps, final answer, optional graph) to `.tex` using proper math formatting | M |
| FR-10.2 | Export to PDF (ReportLab, or `pdflatex` if installed) | M |
| FR-10.3 | Worksheet mode: generate N problems of a chosen type and difficulty, with a separate answer key | S |
| FR-10.4 | Include a title, date, and student-name field on the worksheet | S |
| FR-10.5 | Export history as Markdown/CSV | C |
| FR-10.6 | Convert AST to LaTeX (`\frac`, `\sqrt`, `^{}`) via a dedicated renderer | M |

### 3.11 App, history, packaging (MVP 11)

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-11.1 | Streamlit UI with tabs: Type, Speak, Draw/Upload, Matrix/Stats, History | M |
| FR-11.2 | SQLite history storing input, channel, expression, result, steps, strategy, timestamp | M |
| FR-11.3 | `mathmind` CLI entry point via `pyproject.toml` (`mathmind solve "..."`, `mathmind repl`, `mathmind export`) | M |
| FR-11.4 | README with install steps, screenshots, and a demo GIF | M |
| FR-11.5 | GitHub Actions workflow running tests and linting | S |

## 4. External Interface Requirements

**User interfaces:** CLI/REPL (rich output), Streamlit web app (canvas, file upload, mic button). **Hardware:** microphone (voice), camera or image files (handwriting), GPU optional for training. **Software interfaces:** SymPy, NumPy, SciPy, PyTorch, OpenCV, Whisper, scikit-learn, Hugging Face Transformers, matplotlib, ReportLab, SQLite. **Library API:** `from mathmind import solve; solve("x**2 - 5x + 6 = 0")` returns a `Result` object (`value`, `steps`, `latex`, `strategy`).

## 5. Non-Functional Requirements

| ID | Category | Requirement |
| --- | --- | --- |
| NFR-1 | Performance | Typed expression evaluates in under 100 ms; symbolic solve under 2 s; handwriting recognition under 3 s on CPU; Whisper `base` transcription under 5 s for a 10 s clip |
| NFR-2 | Accuracy | CNN symbol accuracy at least 95%; end-to-end handwritten equation accuracy at least 80% (with parser feedback); strategy classifier macro F1 at least 0.90 on the held-out set |
| NFR-3 | Reliability | No unhandled exceptions on user input; fuzz-test the parser with 1,000+ random strings |
| NFR-4 | Security | No `eval`/`exec`; sandbox-free parsing only; uploaded files validated by type and size (max 10 MB) |
| NFR-5 | Usability | Every error message states what went wrong and how to fix it; every ML output shows confidence |
| NFR-6 | Maintainability | Type hints everywhere, `mypy` clean, `ruff`/`black` formatted, test coverage at least 85% on core modules |
| NFR-7 | Portability | Runs on Windows, Linux, macOS with `pip install` |
| NFR-8 | Privacy | Audio, images, and history stay local unless the user enables an external backend |
| NFR-9 | Reproducibility | Fixed random seeds; training scripts and dataset generators committed; model versions recorded |

## 6. Data Requirements

### 6.1 SQLite schema

```
history(id, created_at, channel, raw_input, expression, result, strategy, confidence, latex)
steps(id, history_id, step_no, description, before, after)
variables(name, value, defined_at)
feedback(id, history_id, kind, corrected_value, created_at)  -- corrections for retraining
model_registry(name, version, metrics_json, trained_at)
```

### 6.2 Datasets

| Dataset | Use | Source |
| --- | --- | --- |
| MNIST | Digits | Public |
| Handwritten math symbols | Operators, letters | Kaggle |
| Own samples | 100+ handwritten equations for end-to-end testing | You collect |
| NL pairs | Text to expression | Template generator (self-built) |
| Strategy set | 10,000+ labeled problems | Generator (self-built) |
| Voice samples | 50+ spoken equations for normalization tests | You record |

## 7. Architecture and Tech Stack

### 7.1 Layers

1. **Input adapters:** typed, NL, vision, voice. Each returns a canonical string plus metadata (confidence, alternatives).
2. **Core engine:** lexer, parser, AST, evaluator, symbol table.
3. **Math services:** symbolic (SymPy plus own rules), matrix/stats (NumPy/SciPy), step generator.
4. **Intelligence:** strategy predictor, error-correction ranker.
5. **Output adapters:** CLI renderer, Streamlit renderer, LaTeX/PDF exporter.
6. **Persistence:** SQLite repository.

### 7.2 Folder structure

```
mathmind/
├── pyproject.toml
├── README.md
├── src/mathmind/
│   ├── core/       lexer.py parser.py ast.py evaluator.py errors.py symbols.py
│   ├── math/       symbolic.py steps.py matrix.py stats.py
│   ├── inputs/     nl.py voice.py vision/{segment.py model.py infer.py}
│   ├── ml/         strategy/{features.py train.py predict.py} nl_model/ datasets/
│   ├── export/     latex.py pdf.py worksheet.py
│   ├── store.py
│   ├── cli.py
│   └── app.py      (Streamlit)
├── notebooks/      training and evaluation
├── models/         saved weights (git-ignored, with a download script)
└── tests/
```

### 7.3 Design patterns

Visitor (AST evaluation, LaTeX rendering, simplification), Strategy (solving methods), Adapter (input channels), Repository (SQLite), Factory (model loading).

### 7.4 Tech stack

Python 3.11, SymPy, NumPy, SciPy, PyTorch, torchvision, OpenCV, Hugging Face Transformers, openai-whisper or faster-whisper, sounddevice, scikit-learn, Typer, prompt\_toolkit, rich, Streamlit, streamlit-drawable-canvas, matplotlib, ReportLab, SQLite, pytest, mypy, ruff.

## 8. MVP Roadmap

Each MVP has a goal, deliverables, and a "done when" test. Do them in order. Every MVP ends with passing tests and a git commit.

| MVP | Name | Covers | Done when |
| --- | --- | --- | --- |
| 1 | Expression engine | FR-1.x | Precedence correct, 50+ unit tests pass |
| 2 | Variables, functions, REPL | FR-2.x | `f(x) = x**2+1` then `f(3)+2` gives 12 |
| 3 | Symbolic math with steps | FR-3.x | `solve x**2 - 5x + 6 = 0` prints steps and `x = 2, 3` |
| 4 | Matrix and statistics | FR-4.x | `det [[1,2],[3,4]]` gives -2; `mean [2,4,6]` gives 4 |
| 5 | Natural language | FR-5.x | "20% off 2400" gives `2400*(1-0.20)` and 1920 |
| 6 | Handwriting recognition | FR-6.x | A photo of `3x+5=20` is read correctly with confidences |
| 7 | Error correction and evaluation | FR-7.x | Table shows parser feedback improves end-to-end accuracy |
| 8 | Voice input (Whisper) | FR-8.x | Saying "x squared minus nine equals zero" returns `x = 3, -3` |
| 9 | Strategy prediction | FR-9.x | Held-out macro F1 at least 0.90; UI shows strategy and confidence |
| 10 | LaTeX/PDF export | FR-10.x | A solved problem and a 10-question worksheet with answer key export to PDF |
| 11 | App and packaging | FR-11.x | `pip install .` then `mathmind repl` works; Streamlit runs all five tabs |

### 8.1 Suggested timeline (about 10 to 12 weeks)

Weeks 1-2: MVP 1-2. Weeks 3-4: MVP 3-4. Week 5: MVP 5. Weeks 6-7: MVP 6-7. Week 8: MVP 8. Week 9: MVP 9. Week 10: MVP 10. Weeks 11-12: MVP 11, polish, report.

### 8.2 Cross-MVP integration points

- MVP 6, 7, and 8 all end at the MVP 1 parser, which is the validity check for every input channel.
- MVP 9 reuses AST features from MVP 1 and strategies implemented in MVP 3 and 4.
- MVP 10 reuses the step list from MVP 3 and the AST-to-LaTeX visitor.

## 9. Testing and Evaluation Plan

| Level | What | Tool |
| --- | --- | --- |
| Unit | Lexer, parser, evaluator, simplifier, matrix/stats functions | pytest, parametrized |
| Property/fuzz | Parser never crashes; `parse(print(ast)) == ast` round-trip | hypothesis |
| Integration | Input channel to answer for each channel | pytest |
| ML evaluation | Accuracy, precision/recall, macro F1, confusion matrices, per-class errors | scikit-learn, notebooks |
| End-to-end | 100 handwritten and 50 spoken equations | custom script |
| Export | Compile generated `.tex`, open PDFs, check answer keys against the solver | script |
| Performance | Time budgets from NFR-1 | pytest-benchmark |

**Required report tables for your portfolio:** (1) CNN symbol accuracy and confusion matrix, (2) end-to-end accuracy with and without parser feedback, (3) strategy classifier comparison across three models, (4) Whisper word error rate on your spoken samples before and after normalization.

## 10. Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Handwriting accuracy is low on your own writing | Heavy augmentation, add your own samples, parser feedback (MVP 7), user correction UI |
| Whisper mishears math words | Normalizer with a math vocabulary, confirmation step, fallback to the NL translator |
| Step generator becomes too complex | Start with linear and quadratic only; add types incrementally |
| T5 training is heavy | Start with rule-based NL (FR-5.1); train T5-small on Colab GPU |
| Strategy labels become trivial (model just learns rules) | Add noisy and mixed problems, evaluate on unseen templates, report honestly |
| Scope creep | Freeze requirements marked C until all M items pass |

## 11. Acceptance Criteria

The project is complete when:

1. All **M** requirements pass their tests.
2. NFR-1 to NFR-4 are met and measured.
3. All four evaluation tables from section 9 are produced.
4. A new user can install the package and solve a typed, spoken, and handwritten problem, then export a PDF, following only the README.

## Appendix A: Example end-to-end flows

1. **Handwritten:** photo of `x² - 5x + 6 = 0` leads to segmentation, CNN, parser check, strategy `quadratic-formula` (0.94), steps, answer `x = 2, 3`, exported to PDF.
2. **Voice:** "mean of two, four, six, ten" leads to Whisper, normalizer `mean([2,4,6,10])`, answer 5.5.
3. **NL:** "A shirt costs 2400 with 20% off, what's the final price?" leads to expression `2400*(1-0.20)`, answer 1920, LaTeX worksheet entry.
4. **Matrix:** `inverse [[4,7],[2,6]]` leads to determinant 10, inverse with steps, LaTeX matrix export.

## Appendix B: Build order checklist

- [ ] MVP 1 Engine
- [ ] MVP 2 Variables and REPL
- [ ] MVP 3 Symbolic and steps
- [ ] MVP 4 Matrix and stats
- [ ] MVP 5 Natural language
- [ ] MVP 6 Handwriting
- [ ] MVP 7 Error correction and evaluation
- [ ] MVP 8 Voice
- [ ] MVP 9 Strategy model
- [ ] MVP 10 Export
- [ ] MVP 11 App and packaging
