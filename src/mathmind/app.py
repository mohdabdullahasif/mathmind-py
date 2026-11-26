"""Streamlit web app for MathMind."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="MathMind — AI Math Studio",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

from mathmind.core.engine import solve
from mathmind.core.symbols import SymbolTable
from mathmind.inputs.nl import translate
from mathmind.inputs.voice import normalize_transcript
from mathmind.math.matrix import mat_det, mat_inverse, mat_ops, stats_summary
from mathmind.math.symbolic import (
    symbolic_diff,
    symbolic_expand,
    symbolic_factor,
    symbolic_integrate,
    symbolic_simplify,
    symbolic_solve,
)
from mathmind.ml.strategy.predict import predict as predict_strategy
from mathmind.models import Result
from mathmind.store import get_history, save_result

# ---- custom styling ----
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    .mm-hero {
        background: linear-gradient(135deg, #1E1B4B 0%, #111827 55%, #0F172A 100%);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 16px;
        padding: 1.35rem 1.75rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    .mm-hero h1 {
        margin: 0;
        font-size: 2.05rem;
        font-weight: 800;
        background: linear-gradient(90deg, #A5B4FC, #60A5FA, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .mm-hero p {
        margin: 0.35rem 0 0 0;
        color: #94A3B8;
        font-size: 0.95rem;
    }
    .mm-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.75rem;
    }
    .mm-badge {
        background: rgba(99, 102, 241, 0.16);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #C7D2FE;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .mm-answer-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.14), rgba(59, 130, 246, 0.12));
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .mm-answer-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6EE7B7;
        font-weight: 700;
    }
    .mm-answer-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 0.2rem;
        word-break: break-word;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #131A2A;
        padding: 0.4rem;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.45rem 0.95rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.22) !important;
        color: #C7D2FE !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- session state ----
if "symbols" not in st.session_state:
    st.session_state.symbols = SymbolTable()
if "expr_input" not in st.session_state:
    st.session_state.expr_input = ""

symbols: SymbolTable = st.session_state.symbols


def _show_result(result: Result) -> None:
    if result.error:
        st.error(result.error)
        return

    st.markdown(
        f"""
        <div class="mm-answer-box">
            <div class="mm-answer-label">Result</div>
            <div class="mm-answer-value">{result.value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result.strategy:
        st.caption(f"Strategy: `{result.strategy}` • Confidence: **{result.confidence:.0%}**")

    if result.latex:
        try:
            st.latex(result.latex)
        except Exception:
            pass
        with st.expander("LaTeX Source"):
            st.code(result.latex, language="latex")

    if result.steps:
        with st.expander("Solution Steps", expanded=True):
            for s in result.steps:
                detail = f" — `{s.after}`" if s.after else ""
                st.markdown(f"**{s.step_no}.** {s.description}{detail}")


def _process_handwriting_image(img_path: str, source_name: str) -> None:
    try:
        from mathmind.inputs.vision.infer import predict_symbols, predictions_to_expression
        from mathmind.inputs.vision.model import load_model
        from mathmind.inputs.vision.segment import preprocess, segment_symbols
    except ImportError as exc:
        st.error(f"Missing vision dependency: {exc}")
        return

    binary = preprocess(img_path)
    crops = segment_symbols(binary)
    st.success(f"Detected **{len(crops)}** symbol(s).")

    if crops:
        cols = st.columns(min(len(crops), 8))
        for idx, crop in enumerate(crops[:8]):
            cols[idx].image(crop, caption=f"#{idx + 1}", width=48)

    model_path = Path("models/symbol_cnn.pt")
    if model_path.exists():
        model = load_model(str(model_path))
        preds = predict_symbols(crops, model)
        expr, conf = predictions_to_expression(preds)
        st.info(f"Recognized: `{expr}` (confidence: {conf:.0%})")
        result = solve(expr, symbols, channel="vision")
        _show_result(result)
        save_result(result, raw_input=source_name)
    else:
        st.warning(
            "Symbol segmentation succeeded, but `models/symbol_cnn.pt` is not trained yet. "
            "Train the CNN weights locally and place them in `models/symbol_cnn.pt` to enable recognition."
        )


# ---- sidebar ----
with st.sidebar:
    st.markdown("### ⚙️ Studio Settings")
    angle_mode = st.radio("Angle Mode", ["radians", "degrees"], horizontal=True)
    symbols.angle_mode = angle_mode

    st.markdown("---")
    st.markdown("### ⚡ Quick Examples")
    examples = [
        ("Arithmetic", "2**10 + sqrt(144)"),
        ("Quadratic", "x**2 - 5*x + 6 = 0"),
        ("Derivative", "x**3 + 2*x"),
        ("Trig", "sin(pi / 2) + cos(0)"),
    ]
    for label, sample in examples:
        if st.button(f"{label}: {sample}", use_container_width=True):
            st.session_state.expr_input = sample

    st.markdown("---")
    st.markdown("### 📦 Session Variables")
    if symbols._vars:
        for k, v in symbols._vars.items():
            st.code(f"{k} = {v}")
    else:
        st.caption(f"ans = {symbols.ans}")


# ---- hero header ----
st.markdown(
    """
    <div class="mm-hero">
        <h1>🧮 MathMind Studio</h1>
        <p>Custom AST expression engine, symbolic algebra &amp; calculus, matrix/stats lab, and multi-modal math input.</p>
        <div class="mm-badges">
            <span class="mm-badge">Custom AST Engine</span>
            <span class="mm-badge">SymPy Symbolic</span>
            <span class="mm-badge">ML Strategy Predictor</span>
            <span class="mm-badge">Voice &amp; Handwriting</span>
            <span class="mm-badge">LaTeX / PDF Export</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["✏️ Type & Solve", "🎙️ Speak / Audio", "🖼️ Draw / Upload", "📊 Matrix & Stats", "📜 History"]
)

# ======== Tab 1: Typed Input ========
with tab1:
    left_col, right_col = st.columns([1.15, 1.0], gap="large")

    with left_col:
        st.markdown("#### Expression or Equation")
        expr = st.text_input(
            "Math expression",
            key="expr_input",
            placeholder="e.g. x**2 - 5*x + 6 = 0  or  2*(3 + 4)",
        )
        op = st.selectbox(
            "Operation",
            ["solve (auto)", "solve (equation)", "simplify", "expand", "factor", "diff", "integrate"],
        )
        nl_text = st.text_input(
            "Or ask in plain English",
            placeholder="e.g. 20% off 2400  or  square root of 144",
        )

        btn_col1, btn_col2 = st.columns(2)
        run_clicked = btn_col1.button("Calculate", type="primary", use_container_width=True)
        export_clicked = btn_col2.button("Generate PDF", use_container_width=True)

    with right_col:
        st.markdown("#### Output")
        target_expr = expr
        if run_clicked:
            if nl_text.strip():
                translated_expr, was_t = translate(nl_text)
                if was_t:
                    st.info(f"Translated: `{translated_expr}`")
                target_expr = translated_expr

            if target_expr.strip():
                strategy, confidence, alts = predict_strategy(target_expr)
                if op == "solve (equation)" or (
                    op == "solve (auto)" and "=" in target_expr and "x" in target_expr
                ):
                    result = symbolic_solve(target_expr)
                elif op == "solve (auto)":
                    result = solve(target_expr, symbols, strategy=strategy, confidence=confidence)
                elif op == "simplify":
                    result = symbolic_simplify(target_expr)
                elif op == "expand":
                    result = symbolic_expand(target_expr)
                elif op == "factor":
                    result = symbolic_factor(target_expr)
                elif op == "diff":
                    result = symbolic_diff(target_expr)
                elif op == "integrate":
                    result = symbolic_integrate(target_expr)
                else:
                    result = solve(target_expr, symbols)

                _show_result(result)
                if alts:
                    st.caption(
                        "Alternatives: " + ", ".join(f"{s} ({c:.0%})" for s, c in alts)
                    )
                save_result(result, raw_input=nl_text or target_expr)
            else:
                st.info("Enter an expression or natural-language query on the left.")

        elif export_clicked and expr.strip():
            from mathmind.export.pdf import export_pdf

            strategy, confidence, _ = predict_strategy(expr)
            result = solve(expr, symbols, strategy=strategy, confidence=confidence)
            pdf_path = export_pdf(result, "mathmind_export.pdf")
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Download Solution PDF",
                    f,
                    file_name="mathmind_export.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        else:
            st.caption("Results, LaTeX rendering, and step-by-step explanations appear here.")


# ======== Tab 2: Voice / Transcribe ========
with tab2:
    st.markdown("#### Record Voice or Upload Audio")
    v_col1, v_col2 = st.columns(2, gap="large")

    with v_col1:
        recorded_audio = st.audio_input("Record with your microphone")
        audio_file = st.file_uploader(
            "Or upload an audio file", type=["wav", "mp3", "m4a", "ogg", "webm"]
        )

    with v_col2:
        transcript_manual = st.text_area(
            "Or type / paste spoken math",
            placeholder="e.g. x squared minus nine equals zero",
            height=135,
        )
        process_voice = st.button("Transcribe & Solve", type="primary", use_container_width=True)

    if process_voice:
        audio_source = recorded_audio or audio_file
        transcript = transcript_manual.strip()

        if audio_source is not None:
            suffix = ".wav" if recorded_audio is not None else Path(audio_source.name).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_source.read())
                tmp_path = tmp.name
            try:
                from mathmind.inputs.voice import transcribe_file

                transcript = transcribe_file(tmp_path)
                st.info(f"Transcript: `{transcript}`")
            except Exception as exc:
                st.warning(f"Audio transcription unavailable ({exc}). Using the text box if provided.")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        if transcript:
            normalized = normalize_transcript(transcript)
            st.code(f"Normalized expression: {normalized}")
            if "=" in normalized and any(c.isalpha() for c in normalized):
                result = symbolic_solve(normalized)
                result.channel = "voice"
            else:
                result = solve(normalized, symbols, channel="voice")
            _show_result(result)
            save_result(result, raw_input=transcript)
        else:
            st.info("Record audio, upload a file, or enter a spoken math phrase.")


# ======== Tab 3: Draw / Upload ========
with tab3:
    st.markdown("#### Handwritten Math Recognition")
    d_col1, d_col2 = st.columns(2, gap="large")

    with d_col1:
        st.markdown("**Upload Image**")
        img_file = st.file_uploader("Upload PNG or JPG (max 10 MB)", type=["png", "jpg", "jpeg"])
        if img_file is not None:
            st.image(img_file, caption="Uploaded image", use_container_width=True)
            if st.button("Recognize Uploaded Image", type="primary", use_container_width=True):
                if len(img_file.getvalue()) > 10 * 1024 * 1024:
                    st.error("File too large (max 10 MB)")
                else:
                    suffix = Path(img_file.name).suffix or ".png"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(img_file.getvalue())
                        tmp_path = tmp.name
                    try:
                        _process_handwriting_image(tmp_path, img_file.name)
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)

    with d_col2:
        st.markdown("**Draw on Canvas**")
        try:
            from PIL import Image
            from streamlit_drawable_canvas import st_canvas  # type: ignore[import]

            stroke_w = st.slider("Brush width", 2, 10, 4)
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=stroke_w,
                stroke_color="#000000",
                background_color="#ffffff",
                height=210,
                width=520,
                drawing_mode="freedraw",
                return_image_data=True,
                key="canvas",
            )
            if st.button("Recognize Drawing", type="primary", use_container_width=True):
                if canvas_result.image_data is not None:
                    img = Image.fromarray(canvas_result.image_data.astype(np.uint8))
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        img.save(tmp.name)
                        tmp_path = tmp.name
                    try:
                        _process_handwriting_image(tmp_path, "canvas_drawing.png")
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                else:
                    st.info("Draw an expression on the canvas first.")
        except ImportError:
            st.info("Install `streamlit-drawable-canvas` and `Pillow` to enable canvas drawing.")


# ======== Tab 4: Matrix & Stats ========
with tab4:
    m_col, s_col = st.columns(2, gap="large")

    with m_col:
        st.markdown("#### Matrix Lab")
        mat_expr = st.text_input("Matrix literal", value="[[4,7],[2,6]]", placeholder="[[1,2],[3,4]]")
        mat_op = st.selectbox(
            "Matrix operation", ["det", "inverse", "transpose", "rank", "trace", "eigenvalues"]
        )
        if st.button("Compute Matrix", type="primary", use_container_width=True):
            r = solve(mat_expr, symbols)
            if r.error or not isinstance(r.value, np.ndarray):
                st.error(r.error or "Not a valid matrix")
            else:
                try:
                    if mat_op == "det":
                        result = mat_det(r.value)
                    elif mat_op == "inverse":
                        result = mat_inverse(r.value)
                    else:
                        result = mat_ops(r.value, mat_op)
                    _show_result(result)
                    save_result(result, raw_input=mat_expr)
                except Exception as exc:
                    st.error(str(exc))

    with s_col:
        st.markdown("#### Statistics Lab")
        data_str = st.text_input("Dataset (comma-separated)", value="2, 4, 6, 8, 10")
        if st.button("Compute Stats", type="primary", use_container_width=True):
            try:
                numbers = [float(x.strip()) for x in data_str.split(",") if x.strip()]
                result = stats_summary(numbers)
                if isinstance(result.value, dict):
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Mean", f"{result.value['mean']:.4g}")
                    mc2.metric("Median", f"{result.value['median']:.4g}")
                    mc3.metric("Std Dev", f"{result.value['std']:.4g}")

                    df = pd.DataFrame(list(result.value.items()), columns=["Statistic", "Value"])
                    df["Value"] = df["Value"].apply(lambda v: f"{v:.4g}")
                    st.dataframe(df, use_container_width=True, hide_index=True)
                save_result(result, raw_input=data_str)
            except Exception as exc:
                st.error(str(exc))


# ======== Tab 5: History ========
with tab5:
    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.markdown("#### Recent Calculations")
    with h_col2:
        limit = st.slider("Entries", 5, 100, 20)

    rows = get_history(limit)
    if not rows:
        st.info("No history yet. Solve something in any tab!")
    else:
        df = pd.DataFrame(rows)[["id", "created_at", "channel", "expression", "result", "strategy"]]
        st.dataframe(df, use_container_width=True, hide_index=True)
