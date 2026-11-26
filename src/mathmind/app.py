"""Streamlit web app for MathMind."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from mathmind.core.engine import solve
from mathmind.core.errors import MathMindError
from mathmind.core.parser import parse
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

st.set_page_config(page_title="MathMind", layout="wide")

if "symbols" not in st.session_state:
    st.session_state.symbols = SymbolTable()
if "expr_input" not in st.session_state:
    st.session_state.expr_input = ""

symbols: SymbolTable = st.session_state.symbols


def _show_result(result: Result) -> None:
    if result.error:
        st.error(result.error)
        return

    st.success(f"Result: {result.value}")

    if result.strategy:
        st.caption(f"Strategy: {result.strategy} (confidence: {result.confidence:.0%})")

    if result.latex:
        st.latex(result.latex)
        with st.expander("LaTeX"):
            st.code(result.latex, language="latex")

    if result.steps:
        with st.expander("Steps", expanded=True):
            for s in result.steps:
                detail = f" -> {s.after}" if s.after else ""
                st.text(f"{s.step_no}. {s.description}{detail}")


def _process_handwriting_image(img_path: str, source_name: str) -> None:
    try:
        from mathmind.inputs.vision.infer import (
            beam_search_expression,
            predict_symbols,
            predict_symbols_fallback,
        )
        from mathmind.inputs.vision.model import load_model
        from mathmind.inputs.vision.segment import preprocess, segment_symbols
    except ImportError as exc:
        st.error(f"Missing vision dependency: {exc}")
        return

    binary = preprocess(img_path)
    crops = segment_symbols(binary)
    st.write(f"Detected {len(crops)} symbol(s)")

    if not crops:
        st.warning("No symbols detected. Draw thicker strokes and try again.")
        return

    cols = st.columns(min(len(crops), 8))
    for idx, crop in enumerate(crops[:8]):
        cols[idx].image(crop, caption=f"#{idx + 1}", width=48)

    model_path = Path("models/symbol_cnn.pt")
    preds = None
    if model_path.exists():
        try:
            model = load_model(str(model_path))
            preds = predict_symbols(crops, model)
        except ImportError:
            preds = None

    if preds is None:
        preds = predict_symbols_fallback(crops)

    expr, conf = beam_search_expression(preds, parse)
    st.info(f"Recognized: {expr} (confidence: {conf:.0%})")
    if "=" in expr and any(c.isalpha() for c in expr):
        result = symbolic_solve(expr)
        result.channel = "vision"
    else:
        result = solve(expr, symbols, channel="vision")
    _show_result(result)
    save_result(result, raw_input=source_name)


with st.sidebar:
    st.subheader("Settings")
    angle_mode = st.radio("Angle mode", ["radians", "degrees"], horizontal=True)
    symbols.angle_mode = angle_mode

    st.divider()
    st.subheader("Examples")
    examples = [
        "2**10 + sqrt(144)",
        "x**2 - 5*x + 6 = 0",
        "x**3 + 2*x",
        "sin(pi / 2) + cos(0)",
    ]
    for sample in examples:
        if st.button(sample, use_container_width=True):
            st.session_state.expr_input = sample

    st.divider()
    st.subheader("Variables")
    if symbols._vars:
        for k, v in symbols._vars.items():
            st.text(f"{k} = {v}")
    else:
        st.text(f"ans = {symbols.ans}")


st.title("MathMind")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Type", "Speak", "Draw / Upload", "Matrix & Stats", "History"]
)

with tab1:
    left_col, right_col = st.columns([1.15, 1.0], gap="large")

    with left_col:
        expr = st.text_input(
            "Expression",
            key="expr_input",
            placeholder="e.g. x**2 - 5*x + 6 = 0",
        )
        op = st.selectbox(
            "Operation",
            ["solve (auto)", "solve (equation)", "simplify", "expand", "factor", "diff", "integrate"],
        )
        nl_text = st.text_input(
            "Natural language",
            placeholder="e.g. 20% off 2400",
        )

        btn_col1, btn_col2 = st.columns(2)
        run_clicked = btn_col1.button("Solve", type="primary", use_container_width=True)
        export_clicked = btn_col2.button("Export PDF", use_container_width=True)

    with right_col:
        target_expr = expr
        if run_clicked:
            if nl_text.strip():
                translated_expr, was_t = translate(nl_text)
                if was_t:
                    st.info(f"Translated: {translated_expr}")
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
                st.info("Enter an expression or natural-language query.")

        elif export_clicked and expr.strip():
            from mathmind.export.pdf import export_pdf

            strategy, confidence, _ = predict_strategy(expr)
            result = solve(expr, symbols, strategy=strategy, confidence=confidence)
            pdf_path = export_pdf(result, "mathmind_export.pdf")
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Download PDF",
                    f,
                    file_name="mathmind_export.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )


with tab2:
    v_col1, v_col2 = st.columns(2, gap="large")

    with v_col1:
        recorded_audio = st.audio_input("Record audio")
        audio_file = st.file_uploader(
            "Upload audio", type=["wav", "mp3", "m4a", "ogg", "webm"]
        )

    with v_col2:
        transcript_manual = st.text_area(
            "Transcript",
            placeholder="x squared minus nine equals zero",
            height=135,
        )
        process_voice = st.button("Solve voice input", type="primary", use_container_width=True)

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
                st.info(f"Transcript: {transcript}")
            except (ImportError, OSError, ValueError, RuntimeError) as exc:
                st.warning(f"Audio transcription unavailable ({exc}). Using transcript text.")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        if transcript:
            normalized = normalize_transcript(transcript)
            st.code(f"Normalized: {normalized}")
            if "=" in normalized and any(c.isalpha() for c in normalized):
                result = symbolic_solve(normalized)
                result.channel = "voice"
            else:
                result = solve(normalized, symbols, channel="voice")
            _show_result(result)
            save_result(result, raw_input=transcript)
        else:
            st.info("Record audio, upload a file, or enter a transcript.")


with tab3:
    d_col1, d_col2 = st.columns(2, gap="large")

    with d_col1:
        img_file = st.file_uploader("Upload image (PNG/JPG, max 10 MB)", type=["png", "jpg", "jpeg"])
        if img_file is not None:
            st.image(img_file, caption="Uploaded image", use_container_width=True)
            if st.button("Recognize image", type="primary", use_container_width=True):
                if len(img_file.getvalue()) > 10 * 1024 * 1024:
                    st.error("File size exceeds 10 MB limit.")
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
        try:
            from PIL import Image
            from streamlit_drawable_canvas import st_canvas  # type: ignore[import-untyped]

            stroke_w = st.slider("Brush width", 2, 10, 4)
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=stroke_w,
                stroke_color="#000000",
                background_color="#ffffff",
                height=210,
                width=520,
                drawing_mode="freedraw",
                return_image_data=True,
                key="canvas",
            )
            if st.button("Recognize drawing", type="primary", use_container_width=True):
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
            st.info("Install streamlit-drawable-canvas and Pillow to enable drawing.")


with tab4:
    m_col, s_col = st.columns(2, gap="large")

    with m_col:
        st.subheader("Matrix")
        mat_expr = st.text_input("Matrix literal", value="[[4,7],[2,6]]")
        mat_op = st.selectbox(
            "Matrix operation", ["det", "inverse", "transpose", "rank", "trace", "eigenvalues"]
        )
        if st.button("Compute matrix", type="primary", use_container_width=True):
            r = solve(mat_expr, symbols)
            if r.error or not isinstance(r.value, np.ndarray):
                st.error(r.error or "Invalid matrix literal.")
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
                except MathMindError as exc:
                    st.error(str(exc))

    with s_col:
        st.subheader("Statistics")
        data_str = st.text_input("Dataset (comma-separated)", value="2, 4, 6, 8, 10")
        if st.button("Compute stats", type="primary", use_container_width=True):
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
            except ValueError as exc:
                st.error(str(exc))


with tab5:
    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.subheader("History")
    with h_col2:
        limit = st.slider("Entries", 5, 100, 20)

    rows = get_history(limit)
    if not rows:
        st.info("No history yet.")
    else:
        df = pd.DataFrame(rows)[["id", "created_at", "channel", "expression", "result", "strategy"]]
        st.dataframe(df, use_container_width=True, hide_index=True)
