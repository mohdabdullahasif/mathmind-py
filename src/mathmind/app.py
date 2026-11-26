"""Streamlit web app for MathMind."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="MathMind", page_icon="🧮", layout="wide")


# ---- imports ----
from mathmind.core.engine import solve
from mathmind.core.symbols import SymbolTable
from mathmind.inputs.nl import translate
from mathmind.math.symbolic import (
    symbolic_diff, symbolic_expand, symbolic_factor,
    symbolic_integrate, symbolic_simplify, symbolic_solve,
)
from mathmind.math.matrix import mat_det, mat_inverse, mat_ops, stats_summary
from mathmind.ml.strategy.predict import predict as predict_strategy
from mathmind.store import get_history, save_result
from mathmind.models import Result

if "symbols" not in st.session_state:
    st.session_state.symbols = SymbolTable()

symbols: SymbolTable = st.session_state.symbols


def _show_result(result: Result) -> None:
    if result.error:
        st.error(result.error)
        return
    st.success(f"**Answer:** {result.value}")
    if result.strategy:
        st.info(f"Strategy: `{result.strategy}` — confidence: {result.confidence:.0%}")
    if result.latex:
        st.code(result.latex, language="latex")
        try:
            st.latex(result.latex)
        except Exception:
            pass
    if result.steps:
        with st.expander("Solution Steps"):
            for s in result.steps:
                st.markdown(f"**Step {s.step_no}:** {s.description}")
                if s.after:
                    st.code(s.after)


st.title("🧮 MathMind")
st.caption("AI-powered math assistant")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["✏️ Type", "🗣️ Speak / Transcribe", "🖼️ Draw / Upload", "📊 Matrix & Stats", "📜 History"])

# ======== Tab 1: Typed Input ========
with tab1:
    st.subheader("Expression Solver")
    col1, col2 = st.columns([3, 1])
    with col1:
        expr = st.text_input("Enter expression or equation", placeholder="e.g. x**2 - 5*x + 6 = 0")
    with col2:
        op = st.selectbox("Operation", ["solve (auto)", "simplify", "expand", "factor", "diff", "integrate"])

    angle_mode = st.radio("Angle mode", ["radians", "degrees"], horizontal=True)
    symbols.angle_mode = angle_mode

    nl_text = st.text_input("Or natural language", placeholder="e.g. 20% off 2400")

    if st.button("Calculate", key="calc_typed"):
        if nl_text:
            translated_expr, was_t = translate(nl_text)
            if was_t:
                st.info(f"Translated: `{translated_expr}`")
            expr = translated_expr

        if expr:
            strategy, confidence, alts = predict_strategy(expr)
            if op == "solve (auto)":
                result = solve(expr, symbols, strategy=strategy, confidence=confidence)
            elif op == "simplify":
                result = symbolic_simplify(expr)
            elif op == "expand":
                result = symbolic_expand(expr)
            elif op == "factor":
                result = symbolic_factor(expr)
            elif op == "diff":
                result = symbolic_diff(expr)
            elif op == "integrate":
                result = symbolic_integrate(expr)
            else:
                result = solve(expr, symbols)

            _show_result(result)
            if alts:
                st.caption(f"Alternatives: {', '.join(f'{s} ({c:.0%})' for s, c in alts)}")
            save_result(result, raw_input=nl_text or expr)

    # PDF export
    if expr and st.button("Export PDF"):
        from mathmind.export.pdf import export_pdf
        strategy, confidence, _ = predict_strategy(expr)
        result = solve(expr, symbols, strategy=strategy, confidence=confidence)
        path = export_pdf(result, "mathmind_export.pdf")
        with open(path, "rb") as f:
            st.download_button("Download PDF", f, file_name="mathmind_export.pdf", mime="application/pdf")


# ======== Tab 2: Voice / Transcribe ========
with tab2:
    st.subheader("Voice / Audio Input")
    st.info("Upload a WAV/MP3 file for Whisper transcription, or type a transcript manually.")

    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3"])
    transcript_manual = st.text_area("Or paste transcript", placeholder="x squared minus nine equals zero")

    if st.button("Process Voice"):
        from mathmind.inputs.voice import normalize_transcript
        if audio_file:
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            try:
                from mathmind.inputs.voice import transcribe_file
                transcript = transcribe_file(tmp_path)
                st.info(f"Transcript: {transcript}")
            except ImportError:
                st.warning("openai-whisper not installed. Using manual transcript.")
                transcript = transcript_manual or ""
            finally:
                os.unlink(tmp_path)
        else:
            transcript = transcript_manual or ""

        if transcript:
            normalized = normalize_transcript(transcript)
            st.code(f"Normalized: {normalized}")
            result = solve(normalized, symbols, channel="voice")
            _show_result(result)
            save_result(result, raw_input=transcript)


# ======== Tab 3: Draw / Upload ========
with tab3:
    st.subheader("Handwritten Expression Recognition")
    st.info("Upload an image of a handwritten math expression.")

    img_file = st.file_uploader("Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if img_file and st.button("Recognize"):
        import tempfile, os
        file_size = len(img_file.getvalue())
        if file_size > 10 * 1024 * 1024:
            st.error("File too large (max 10 MB)")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(img_file.read())
                tmp_path = tmp.name
            try:
                from mathmind.inputs.vision.segment import preprocess, segment_symbols
                from mathmind.inputs.vision.infer import predictions_to_expression
                from mathmind.inputs.vision.model import load_model
                from pathlib import Path

                binary = preprocess(tmp_path)
                crops = segment_symbols(binary)
                st.write(f"Detected {len(crops)} symbols")

                model_path = Path("models/symbol_cnn.pt")
                if model_path.exists():
                    from mathmind.inputs.vision.infer import predict_symbols
                    model = load_model(str(model_path))
                    preds = predict_symbols(crops)
                    expr, conf = predictions_to_expression(preds)
                    st.success(f"Recognized: `{expr}` (confidence: {conf:.0%})")
                    result = solve(expr, symbols, channel="vision")
                    _show_result(result)
                    save_result(result, raw_input=img_file.name)
                else:
                    st.warning("CNN model not trained yet. Run `mathmind train` first.")
            except ImportError as e:
                st.error(f"Missing dependency: {e}")
            finally:
                os.unlink(tmp_path)

    # Canvas drawing
    try:
        from streamlit_drawable_canvas import st_canvas  # type: ignore[import]
        st.markdown("---")
        st.subheader("Or draw on canvas")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            stroke_color="#000000",
            background_color="#ffffff",
            height=200,
            width=600,
            drawing_mode="freedraw",
            return_image_data=True,
            key="canvas",
        )
        if st.button("Recognize Drawing") and canvas_result.image_data is not None:
            st.info("Canvas recognition: save as image and use the uploader above.")
    except ImportError:
        pass


# ======== Tab 4: Matrix & Stats ========
with tab4:
    st.subheader("Matrix Operations")
    mat_expr = st.text_input("Matrix literal", placeholder="[[1,2],[3,4]]")
    mat_op = st.selectbox("Operation", ["det", "inverse", "transpose", "rank", "trace", "eigenvalues"])

    if st.button("Compute Matrix"):
        import numpy as np
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
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    st.subheader("Statistics")
    data_str = st.text_input("Dataset (comma-separated)", placeholder="2, 4, 6, 8, 10")

    if st.button("Compute Stats"):
        try:
            numbers = [float(x.strip()) for x in data_str.split(",") if x.strip()]
            result = stats_summary(numbers)
            if isinstance(result.value, dict):
                import pandas as pd
                df = pd.DataFrame(list(result.value.items()), columns=["Statistic", "Value"])
                df["Value"] = df["Value"].apply(lambda v: f"{v:.4g}")
                st.dataframe(df, use_container_width=True)
            save_result(result, raw_input=data_str)
        except Exception as e:
            st.error(str(e))


# ======== Tab 5: History ========
with tab5:
    st.subheader("Solve History")
    limit = st.slider("Show last N entries", 5, 100, 20)
    rows = get_history(limit)
    if not rows:
        st.info("No history yet. Solve something first!")
    else:
        import pandas as pd
        df = pd.DataFrame(rows)
        df = df[["id", "created_at", "channel", "expression", "result", "strategy"]]
        st.dataframe(df, use_container_width=True)
