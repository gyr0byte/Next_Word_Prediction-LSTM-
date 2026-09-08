from src.prediction import generate_text
from src.model import load_prediction_assets
import streamlit as st
from pathlib import Path
import pickle
import sys
from html import escape


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


st.set_page_config(
    page_title="Next Word Studio",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
	<style>
	@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

	:root {
        color-scheme: dark;
        --ink: #edf6f1;
        --muted: #a4b9b4;
        --paper: #0b1517;
        --panel: #132326;
        --line: #2a4446;
        --teal: #73d4c9;
        --coral: #ff866f;
        --yellow: #f5cb6b;
	}

	.stApp {
		background: var(--paper);
		color: var(--ink);
		font-family: 'DM Sans', sans-serif;
	}

	[data-testid="stHeader"] { background: transparent; }
	[data-testid="stSidebar"] {
        background: #101f22;
		border-right: 1px solid var(--line);
	}
	[data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] label { color: var(--ink) !important; }

	.hero {
		padding: 1.5rem 0 2.25rem;
		max-width: 860px;
	}
	.eyebrow {
		color: var(--coral);
		font-size: 0.76rem;
		font-weight: 700;
		letter-spacing: 0.14em;
		text-transform: uppercase;
	}
	h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink) !important; }
	h1 { font-size: clamp(2.7rem, 6vw, 5.4rem) !important; line-height: 0.96 !important; letter-spacing: -0.04em !important; margin: 0.65rem 0 1rem !important; }
	.hero-copy { color: var(--muted); font-size: 1.08rem; line-height: 1.6; max-width: 620px; }

	.workspace {
		background: var(--panel);
		border: 1px solid var(--line);
		border-radius: 10px;
		padding: 1.25rem 1.35rem 1.35rem;
        box-shadow: 0 14px 35px rgba(0, 0, 0, 0.24);
	}
	.section-label {
		color: var(--teal);
		font-size: 0.78rem;
		font-weight: 700;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		margin-bottom: 0.55rem;
	}
	.result-label { margin-top: 1.1rem; }
	.result-box {
        background: #071c1d;
        border: 1px solid #2b5a58;
		border-radius: 8px;
		color: #f7f4e9;
		font-family: 'Space Grotesk', sans-serif;
		font-size: clamp(1.2rem, 2.2vw, 1.7rem);
		line-height: 1.45;
		min-height: 145px;
		padding: 1.3rem 1.4rem;
	}
	.placeholder { color: #a9c2bb; }
	.tip {
        background: #302b1d;
		border-left: 4px solid var(--yellow);
		border-radius: 4px;
        color: #f5d98c;
		font-size: 0.88rem;
		line-height: 1.5;
		padding: 0.75rem 0.9rem;
	}
	.stTextArea textarea {
        background: #0d1b1d;
		border: 1px solid var(--line);
		border-radius: 6px;
		color: var(--ink);
		font-size: 1.05rem;
		line-height: 1.5;
	}
	.stButton > button {
		background: var(--coral);
		border: 0;
		border-radius: 5px;
		color: white;
		font-family: 'DM Sans', sans-serif;
		font-weight: 700;
		min-height: 2.8rem;
		width: 100%;
	}
	.stButton > button:hover { background: #c95e4c; color: white; }
    .stButton > button p { color: white !important; }
    .stTextArea label, .stSlider label { color: var(--ink) !important; }
    [data-testid="stMarkdownContainer"] p { color: var(--ink); }
	.stSlider [data-baseweb="slider"] { padding-top: 0.2rem; }
	footer { visibility: hidden; }
	</style>
	""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the LSTM model...")
def get_prediction_assets():
    return load_prediction_assets(MODEL_DIR)


with st.sidebar:
    st.markdown("## Next Word Studio")
    st.markdown("A small writing companion powered by your trained LSTM model.")
    st.divider()
    st.markdown("### Generation controls")
    word_count = st.slider("Words to generate",
                           min_value=1, max_value=30, value=10)
    temperature = st.slider(
        "Temperature",
        min_value=0.2,
        max_value=1.8,
        value=1.0,
        step=0.1,
        help="Lower values stay closer to likely words. Higher values create more variation.",
    )
    st.divider()
    st.markdown(
        '<div class="tip"><strong>Try a strong opening.</strong><br>'
        "The model works best with a few meaningful words to continue.</div>",
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="hero"><div class="eyebrow">LSTM text generation</div>'
    '<h1>Give the thought<br>somewhere to go.</h1>'
    '<div class="hero-copy">Start with a phrase, a feeling, or half a sentence. '
    "Your model will suggest what might come next.</div></div>",
    unsafe_allow_html=True,
)


left_column, right_column = st.columns([1.35, 1], gap="large")
with left_column:
    st.markdown('<div class="section-label">Your opening</div>',
                unsafe_allow_html=True)
    prompt = st.text_area(
        "Prompt",
        value=st.session_state.get(
            "selected_prompt", "The future belongs to those who"),
        height=145,
        label_visibility="collapsed",
        placeholder="Write a few words to begin...",
    )
    st.markdown('<div class="section-label result-label">Continuation</div>',
                unsafe_allow_html=True)
    result = st.session_state.get("generated_text")
    if result:
        st.markdown(
            f'<div class="result-box">{escape(result)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="result-box placeholder">Your generated continuation will appear here.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)
    generate_clicked = st.button("Generate continuation", type="primary")

with right_column:
    st.markdown("### Need a starting point?")
    sample_prompts = [
        "The best way to find yourself is",
        "In the middle of difficulty",
        "Every morning is a chance to",
    ]
    for sample in sample_prompts:
        if st.button(sample, key=f"sample_{sample}"):
            st.session_state["selected_prompt"] = sample
            st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="tip"><strong>Temperature guide</strong><br>'
        "Use a lower temperature for familiar, focused phrasing. Increase it when you want the model to take a more unexpected turn.</div>",
        unsafe_allow_html=True,
    )


if "selected_prompt" in st.session_state:
    st.session_state.pop("selected_prompt")

if generate_clicked:
    if not prompt.strip():
        st.warning("Enter a few words before generating a continuation.")
    else:
        try:
            model, tokenizer, max_len, index_to_word = get_prediction_assets()
            with st.spinner("Finding the next words..."):
                st.session_state["generated_text"] = generate_text(
                    model,
                    tokenizer,
                    index_to_word,
                    prompt,
                    max_len,
                    word_count,
                    temperature,
                )
            st.rerun()
        except (FileNotFoundError, OSError, ValueError, pickle.UnpicklingError) as error:
            st.error(f"The prediction assets could not be loaded: {error}")
