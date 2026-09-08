from pathlib import Path
import pickle
from html import escape

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"


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
		--ink: #172323;
		--muted: #61706d;
		--paper: #f5f3ec;
		--panel: #fffdf8;
		--line: #d9ded7;
		--teal: #1c726d;
		--coral: #dd735f;
		--yellow: #f1c75b;
	}

	.stApp {
		background: var(--paper);
		color: var(--ink);
		font-family: 'DM Sans', sans-serif;
	}

	[data-testid="stHeader"] { background: transparent; }
	[data-testid="stSidebar"] {
		background: #e5eee8;
		border-right: 1px solid var(--line);
	}
	[data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }

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
		box-shadow: 0 14px 35px rgba(23, 35, 35, 0.06);
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
		background: #173c3a;
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
		background: #fff2ca;
		border-left: 4px solid var(--yellow);
		border-radius: 4px;
		color: #66521b;
		font-size: 0.88rem;
		line-height: 1.5;
		padding: 0.75rem 0.9rem;
	}
	.stTextArea textarea {
		background: #fffefb;
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
	.stSlider [data-baseweb="slider"] { padding-top: 0.2rem; }
	footer { visibility: hidden; }
	</style>
	""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the LSTM model...")
def load_prediction_assets():
    model = load_model(MODEL_DIR / "lstm_model.h5")
    with (MODEL_DIR / "tokenizer.pkl").open("rb") as file:
        tokenizer = pickle.load(file)
    with (MODEL_DIR / "max_len.pkl").open("rb") as file:
        max_len = pickle.load(file)
    index_to_word = {index: word for word,
                     index in tokenizer.word_index.items()}
    return model, tokenizer, max_len, index_to_word


def predict_next_word(model, tokenizer, index_to_word, text, max_len, temperature):
    token_sequence = tokenizer.texts_to_sequences([text.lower()])
    padded_sequence = pad_sequences(
        token_sequence,
        maxlen=max_len,
        padding="post",
        truncating="pre",
    )
    predictions = model.predict(padded_sequence, verbose=0)[0]
    predictions[0] = 0
    logits = np.log(predictions + 1e-8) / temperature
    probabilities = tf.nn.softmax(logits).numpy()
    predicted_index = np.random.choice(len(probabilities), p=probabilities)
    return index_to_word.get(int(predicted_index), "<unk>")


def generate_text(model, tokenizer, index_to_word, prompt, max_len, word_count, temperature):
    generated_text = prompt.strip()
    for _ in range(word_count):
        next_word = predict_next_word(
            model, tokenizer, index_to_word, generated_text, max_len, temperature
        )
        if next_word == "<unk>":
            break
        generated_text = f"{generated_text} {next_word}".strip()
    return generated_text


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
    st.markdown('<div class="workspace">', unsafe_allow_html=True)
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
    st.markdown("</div>", unsafe_allow_html=True)

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
            model, tokenizer, max_len, index_to_word = load_prediction_assets()
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
