import sys
from pathlib import Path
import pytest

# Ensure repository root is in sys.path for src imports
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model import load_prediction_assets
from src.prediction import generate_text, predict_next_word


@pytest.fixture(scope="module")
def prediction_assets():
    return load_prediction_assets()


def test_generate_text(prediction_assets):
    model, tokenizer, max_len, index_to_word = prediction_assets
    prompt = "the only way to"
    word_count = 5
    temperature = 1.0

    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        index_to_word=index_to_word,
        prompt=prompt,
        max_len=max_len,
        word_count=word_count,
        temperature=temperature,
    )

    assert isinstance(output, str)
    assert len(output.strip()) > 0
    prompt_word_count = len(prompt.split())
    output_word_count = len(output.split())
    assert output_word_count > prompt_word_count


def test_predict_next_word(prediction_assets):
    model, tokenizer, max_len, index_to_word = prediction_assets
    prompt = "the only way to"
    temperature = 1.0

    next_word = predict_next_word(
        model=model,
        tokenizer=tokenizer,
        index_to_word=index_to_word,
        text=prompt,
        max_len=max_len,
        temperature=temperature,
    )

    assert isinstance(next_word, str)
    assert len(next_word.strip()) > 0
