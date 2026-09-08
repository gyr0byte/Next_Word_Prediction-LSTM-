from pathlib import Path
import pickle


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def load_prediction_assets(model_dir=MODEL_DIR):
    from tensorflow.keras.models import load_model

    model = load_model(model_dir / "lstm_model.h5")
    with (model_dir / "tokenizer.pkl").open("rb") as file:
        tokenizer = pickle.load(file)
    with (model_dir / "max_len.pkl").open("rb") as file:
        max_len = pickle.load(file)
    index_to_word = {
        index: word for word, index in tokenizer.word_index.items()
    }
    return model, tokenizer, max_len, index_to_word
