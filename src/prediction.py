import numpy as np


def predict_next_word(model, tokenizer, index_to_word, text, max_len, temperature):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

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
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
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
