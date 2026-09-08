# Next Word Prediction with LSTM

An end-to-end next-word prediction project built with TensorFlow/Keras and Streamlit. The model is trained on a quote dataset and generates a continuation from a user-provided prompt.

## Features

- LSTM language model trained on quotes
- Saved tokenizer, sequence length, and trained model artifacts
- Streamlit interface with prompt suggestions
- Adjustable generation length
- Temperature sampling for more focused or more varied text
- Dark, responsive user interface

## Project Structure

```text
.
├── app/
│   └── app.py                  # Streamlit application entry point
├── data/
│   └── quote_dataset.csv       # Training dataset
├── models/
│   ├── lstm_model.h5           # Trained Keras model
│   ├── tokenizer.pkl           # Fitted text tokenizer
│   └── max_len.pkl             # Maximum input sequence length
├── notebooks/
│   └── next_word_pred.ipynb    # Data preparation and model training
├── src/
│   ├── model.py                # Model and artifact loading
│   └── prediction.py           # Next-word prediction and generation
├── tests/                      # Reserved for automated tests
├── .gitignore
├── requirements.txt
└── README.md
```

## Technical Overview

The application uses the following inference pipeline:

1. The prompt is converted to lowercase.
2. The saved tokenizer converts words to integer IDs.
3. The sequence is padded or truncated to the saved `max_len` value.
4. The LSTM model predicts a probability distribution for the next token.
5. Temperature sampling selects the next word.
6. The new word is appended to the prompt and the process repeats.

TensorFlow is imported lazily, so the interface can render before the ML runtime and model are initialized. The loaded model is cached for subsequent predictions.

The Streamlit layer in `app/app.py` handles layout, user input, and session state. The reusable model and prediction logic lives in `src/`, keeping it available for automated tests or a future API without importing the UI.

## Requirements

- Python 3.10 or newer
- TensorFlow
- Streamlit
- NumPy
- pandas

The trained model artifacts are already included in `models/`. No separate download is required.

## Installation

Clone the repository and move into the project directory:

```bash
git clone <repository-url>
cd <repository-directory>
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### Windows

If `python` is mapped to the Microsoft Store alias, use the Python launcher instead:

```powershell
py -m pip install -r requirements.txt
```

You can also use the exact Python executable selected by VS Code:

```powershell
C:\Path\To\Python\python.exe -m pip install -r requirements.txt
```

## Start the Application

Run this command from the repository root:

```bash
python -m streamlit run app/app.py
```

On Windows, the equivalent launcher command is:

```powershell
py -m streamlit run app/app.py
```

Streamlit will display a local URL. Open the default address below if it is not opened automatically:

```text
http://localhost:8501
```

To use a different port:

```bash
python -m streamlit run app/app.py --server.port 8502
```

## Using the App

1. Enter a phrase in the prompt box or select a suggested prompt.
2. Choose how many words to generate.
3. Adjust temperature:
   - Lower values produce more predictable continuations.
   - Higher values produce more varied continuations.
4. Select **Generate continuation**.

The first generation may take longer because TensorFlow and the saved model are loaded on demand. Later generations reuse the cached model.

## Training Notebook

The notebook in `notebooks/next_word_pred.ipynb` contains the dataset preparation, tokenizer creation, sequence padding, LSTM training, evaluation, and artifact export steps.

If you retrain the model, keep these output files in `models/` so the Streamlit app can load them:

- `lstm_model.h5`
- `tokenizer.pkl`
- `max_len.pkl`

## Troubleshooting

### `ModuleNotFoundError: No module named streamlit`

Install the dependencies using the same Python interpreter that starts Streamlit:

```bash
python -m pip install -r requirements.txt
```

### The page is slow on the first request

TensorFlow initialization and model loading happen lazily on the first generation. This is expected. The model is cached afterward.

### Port 8501 is already in use

Start Streamlit on another port:

```bash
python -m streamlit run app/app.py --server.port 8502
```

## License

No license has been added yet. Add a `LICENSE` file before distributing this project publicly.
