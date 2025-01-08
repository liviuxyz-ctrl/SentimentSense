# SentimentSense

**SentimentSense** is a sentiment analysis application built with a fine-tuned **BERT** model. The app predicts whether a movie review is **positive** or **negative** and provides a confidence score for the prediction. It includes a user-friendly GUI built with **Streamlit**, making it easy to analyze custom or example reviews.

---

## Features

- Fine-tuned **BERT** model trained on the **IMDB Dataset** of 50,000 reviews.
- Supports **real-time sentiment analysis** with a simple GUI.
- Confidence scores displayed alongside predictions for better insights.
- Option to enter custom reviews or select from predefined examples.
- Saves the trained model for reuse, avoiding repeated training.

---

## Dataset

The model is trained on the [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) containing 50,000 movie reviews with binary sentiment labels:

- **Columns Used**:
  - `review`: Text of the movie review.
  - `sentiment`: Label for sentiment (**positive** = 1, **negative** = 0).

---

## Setup

### 1. Install Dependencies

Make sure you have Python 3.8+ installed. Use the `requirements.txt` file to install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download the **IMDB Dataset.csv** file and place it in the project directory. Ensure the file contains the required columns (`review`, `sentiment`).

### 3. Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

### 4. Open in Browser

After running, the app will open in your default browser, or you can access it at the local URL provided (e.g., `http://localhost:8501`).

---

## Usage

### GUI Features:

1. **Pre-Trained Model Check**:
   - If a trained model exists, it will load it automatically.
   - If no model is found, the app will train the model and save it for future use.

2. **Analyze Sentiment**:
   - Enter your own movie review in the provided text box.
   - Or select a predefined review from the dropdown menu.

3. **See Results**:
   - The app will display:
     - Sentiment Prediction: **Positive** or **Negative**.
     - Confidence Score: A value between 0 and 1, indicating the model's certainty.

---

## Example

### Input:
**Review:** `"The movie was absolutely fantastic! A must-watch."`

### Output:
```plaintext
Prediction: Positive
Confidence: 0.982
```

---

## Project Structure

- **`app.py`**: Main application file for Streamlit.
- **`IMDB Dataset.csv`**: The dataset used for training and testing.
- **`bert-imdb-output/`**: Directory where trained models are saved.
- **`requirements.txt`**: List of dependencies.

---

## Requirements

### Python Version:
- **Python 3.8+**

### Dependencies:
Install all dependencies using `requirements.txt`:
```plaintext
streamlit==1.15.2
transformers==4.33.3
torch==2.0.1
scikit-learn==1.3.0
pandas==1.5.3
numpy==1.23.5
```

---

## Notes

- The application will only train the model if a pre-trained model is not found.
- Training requires a GPU for faster performance.
- The trained model is saved in the `bert-imdb-output/best_model` directory.

---

## Contributing

Contributions are welcome! Feel free to fork this repository and submit a pull request with any improvements or new features.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
```
