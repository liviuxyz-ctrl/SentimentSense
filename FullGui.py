import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Hugging Face
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Streamlit for GUI
import streamlit as st


###########################################
#             CONFIG & PATHS
###########################################
DATA_PATH = "IMDB Dataset.csv"             # CSV with columns ['review', 'sentiment']
MODEL_OUTPUT_DIR = "./bert-imdb-output"    # Where to store outputs
BEST_MODEL_DIR = os.path.join(MODEL_OUTPUT_DIR, "best_model")


###########################################
#       DATASET CLASS FOR IMDB
###########################################
class IMDBTransformersDataset(Dataset):
    """
    Simple dataset class for IMDB text + label pairs.
    """
    def __init__(self, df, tokenizer, max_length=512):
        self.texts = df['review'].tolist()
        self.labels = df['sentiment'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


###########################################
#          METRIC FUNCTION
###########################################
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


###########################################
#    FUNCTION TO LOAD OR TRAIN MODEL
###########################################
@st.cache_resource  # so we don't repeatedly retrain or reload in Streamlit
def load_or_train_model():
    """
    If a trained model is found at BEST_MODEL_DIR, load it.
    Otherwise, train from scratch on the IMDB CSV data, save, and load.
    Returns tokenizer, model
    """
    if os.path.exists(BEST_MODEL_DIR):
        st.write(f"[INFO] Found existing trained model in {BEST_MODEL_DIR}.")
        st.write("[INFO] Skipping training, loading the saved model...")
        tokenizer = BertTokenizerFast.from_pretrained(BEST_MODEL_DIR)
        model = BertForSequenceClassification.from_pretrained(BEST_MODEL_DIR)
        return tokenizer, model
    else:
        st.write("[INFO] No pre-trained model found locally. We will train from scratch... this is gonna take a while "
                 "go and make some tea")
        # 1) Load data
        df = pd.read_csv(DATA_PATH)
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        st.write(f"Training size: {len(train_df)} | Test size: {len(test_df)}")

        # 2) Tokenizer
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        # 3) Create datasets
        train_dataset = IMDBTransformersDataset(train_df, tokenizer)
        test_dataset = IMDBTransformersDataset(test_df, tokenizer)

        # 4) Load base BERT model
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        # 5) Training arguments
        training_args = TrainingArguments(
            output_dir=MODEL_OUTPUT_DIR,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            fp16=True,  # Mixed precision
        )

        # 6) Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        # 7) Train
        trainer.train()

        # 8) Evaluate
        results = trainer.evaluate(test_dataset)
        st.write("Evaluation Results:", results)

        # 9) Save best model
        best_ckpt_path = os.path.join(MODEL_OUTPUT_DIR, "best_model")
        st.write(f"[INFO] Saving best model to {best_ckpt_path}")
        trainer.save_model(best_ckpt_path)  # Saves tokenizer & model config too

        # reload to ensure everything is fresh
        tokenizer = BertTokenizerFast.from_pretrained(best_ckpt_path)
        model = BertForSequenceClassification.from_pretrained(best_ckpt_path)

        return tokenizer, model


###########################################
#           INFERENCE FUNCTION
###########################################
def predict_sentiment(text, tokenizer, model, max_length=512):
    """
    Predicts sentiment (0=Negative, 1=Positive) and confidence.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    pred_label = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_label].item()
    return pred_label, confidence


###########################################
#         STREAMLIT APP
###########################################
def main():
    st.title("IMDB Sentiment Analysis with BERT")
    st.write("This app will train (if needed) and load a BERT model for sentiment classification on IMDB reviews.")

    # 1) Load or train the model
    tokenizer, model = load_or_train_model()

    # 2) Provide some example reviews
    examples = [
        "I absolutely loved this movie! The acting was wonderful, and the story was so compelling.",
        "This was the worst film I have ever seen. Complete waste of time.",
        "An average movie. It had some good moments but also a lot of flaws.",
        "Fantastic cinematography and great performances. I'd definitely watch it again!",
        "Poor script and mediocre acting. I couldn't wait for it to end."
    ]

    st.subheader("Try an Example Review")
    selected_example = st.selectbox("Select an example review:", examples)
    if st.button("Analyze Selected Example"):
        label, conf = predict_sentiment(selected_example, tokenizer, model)
        label_str = "Positive" if label == 1 else "Negative"
        st.write(f"**Prediction:** {label_str}")
        st.write(f"**Confidence:** {conf:.4f}")

    st.write("---")

    # 3) Let the user type their own review
    st.subheader("Or Write Your Own Review")
    user_review = st.text_area("Enter your movie review here:", height=150)
    if st.button("Analyze My Review"):
        if len(user_review.strip()) == 0:
            st.warning("Please enter some text first.")
        else:
            label, conf = predict_sentiment(user_review, tokenizer, model)
            label_str = "Positive" if label == 1 else "Negative"
            st.write(f"**Prediction:** {label_str}")
            st.write(f"**Confidence:** {conf:.4f}")


if __name__ == "__main__":
    main()
