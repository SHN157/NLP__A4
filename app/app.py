import streamlit as st
import torch
import pickle
import os
import torch.nn as nn
from utils import BERT, SimpleTokenizer, calculate_similarity, predict_nli

# Cache the model and tokenizer to avoid reloading

@st.cache_resource
def load_model():
    """Loads the trained BERT model and tokenizer."""
    # Get the correct paths for model files
    param_path = os.path.join(os.path.dirname(__file__), "models", "param.pkl")
    model_path = os.path.join(os.path.dirname(__file__), "models", "s_model.pt")

    # Ensure files exist
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"Error: '{param_path}' not found. Make sure 'param.pkl' is inside 'app/models/'.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: '{model_path}' not found. Make sure 's_model.pt' is inside 'app/models/'.")

    # Load parameters
    data = pickle.load(open(param_path, "rb"))
    word2id = data["word2id"]
    tokenizer = SimpleTokenizer(word2id)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize BERT model
    model = BERT()
    model.to(device)

    # Load model state dict
    state_dict = torch.load(model_path, map_location=device)

    # Ensure the classifier input matches the checkpoint shape
    expected_input_size = 768  # Adjust this based on your original trained model

    if model.classifier.in_features != expected_input_size:
        print(f"⚠️ Reinitializing classifier: Expected in_features={expected_input_size}, Found={model.classifier.in_features}")
        model.classifier = nn.Linear(expected_input_size, 2).to(device)  # Match original model

    # Load model weights (ignoring mismatches)
    model.load_state_dict(state_dict, strict=False)

    return model, tokenizer, device


# Load model once and cache it
model, tokenizer, device = load_model()

# Streamlit UI
st.title("Natural Language Inference (NLI) & Similarity Predictor")
st.write("Enter a **Premise** and a **Hypothesis** to analyze their relationship.")

# User input fields
premise = st.text_area("Premise", "A man is playing a guitar on stage.")
hypothesis = st.text_area("Hypothesis", "The man is performing music.")

# Prediction button
if st.button("Predict"):
    with st.spinner("Analyzing..."):
        # Compute Sentence Similarity
        similarity_score = calculate_similarity(model, tokenizer, premise, hypothesis, device)

        # Compute NLI Classification
        nli_label, confidence = predict_nli(model, tokenizer, premise, hypothesis, device)

        # Display Results
        st.subheader(f"NLI Relationship: **{nli_label}**")
        st.write(f" Confidence Score: {confidence:.2f}")
        st.subheader(f"Sentence Similarity Score: {float(similarity_score):.3f}")  

