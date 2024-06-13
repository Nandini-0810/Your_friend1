import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the pre-trained model and tokenizer
@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Function to generate a response
def generate_response(input_text, history):
    # Tokenize the input
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    # Append the new user input tokens to the chat history
    bot_input_ids = torch.cat([history, new_user_input_ids], dim=-1) if history is not None else new_user_input_ids
    # Generate a response
    history = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(history[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, history

# Streamlit user interface
st.title("Your one and only Buddy!")
st.write("hey you!\nDon't think just talk!")

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state.history = None

# User input
input_text = st.text_input("You:", "")

if st.button("Send"):
    if input_text:
        # Generate response
        response, st.session_state.history = generate_response(input_text, st.session_state.history)
        # Display the conversation
        st.write(f"*You*: {input_text}")
        st.write(f"*Buddy*: {response}")



