import streamlit as st
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import os

# Page configuration
st.set_page_config(
    page_title="Mental Health Support Chatbot",
    page_icon="🧠",
    layout="centered"
)

# --- Load Model and Tokenizer ---
@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    model_path = './saved_model_v2'
    
    try:
        # Load tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        
        # Load model
        model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load model at startup
model, tokenizer = load_model_and_tokenizer()

# Configuration
MAX_LENGTH = 150

# --- Prediction Function ---
def predict_depression(text, model, tokenizer):
    """
    Predict if text indicates depression
    Returns: (prediction, confidence)
    """
    # Tokenize input
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='tf'
    )
    
    # Get prediction
    outputs = model(encoding)
    logits = outputs.logits
    
    # Get probabilities
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    
    # Get prediction (0: Not Depressed, 1: Depressed)
    prediction = int(tf.argmax(logits, axis=-1).numpy()[0])
    confidence = float(probs[prediction])
    
    return prediction, confidence, probs

# --- UI Design ---
st.title("🧠 Mental Health Support Chatbot")
st.markdown("### AI-Powered Depression Detection Assistant")

st.markdown("""
This chatbot uses a fine-tuned DistilBERT model to analyze text and provide supportive responses. 
Share your thoughts or feelings, and I'll try to understand how you're doing.

**⚠️ Important:** This is not a substitute for professional medical advice. If you're experiencing 
severe distress, please contact a mental health professional or crisis helpline.
""")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model:** DistilBERT  
    **Task:** Binary Classification  
    **Classes:**
    - 😊 Not Depressed
    - 😔 Depressed
    
    ---
    
    **Crisis Resources:**
    - National Suicide Prevention Lifeline: 988
    - Crisis Text Line: Text HOME to 741741
    - International Association for Suicide Prevention: [iasp.info](https://www.iasp.info/resources/Crisis_Centres/)
    """)
    
    st.header("📊 Model Info")
    st.info(f"Model loaded from: `{os.path.basename('./saved_model_v2')}`")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I'm here to listen. Feel free to share what's on your mind."
    })

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            # Get prediction
            prediction, confidence, probs = predict_depression(prompt, model, tokenizer)
            
            # Generate response based on prediction
            if prediction == 1:  # Depressed
                if confidence > 0.75:
                    response = f"""I sense you might be going through a difficult time. Your words suggest you may be experiencing some challenging emotions.
                    
**Analysis:** The model detects signs of distress with {confidence*100:.1f}% confidence.

I want you to know that what you're feeling is valid, and you don't have to face this alone. Here are some things that might help:

💙 **Talk to someone you trust** - a friend, family member, or counselor
💙 **Consider professional support** - a therapist can provide specialized help
💙 **Take small steps** - even small acts of self-care matter
💙 **You matter** - your feelings are important and deserve attention

Would you like to talk more about what you're experiencing?"""
                else:
                    response = f"""I'm picking up on some concerning signs in what you've shared ({confidence*100:.1f}% confidence).
                    
While I'm here to listen, please remember that reaching out to a mental health professional can provide the support you deserve. You don't have to navigate this alone.

How are you feeling right now? Is there anything specific that's been weighing on you?"""
            else:  # Not Depressed
                if confidence > 0.75:
                    response = f"""It's good to hear from you! Your message suggests you're in a relatively positive space ({confidence*100:.1f}% confidence).
                    
That's wonderful to see. Keep taking care of yourself and maintaining the things that help you feel good.

Is there anything else you'd like to talk about?"""
                else:
                    response = f"""Thank you for sharing. I'm here to listen and support you.
                    
**Analysis:** The model shows {confidence*100:.1f}% confidence (mixed signals detected).

Whether you're having a good day or a challenging one, your feelings are valid. Would you like to share more about what's on your mind?"""
            
            # Display response
            st.markdown(response)
            
            # Show detailed probabilities in expander
            with st.expander("📊 View Detailed Analysis"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Not Depressed",
                        value=f"{probs[0]*100:.2f}%",
                        delta=None
                    )
                with col2:
                    st.metric(
                        label="Depressed",
                        value=f"{probs[1]*100:.2f}%",
                        delta=None
                    )
                
                st.caption("These probabilities represent the model's confidence in each category.")
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Clear chat button
if st.button("🔄 Start New Conversation"):
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I'm here to listen. Feel free to share what's on your mind."
    })
    st.rerun()