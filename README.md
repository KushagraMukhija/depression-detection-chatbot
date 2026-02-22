# 🧠 Mental Health Support Chatbot  
AI-Powered Depression Detection using DistilBERT

## 📌 Overview

This project is a web-based mental health support chatbot built using **Streamlit**, **TensorFlow**, and **Hugging Face Transformers**.

It uses a fine-tuned **DistilBERT** model to classify user input text into:

- 😊 Not Depressed  
- 😔 Depressed  

Based on the prediction and confidence score, the chatbot generates supportive and empathetic responses.

> ⚠️ This application is intended for educational and research purposes only. It is not a substitute for professional medical advice or diagnosis.

---

## 🚀 Features

- Fine-tuned DistilBERT for binary text classification  
- Real-time text analysis  
- Confidence score display  
- Supportive response generation  
- Interactive chat interface  
- Conversation history using session state  
- Crisis support information in sidebar  
- Detailed probability breakdown for transparency  

---

## 🏗️ Project Structure

```
depression-detection-chatbot-main/
│
├── app.py                          # Streamlit application
├── combined_depression_dataset.csv # Dataset used for training
├── saved_model_distilbert/         # Saved fine-tuned model
│   ├── config.json
│   ├── special_tokens_map.json
│   └── ...
├── training_plot_distilbert.png    # Training visualization
├── Results.txt                     # Model results summary
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

---

## 🧠 Model Details

- **Base Model:** DistilBERT  
- **Framework:** TensorFlow  
- **Task:** Binary Text Classification  
- **Max Sequence Length:** 150 tokens  
- **Output:** Softmax probabilities  

### Prediction Pipeline

1. User enters text  
2. Text is tokenized using `DistilBertTokenizer`  
3. Input is passed to `TFDistilBertForSequenceClassification`  
4. Softmax probabilities are computed  
5. Final prediction + confidence score generated  
6. Chatbot responds accordingly  

---

## 💻 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/KushagraMukhija/depression-detection-chatbot
cd depression-detection-chatbot-main
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually):

```
http://localhost:8501
```

---

## 📊 How It Works

When a user submits a message:

- The model predicts whether the text indicates signs of depression.  
- A confidence score is calculated.  
- Based on prediction and confidence:
  - Supportive response is generated  
  - Encouraging suggestions may be provided  
- Users can view detailed probability breakdown inside an expandable section.

---

## 🛟 Crisis Resources

The application includes mental health support information such as:

- National Suicide Prevention Lifeline (988)  
- Crisis Text Line  
- International crisis center references  

This ensures responsible usage in sensitive scenarios.

---

## 📈 Model Training

The model was fine-tuned on a combined depression dataset (`combined_depression_dataset.csv`).

Training outputs include:

- Accuracy metrics  
- Loss curves  
- Saved model directory  
- Training plot visualization  

The fine-tuned model is stored inside:

```
saved_model_distilbert/
```

Make sure the model path in `app.py` matches the correct folder name.

---

## ⚠️ Disclaimer

This chatbot:

- Does not provide medical diagnosis  
- Is not a licensed therapist  
- Should not replace professional mental health care  

If someone is experiencing severe distress, they should immediately contact a licensed mental health professional or crisis helpline.

---

## 🧪 Future Improvements

- Multi-class emotional classification  
- Sentiment tracking across conversation history  
- Fine-tuning with larger mental health datasets  
- Cloud deployment  
- Voice input integration  
- Improved response personalization  

---

## 🛠️ Technologies Used

- Python  
- Streamlit  
- TensorFlow  
- Hugging Face Transformers  
- DistilBERT  
- Pandas  
- NumPy  

---

## 👨‍💻 Author

Developed as an AI-powered mental health support research project focused on responsible NLP applications.