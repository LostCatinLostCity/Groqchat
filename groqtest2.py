# -*- coding: utf-8 -*-
import streamlit as st
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch

# Load the RAG model and tokenizer
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

def detect_emotional_query(query):
    emotional_keywords = [
        "depressed", "depression", "lonely", "loneliness", "sad", "sadness", 
        "anxious", "anxiety", "overwhelmed", "unhappy", "hopeless", "worthless", 
        "empty", "guilt", "guilty", "helpless", "tearful", "crying", "cry", 
        "fatigue", "tired", "exhausted", "insomnia", "sleep problems", "sleep issues",
        "appetite loss", "weight loss", "weight gain", "suicidal", "self-harm", 
        "self harm", "cutting", "cut", "hurt myself", "no purpose", "no meaning",
        "stressed", "stress", "panic", "panic attack", "fear", "dread", "worry", 
        "worried", "trouble concentrating", "restless", "agitated", "irritable", 
        "isolated", "isolation", "withdrawn", "disconnected"
    ]
    for keyword in emotional_keywords:
        if keyword in query.lower():
            return True
    return False

def get_supportive_response():
    tips = [
        "1. Talk to a trusted friend or family member about how you're feeling.",
        "2. Consider seeking help from a mental health professional.",
        "3. Engage in activities that you enjoy and that make you feel good.",
        "4. Practice relaxation techniques such as deep breathing or meditation.",
        "5. Make sure to get enough sleep, eat well, and exercise regularly.",
        "6. Keep a journal to express your thoughts and feelings.",
        "7. Join a support group to connect with others who are going through similar experiences.",
        "8. Set small, achievable goals to help build a sense of accomplishment.",
        "9. Try to maintain a routine to provide structure and normalcy.",
        "10. Avoid alcohol and drugs, as they can worsen your mood and emotional state."
    ]
    support_message = "I'm really sorry to hear that you're feeling this way. Please remember that I'm an AI chatbot and not a substitute for professional help."
    recommendation = "It's really important to talk things over with someone who can help, such as a mental health professional or a trusted person in your life."
    tips_message = "Here are some things you can try to help you feel better:\n" + "\n".join(tips)
    return f"{support_message}\n{recommendation}\n\n{tips_message}"

def generate_response(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate a response using RAG
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            num_return_sequences=1,
            num_beams=2
        )

    # Decode the generated response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def main():
    st.title("ThinkHealth")

    user_question = st.text_area("Ask a question:")

    # session state variable
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if user_question:
        # Check if the query is emotional
        if detect_emotional_query(user_question):
            supportive_response = get_supportive_response()
            rag_response = generate_response(user_question)
            combined_response = f"{supportive_response}\n\nRAG Response:\n{rag_response}"
        else:
            # Generate response using RAG for non-emotional queries
            rag_response = generate_response(user_question)
            combined_response = rag_response
        
        message = {'human': user_question, 'AI': combined_response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", combined_response)

if __name__ == "__main__":
    main()
