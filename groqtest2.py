# -*- coding: utf-8 -*-
import streamlit as st
import os
from groq import Groq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import random

load_dotenv()

groq_api_key = "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"

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
        "7. Join a support group to connect with others who are going through similar experiences."
    ]
    support_message = "Please remember that I'm an AI chatbot and not a substitute for professional help."
    recommendation = "Please contact the national helpline if things get too bad."
    selected_tips = random.sample(tips, min(5, len(tips)))
    tips_message = "Here are some things you can try to help you feel better:\n" + "\n".join(selected_tips)
    
    return f"{support_message}\n{recommendation}\n\n{tips_message}"

def main():
    st.title("ThinkHealth")

    model = 'mixtral-8x7b-32768'
    conversational_memory_length = 5

    memory = ConversationBufferWindowMemory(k=conversational_memory_length)

    user_question = st.text_area("Ask a question:")

    # session state variable
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context({'input': message['human']}, {'output': message['AI']})

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )

    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory
    )

    if user_question:
        if detect_emotional_query(user_question):
            supportive_response = get_supportive_response()
            model_response = conversation(user_question)['response']
            combined_response = f"{supportive_response} \n\n {model_response}"
        else:
            combined_response = conversation(user_question)['response']

        message = {'human': user_question, 'AI': combined_response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", combined_response)

if __name__ == "__main__":
    main()
