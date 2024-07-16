# -*- coding: utf-8 -*-
import streamlit as st
import os
import random
from groq import Groq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

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
        "Talk to a trusted friend or family member about how you're feeling.",
        "Consider seeking help from a mental health professional.",
        "Engage in activities that you enjoy and that make you feel good.",
        "Practice relaxation techniques such as deep breathing or meditation.",
        "Make sure to get enough sleep, eat well, and exercise regularly.",
        "Keep a journal to express your thoughts and feelings.",
        "Join a support group to connect with others who are going through similar experiences.",
        "Go out for a walk",
        "Do not isolate yourself",
        "Try finding activities that relax you and make you happy.",
        "Do not engage in activities that seem to stress you or worsen your mood",
        "Set small goals and try to fulfill them. Reward yourself when you accomplish them",
        "Practice self gratitude, be more forgiving to yourself. It is okay to feel down, it is okay to fail.",
        "Practice positivity, always try to to find something good about a situation, believe that it will get better.", "Practice mindfulness or meditation for a few minutes each day.",
    "Engage in regular physical exercise.",
    "Spend time in nature to boost your mood.",
    "Establish a consistent sleep routine.",
    "Eat a balanced diet rich in fruits and vegetables.",
    "Limit caffeine and alcohol intake.",
    "Stay hydrated by drinking plenty of water.",
    "Set small, achievable goals for yourself.",
    "Break tasks into manageable steps.",
    "Practice deep breathing exercises to reduce anxiety.",
    "Try yoga or stretching exercises.",
    "Avoid negative self-talk; challenge those thoughts.",
    "Listen to your favorite music to lift your spirits.",
    "Volunteer or help others in your community.",
    "Join a club or group with shared interests.",
    "Take regular breaks during work or study.",
    "Create a relaxing bedtime routine.",
    "Limit screen time, especially before bed.",
    "Engage in a creative hobby, like painting or writing.",
    "Watch a comedy or funny movie to laugh.",
    "Spend time with pets; they can provide comfort.",
    "Try a new recipe to explore cooking.",
    "Use positive affirmations daily.",
    "Seek professional help when needed.",
    "Attend support groups for shared experiences.",
    "Practice gratitude by listing things you appreciate.",
    "Learn something new, like a language or skill.",
    "Stay organized to reduce stress.",
    "Practice saying no to avoid overcommitting.",
    "Create a vision board of your goals.",
    "Set aside time for self-care each week.",
    "Go for a walk or run to clear your mind.",
    "Explore different relaxation techniques, like aromatherapy.",
    "Limit exposure to negative news or social media.",
    "Create a budget to manage financial stress.",
    "Use guided meditation apps to help focus.",
    "Write letters to yourself for reflection.",
    "Focus on your breathing during stressful moments.",
    "Take a class or workshop to meet new people.",
    "Find a mentor or coach for guidance.",
    "Plan a day trip to explore somewhere new.",
    "Write down your accomplishments to recognize progress.",
    "Spend time doing activities you loved as a child.",
    "Create a relaxing space at home for unwinding.",
    "Use a planner to keep track of tasks and appointments.",
    "Set boundaries with people who drain your energy.",
    "Embrace a new hobby that excites you.",
    "Watch nature documentaries to relax.",
    "Keep a list of books to read for inspiration.",
    "Spend time in the sun for a vitamin D boost.",
    "Develop a morning routine to start your day positively.",
    "Join a fitness class for motivation.",
    "Participate in community events to connect with others.",
    "Practice visualization techniques for goals.",
    "Share your feelings through art or music.",
    "Try gardening to connect with nature.",
    "Write down a daily intention or goal.",
    "Experiment with different relaxation techniques, like progressive muscle relaxation.",
    "Organize a game night with friends or family.",
    "Limit multitasking to improve focus.",
    "Create a self-care kit with your favorite items.",
    "Explore mindfulness-based stress reduction programs.",
    "Attend workshops on personal growth.",
    "Use coloring books for adults to relax.",
    "Take short vacations to recharge.",
    "Make time for regular family or friend check-ins.",
    "Find podcasts that inspire or motivate you.",
    "Use a stress ball or fidget toy to manage anxiety.",
    "Participate in online courses to learn at your own pace.",
    "Create a playlist of uplifting songs.",
    "Start a blog to share your journey and connect with others.",
    "Practice forgiveness to let go of grudges.",
    "Write down your dreams and aspirations.",
    "Explore different cultures through cuisine or travel.",
    "Attend live performances, like concerts or theater.",
    "Limit negativity by surrounding yourself with positive people.",
    "Use essential oils for relaxation.",
    "Incorporate laughter yoga into your routine.",
    "Plan a tech-free day to disconnect.",
    "Take time to reflect on your values and priorities.",
    "Experiment with different relaxation music.",
    "Spend time in a cozy blanket or favorite chair.",
    "Use a gratitude jar to collect positive moments.",
    "Read self-help books for personal development.",
    "Attend local art shows or exhibitions.",
    "Create a mantra to repeat during tough times.",
    "Explore local parks or nature reserves.",
    "Start a gratitude challenge with friends or family.",
    "Write down three things you’re grateful for each day.",
    "Celebrate small victories, no matter how minor.",
    "Create a list of things that bring you joy.",
    "Practice empathy by volunteering for causes you care about.",
    "Explore different forms of meditation.",
    "Engage in random acts of kindness.",
    "Focus on progress, not perfection.",
    "Create a daily ritual that brings you peace.",
    "Share your experiences with a trusted friend."
        
    ]
    support_message = "Please remember that I'm an AI chatbot and not a substitute for professional help."
    recommendation = "Please contact the national helpline if things get too bad."
    
    # Randomly choose 5 tips
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
            combined_response = f"{supportive_response} \n\n\n {model_response}"
        else:
            combined_response = conversation(user_question)['response']

        message = {'human': user_question, 'AI': combined_response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", combined_response)

if __name__ == "__main__":
    main()
