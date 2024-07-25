import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool, DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory
from interpreter import interpreter
import json

# Use the API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

st.title("🌍 Interactive Language Learning with AI Teacher")

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = None
if 'level' not in st.session_state:
    st.session_state.level = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_exercise' not in st.session_state:
    st.session_state.current_exercise = None
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}

# Initialize Open Interpreter
interpreter.api_key = openai_api_key
interpreter.auto_run = True

# Language learning tools
def generate_exercise(query):
    code = f"""
import random
import json

def generate_{st.session_state.language.lower()}_exercise(level):
    vocab = {{
        "Beginner": [("hello", "hola"), ("goodbye", "adiós"), ("please", "por favor")],
        "Intermediate": [("to develop", "desarrollar"), ("to improve", "mejorar"), ("skill", "habilidad")],
        "Advanced": [("to implement", "implementar"), ("to optimize", "optimizar"), ("algorithm", "algoritmo")]
    }}
    
    questions = []
    for _ in range(5):
        word, translation = random.choice(vocab[level])
        questions.append({{"question": f"Translate '{word}' to {st.session_state.language}", "correct_answer": translation}})
    
    return json.dumps({{"questions": questions}})

print(generate_{st.session_state.language.lower()}_exercise("{st.session_state.level}"))
"""
    result = interpreter.run(code)
    return result.strip()

def evaluate_answers(exercise, user_answers):
    code = f"""
import json

exercise = json.loads('''{exercise}''')
user_answers = {user_answers}

feedback = []
for i, question in enumerate(exercise['questions']):
    user_answer = user_answers.get(str(i), "").strip().lower()
    correct_answer = question['correct_answer'].lower()
    if user_answer == correct_answer:
        feedback.append(f"Question {{i+1}}: Correct!")
    else:
        feedback.append(f"Question {{i+1}}: Incorrect. The correct answer is '{{correct_answer}}'. You answered '{{user_answer}}'.")

print("\\n".join(feedback))
"""
    result = interpreter.run(code)
    return result.strip()

# ... (rest of the code remains the same)

# Main chat interface
if st.session_state.language and st.session_state.level:
    st.write(f"You are learning {st.session_state.language} at {st.session_state.level} level.")
    st.write("Chat with your AI language teacher below:")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Exercise interface
    if st.session_state.current_exercise:
        exercise = json.loads(st.session_state.current_exercise)
        st.write("### Current Exercise")
        for i, question in enumerate(exercise['questions']):
            st.text(f"Question {i+1}: {question['question']}")
            st.session_state.user_answers[str(i)] = st.text_input(f"Answer {i+1}", key=f"answer_{i}")
        
        if st.button("Submit Answers"):
            feedback = evaluate_answers(st.session_state.current_exercise, st.session_state.user_answers)
            st.session_state.messages.append({"role": "assistant", "content": feedback})
            st.session_state.current_exercise = None
            st.session_state.user_answers = {}
            st.experimental_rerun()

    # Chat input
    if prompt := st.chat_input("What would you like to learn about?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in agent.run(prompt):
                full_response += response
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        if "Generate Exercise" in prompt:
            st.session_state.current_exercise = generate_exercise(prompt)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.write("Please select your language and level in the sidebar to start learning!")