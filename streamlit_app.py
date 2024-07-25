import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool, DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory
import json
import random

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

# Language learning tools
def generate_exercise(query):
    vocab = {
        "Beginner": [("hello", "hola"), ("goodbye", "adiós"), ("please", "por favor")],
        "Intermediate": [("to develop", "desarrollar"), ("to improve", "mejorar"), ("skill", "habilidad")],
        "Advanced": [("to implement", "implementar"), ("to optimize", "optimizar"), ("algorithm", "algoritmo")]
    }
    
    questions = []
    for _ in range(5):
        word, translation = random.choice(vocab[st.session_state.level])
        questions.append({"question": f"Translate '{word}' to {st.session_state.language}", "correct_answer": translation})
    
    return json.dumps({"questions": questions})

def evaluate_answers(exercise, user_answers):
    exercise_dict = json.loads(exercise)
    feedback = []
    for i, question in enumerate(exercise_dict['questions']):
        user_answer = user_answers.get(str(i), "").strip().lower()
        correct_answer = question['correct_answer'].lower()
        if user_answer == correct_answer:
            feedback.append(f"Question {i+1}: Correct!")
        else:
            feedback.append(f"Question {i+1}: Incorrect. The correct answer is '{correct_answer}'. You answered '{user_answer}'.")
    return "\n".join(feedback)

def translate_text(query):
    return f"Translate the following text to {st.session_state.language}: {query}"

# Initialize DuckDuckGo search with error handling
def safe_ddg_search(query):
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e:
        return f"An error occurred during the search: {str(e)}. I'll try to answer based on my existing knowledge."

# Define tools
tools = [
    Tool(
        name="Internet Search",
        func=safe_ddg_search,
        description="Useful for finding up-to-date information on language topics."
    ),
    Tool(
        name="Generate Exercise",
        func=generate_exercise,
        description="Generates an exercise on a specific topic."
    ),
    Tool(
        name="Translation",
        func=translate_text,
        description="Translates text to the target language."
    )
]

# Initialize the agent with GPT-4o-mini
llm = ChatOpenAI(
    temperature=0.7, 
    streaming=True, 
    openai_api_key=openai_api_key,
    model="gpt-4o-mini"
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
    verbose=True,
    memory=memory
)

# Sidebar for language and level selection
st.sidebar.header("Choose Your Learning Path")
selected_language = st.sidebar.selectbox("Select Language", ["Spanish", "French", "German", "Italian", "Chinese"])
selected_level = st.sidebar.selectbox("Select Your Level", ["Beginner", "Intermediate", "Advanced"])

if st.sidebar.button("Start Learning"):
    st.session_state.language = selected_language
    st.session_state.level = selected_level
    st.session_state.messages = []
    st.session_state.current_exercise = None
    st.session_state.user_answers = {}
    st.experimental_rerun()

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