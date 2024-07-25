import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool, DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory

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

# Language learning tools
def generate_lesson(query):
    return f"Generate a lesson about '{query}' for {st.session_state.language} at {st.session_state.level} level."

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
        name="Generate Lesson",
        func=generate_lesson,
        description="Generates a lesson on a specific topic."
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
    st.experimental_rerun()

# Main chat interface
if st.session_state.language and st.session_state.level:
    st.write(f"You are learning {st.session_state.language} at {st.session_state.level} level.")
    st.write("Chat with your AI language teacher below:")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.write("Please select your language and level in the sidebar to start learning!")