import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool, DuckDuckGoSearchRun
from langchain.utilities import DuckDuckGoSearchAPIWrapper

# Use the API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

st.title("🌍 LangChain - Language Learning Assistant powered by GPT-4o-mini")

# Language learning tools
def generate_vocabulary_exercise(query):
    parts = query.split()
    language = parts[-3]
    level = parts[-1]
    prompt = f"Generate a vocabulary exercise for {language} at {level} level."
    return prompt

def generate_grammar_exercise(query):
    parts = query.split()
    language = parts[-3]
    level = parts[-1]
    prompt = f"Create a grammar exercise for {language} at {level} level."
    return prompt

def translate_text(query):
    parts = query.split(" to ")
    text = parts[0].replace("Translate ", "")
    target_language = parts[1]
    prompt = f"Translate the following text to {target_language}: {text}"
    return prompt

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
        name="Vocabulary Exercise",
        func=generate_vocabulary_exercise,
        description="Generates vocabulary exercises for language learning."
    ),
    Tool(
        name="Grammar Exercise",
        func=generate_grammar_exercise,
        description="Creates grammar exercises for language practice."
    ),
    Tool(
        name="Translation",
        func=translate_text,
        description="Translates text to the target language."
    )
]

# Initialize the agent with GPT-4o-mini
llm = ChatOpenAI(
    temperature=0, 
    streaming=True, 
    openai_api_key=openai_api_key,
    model="gpt-4o-mini"  # Using the GPT-4o-mini model as specified
)

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True
)

# Create a text input for the user's language learning query
user_query = st.text_input("What would you like to learn today? (e.g., 'Create a Spanish vocabulary exercise for beginners')")

if user_query:
    st.write("I'm working on your request using GPT-4o-mini:")
    with st.spinner("Thinking..."):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[st_callback])
    st.write(response)

# Language selection and level
st.sidebar.subheader("Language Learning Settings")
selected_language = st.sidebar.selectbox("Select Language", ["Spanish", "French", "German", "Italian", "Chinese"])
proficiency_level = st.sidebar.selectbox("Select Proficiency Level", ["Beginner", "Intermediate", "Advanced"])

# Quick actions
st.sidebar.subheader("Quick Actions")
if st.sidebar.button("Generate Vocabulary Exercise"):
    with st.spinner("Generating exercise with GPT-4o-mini..."):
        response = agent.run(f"Generate a vocabulary exercise for {selected_language} at {proficiency_level} level.")
    st.write(response)

if st.sidebar.button("Generate Grammar Exercise"):
    with st.spinner("Generating exercise with GPT-4o-mini..."):
        response = agent.run(f"Create a grammar exercise for {selected_language} at {proficiency_level} level.")
    st.write(response)

# Translation feature
text_to_translate = st.text_area("Enter text to translate:")
if st.button("Translate") and text_to_translate:
    with st.spinner("Translating with GPT-4o-mini..."):
        response = agent.run(f"Translate {text_to_translate} to {selected_language}")
    st.write(response)