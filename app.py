import streamlit as st
from langchain_core.callbacks import CallbackManager
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Define the LLM
llm = Ollama(
    model='mistral',
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

# Define the Streamlit app
st.set_page_config(page_title="Streaming Bot", page_icon="🤖")

# Display the main page
st.title("Streaming Bot 🤖")

# Initialize the Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Create a function to perform query on LLM
def perform_query(query, chat_history):
    template = """You are a helpful assistant. Answer the following questions as best you can considering the history 
    of the conversation.
    
    Chat History: {chat_history}
    
    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm | StrOutputParser()
    return chain.stream({
        "chat_history": chat_history,
        "user_question": query
    })


# Conversation
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# Accept user input
user_query = st.chat_input("Type your message...")
# Display on user input in UI
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        bot_response = perform_query(user_query, st.session_state.chat_history)
        streaming_bot_response = st.write_stream(bot_response)

    st.session_state.chat_history.append(AIMessage(content=streaming_bot_response))
