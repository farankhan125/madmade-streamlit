# RAG Model with chat history and conversation in streamlit using cache memory
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage

# -----------------------
# Load API keys
# -----------------------
load_dotenv()

# -----------------------
# Cache models and vectorstore
# -----------------------
@st.cache_resource
def load_chain():
    embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"]
    )
    
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash",
    #     temperature=0,
    #     google_api_key=os.environ["GOOGLE_API_KEY"]
    # )

    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0,
        api_key=os.environ["OPENAI_API_KEY"]
    )

    vector_store = AstraDBVectorStore(
        collection_name="Madmade_Data",       
        embedding=embedding_model,
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],       
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],         
        namespace=None         
    )

    contextualize_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    system_prompt = """You are Madmade assistant chatbot, helping users with questions about Madmade and its details with creativity, clarity, and confidence. 
    Answer using the context provided. Also you can say something relevant from your own side.
    Using the context, give a summarized answer without missing out any detail.
    Give to the point Answer based on the query.
    If the question is outside Madmade data, politely respond: 
    "I am here to assist with Madmade data only."
    Do not provide unrelated answers.
    Context: {context}"""

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    retriever = vector_store.as_retriever(search_kwargs={'k': 2})
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_prompt,
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        document_chain
    )

    return rag_chain

rag_chain = load_chain()

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Madmade AI ChatBot", page_icon="🤖", layout="centered")
st.title("🤖 Madmade AI ChatBot")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display past messages
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Chat input box
if user_input := st.chat_input("Ask me something..."):
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Show assistant placeholder
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Thinking...")

        # Run RAG chain
        response = rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input
        })

        bot_reply = response["answer"]
        message_placeholder.markdown(bot_reply)

    # Update chat history
    st.session_state.chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=bot_reply)
    ])