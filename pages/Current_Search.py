# RAG with TAVILY 
import streamlit as st 
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
import getpass
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.runnables import RunnablePassthrough
load_dotenv()


openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
retriever = TavilySearchAPIRetriever(api_key=tavily_key,k=3)

history = StreamlitChatMessageHistory()

if len(history.messages) == 0:
    history.add_ai_message("Hello, I am upto date. Ask me...")

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)


chain = (
    RunnablePassthrough.assign(context=(lambda x: x["question"]) | retriever)
    | prompt
    | ChatOpenAI(api_key=openai_key, temperature=0.7)
    | StrOutputParser()
)



chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,  # Always return the instance created earlier
    input_messages_key="question",
    history_messages_key="history",
)
st.title('RAG with :green[TAVILY]')


for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    # st.write(retriever.invoke({"question": "Tell me about the recent flood in dubai"}))
    st.chat_message("human").write(prompt)

    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": prompt}, config)
    st.chat_message("ai").write(response)
