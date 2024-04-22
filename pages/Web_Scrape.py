#RAG with web links
import streamlit as st 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_core.runnables import RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=openai_key, temperature=0.7)
SYSTEM_TEMPLATE = """
Answer the user's questions based on the below context. 
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

<context>
{context}
</context>
"""
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_TEMPLATE,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)
question_answering_prompt_without_scrape = ChatPromptTemplate.from_messages(
    [
        (
            "system", "You are a helpful assistant. You are to provide answers to questions asked by the user."
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

output_parser = StrOutputParser()

history = StreamlitChatMessageHistory(key="chat_messages")

if len(history.messages) == 0:
    history.add_ai_message("How can I help you?")



st.title(':orange[Retriveal-Augmentation Generation] with Web Links')

for msg in history.messages:
        st.chat_message(msg.type).write(msg.content)
options = st.selectbox("Scrape ? ",["Yes","No"])
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    config = {"configurable": {"session_id": "any"}}
    
    if(options == "Yes"):
        
        st.markdown("### Enter a link to scrape from")
        web_link = st.text_input("link")
        
        if st.button("Scrape"):
        
            st.write(f"Scraping from {web_link}")
            loader = WebBaseLoader(web_link)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            all_splits = text_splitter.split_documents(data)
            vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())    
        
            retriever = vectorstore.as_retriever(k=4) # k is the number of chunks to retrieve
            #you can invoke retriever directly using similarity searches
            
            query_transforming_retriever_chain = RunnableBranch((
                        lambda x: len(x.get("history", [])) == 1,
                        # If only one message, then we just pass that message's content to retriever
                        (lambda x: x["history"][-1].content) | retriever,
                    ),
                
                question_answering_prompt | llm | StrOutputParser() | retriever,
                ).with_config(run_name="chat_retriever_chain")
                
            chain_with_history = RunnableWithMessageHistory(
                    query_transforming_retriever_chain,
                    lambda session_id: history,
                    input_messages_key="question",
                    history_messages_key="history",
                )
        
            
            
            response = chain_with_history.invoke({"question": prompt}, config)
            st.chat_message("ai").write(response.content)
    else:
        chain = question_answering_prompt_without_scrape | llm | output_parser
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: history,
            input_messages_key="question",
            history_messages_key="history",
        )
        response = chain_with_history.invoke({"question": prompt}, config)
        st.chat_message("ai").write(response)
        
    

