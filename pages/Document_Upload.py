import streamlit as st
from langchain.llms import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
import pdfplumber
import redis
import json


load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(api_key=openai_key, temperature=0.8)
output_parser = StrOutputParser()

REDIS_URL="redis://redis-db:6379"

redis_client = redis.Redis(host='redis-db', port=6379, db=0)
print("Connected to Redis")

list_from_redis = redis_client.lrange("message_store:document_upload", 0, -1)

history = [json.loads(item.decode('utf-8')) for item in list_from_redis]

if len(history) == 0:
    redis_client.rpush("message_store:document_upload", json.dumps({"type": "ai", "data": {"content": "Howdy, what would you like to know about this doc?"}}))


def extract_data(feed):
    data = []
    with pdfplumber.open(feed) as pdf:
        pages = pdf.pages 
        for p in pages:
            data.append(p.extract_text())
    return data


#contextualization
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

#Answer question
TEMPLATE = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", TEMPLATE),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)



st.title("RAG Chatbot with :orange[Document Upload]")

st.markdown("### Please upload a PDF document")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

retriever = None
if uploaded_file is not None:
    docs = extract_data(uploaded_file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    
    documents = text_splitter.create_documents(docs)
    vectorestore = Chroma.from_documents(documents, OpenAIEmbeddings())
    
    retriever = vectorestore.as_retriever()
    


if retriever is not None:
    history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt)
    
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt, output_parser=output_parser)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: RedisChatMessageHistory(
    session_id, url="redis://redis-db:6379"),
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",)
    
    # print the history
    for msg in history[::-1]:
        st.chat_message(msg['type']).write(msg['data']['content'])
    
    if prompt:= st.chat_input():
        st.chat_message("human").write(prompt)
        
        config = {"configurable": {"session_id": "document_upload"}}
        response = conversational_rag_chain.invoke({"input": prompt}, config)
        st.chat_message("ai").write(response["answer"])
else:
    st.info("Please upload a PDF document to continue")





    
# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)




