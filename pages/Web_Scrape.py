import streamlit as  st 
from langchain.llms import OpenAI 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import RedisChatMessageHistory 
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder 
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser 
import os 
from dotenv import load_dotenv 
from bs4 import BeautifulSoup
import redis 
import json
import requests

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(api_key=openai_key, temperature=0.8)
output_parser = StrOutputParser()

REDIS_URL="redis://redis-db:6379"

redis_client = redis.Redis(host='redis-db', port=6379, db=0)
print("Connected to Redis")

list_from_redis = redis_client.lrange("message_store:web_scrape",0,-1)

history = [json.loads(item.decode('utf-8')) for item in list_from_redis]

if len(history) == 0:
    redis_client.rpush("message_store:web_scrape", json.dumps({"type": "ai", "data":{"content" : "Ciao! What do you wish to know?"} }))

def extract_data(feed: str):
    page = requests.get(url=feed)
    soup = BeautifulSoup(page.content, "html.parser")
    return soup.get_text()
    

    
    

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
If you don't know the answer, just say that you don't know.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", TEMPLATE),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

#========================= 
if "loaded" not in st.session_state:
    st.session_state.loaded = False



st.title(":red[Web scrape] Chat bot")

st.markdown("### Please enter link to website containing information you want to know about")

link = st.text_input("Enter link here")

retriever = None
if st.button("Submit"):
    st.session_state.loaded = True
    docs = extract_data(link)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.create_documents(docs)
    
    vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())
    
    retriever = vectorstore.as_retriever()

  
if st.session_state.loaded:
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt,output_parser=output_parser)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(rag_chain, lambda session_id : RedisChatMessageHistory( session_id, url=REDIS_URL), input_messages_key="input", history_messages_key="history", output_messages_key="answer",)
    
    st.write("Chat history")
    for msg in history[::-1]:
        st.chat_message(msg['type']).write(msg['data']['content'])
        
    if prompt:= st.chat_input():
        st.chat_message("human").write(prompt)
        
        config = {"configurable": {"session_id": "web_scrape"}}
        
        response = conversational_rag_chain.invoke({"input": prompt}, config)
        
        st.chat_message("ai").write(response["answer"])
else:
    st.info("Please enter link to proceeed...")