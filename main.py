import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain_community.chat_message_histories import RedisChatMessageHistory
import redis 
import json
load_dotenv()



openai_key = os.getenv("OPENAI_API_KEY")
REDIS_URL="redis://localhost:6379/0"

redis_client = redis.Redis(host='localhost', port=6379, db=0)

st.set_page_config(page_title='Chatbot', page_icon='💻')

st.title('Welcome :blue[GDSC]')



st.header("Chat bot without History")

st.info("Before we proceed, let us raise the heat 🌡 ")

temperature:int = st.slider("Temperature", 0.0, 1.0, 0.8, 0.1)

llm = ChatOpenAI(api_key=openai_key, temperature=temperature)


 
list_from_redis = redis_client.lrange("message_store:any", 0, -1)

history = [json.loads(item.decode('utf-8')) for item in list_from_redis]

output_parser = StrOutputParser()

if len(history) == 0:
    redis_client.rpush("message_store:any", json.dumps({"type": "ai", "data": {"content": "Yo fam"}}))




prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a Nigerian street guy and you speak only pidgin. You are having a conversation with a small boy. You are to be funny, sacarsitic and rude with your responses."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


chain = prompt | llm | output_parser

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: RedisChatMessageHistory(
        session_id, url="redis://localhost:6379"
    ), # Always return the instance created earlier
    input_messages_key="question",
    history_messages_key="history",
)

# print the history
for msg in history:
    st.chat_message(msg['type']).write(msg['data']['content'])

if prompt:= st.chat_input():
    st.chat_message("human").write(prompt)
    
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": prompt}, config)
    st.chat_message("ai").info(response)
   

#NOTE: To track different conversations, you can use the session_id to differentiate between them.