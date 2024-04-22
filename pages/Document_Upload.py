# RAG with document upload
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



# ==== utilitites ====
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

output_parser = StrOutputParser()

history = StreamlitChatMessageHistory(key="chat_messages")

if len(history.messages) == 0:
    history.add_ai_message("How can I help you?")


st.title('RAG with :red[Document Upload OR Web Link]')

load_dotenv()

st.divider()


document, web = st.tabs(["Document Upload", "Web Link"])
with web:
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
        ).with_config(run_name="chat_retriever_chain_for_web")
        
        chain_with_history = RunnableWithMessageHistory(
            query_transforming_retriever_chain,
            lambda session_id: history,
            input_messages_key="question",
            history_messages_key="history",
        )
        
        for msg in history.messages:
            st.chat_message(msg.type).write(msg.content)
        if prompt := st.chat_input():
            st.chat_message("human").write(prompt)

            # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
            config = {"configurable": {"session_id": "any"}}
            response = chain_with_history.invoke({"question": prompt}, config)
            st.chat_message("ai").write(response.content)

with document:
    loader = PyPDFLoader("./resources/file.pdf")
    pages = loader.load_and_split()
    for page in pages:
        content = page.page_content
    doc_chain = create_stuff_documents_chain(llm, question_answering_prompt)
    
    


