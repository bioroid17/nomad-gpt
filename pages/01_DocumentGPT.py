from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema import messages_from_dict, messages_to_dict
import streamlit as st
import json, os

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/documentgpt/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/documentgpt/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def get_history(input):
    return memory.load_memory_variables({})["history"]


def restore_memory(pathname):
    global memory
    with open(pathname, "r") as f:
        memory_json = json.load(f)
        messages = messages_from_dict(memory_json)
        for i in range(0, len(messages), 2):
            if type(messages[i]) == HumanMessage:
                save_message(messages[i].content, "human")
            if type(messages[i + 1]) == AIMessage:
                save_message(messages[i + 1].content, "ai")
            memory.save_context(
                {"input": messages[i].content},
                {"output": messages[i + 1].content},
            )


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

if "memory" not in st.session_state:
    memory = ConversationBufferMemory(return_messages=True)
    st.session_state["memory"] = memory
else:
    memory = st.session_state["memory"]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file.", type=["pdf", "txt", "docx"]
    )

if file:
    retriver = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)

    pathname = f"./.cache/documentgpt/memories/{file.name}/memory.json"
    if os.path.exists(pathname) and not st.session_state["messages"]:
        restore_memory(pathname)

    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "history": RunnableLambda(get_history),
                "context": retriver | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            result = chain.invoke(message)
        memory.save_context({"input": message}, {"output": result.content})
        dirname = f"./.cache/documentgpt/memories/{file.name}/"
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        with open(pathname, "w") as f:
            messages = messages_to_dict(get_history({}))
            json.dump(messages, f)
else:
    st.session_state["messages"] = []
