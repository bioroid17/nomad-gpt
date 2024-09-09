import json
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-0125",
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[
        function,
    ],
)

prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 (TEN) questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
"""
)


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quizgpt/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context": format_docs} | prompt | llm
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


with st.sidebar:
    docs = None
    topic = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Name of the article")
        if topic:
            docs = wiki_search(topic)
    show_answer = st.toggle("Show answers when submitted", False)

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name)
    data_str = response.additional_kwargs["function_call"]["arguments"]
    data = json.loads(data_str)

    with st.form("questions_form"):
        for question in data["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option",
                [answer["answer"].replace("(o)", "") for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
            elif value is not None:
                answer_str = ""
                if show_answer:
                    for index, answer in enumerate(question["answers"]):
                        if "correct" in answer and answer["correct"]:
                            answer_str = f" Answer: {index + 1}"
                            break
                st.error("Wrong!" + answer_str)
        button = st.form_submit_button()
