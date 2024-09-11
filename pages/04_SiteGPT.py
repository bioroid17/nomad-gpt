from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema import messages_from_dict, messages_to_dict, Document
from fake_useragent import UserAgent
import streamlit as st
import asyncio, os, json

asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.title("SiteGPT")

st.markdown(
    """
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)

# Initialize a UserAgent object
ua = UserAgent()


class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message.replace("$", "\$"))


llm = ChatOpenAI(
    temperature=0.1,
)
streaming_llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.
    If the answer answers the user question the score should be high, else it should be low.
    Make sure to always include the answer's score even if it's 0.
    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!
    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    history = inputs["history"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": doc.page_content,
                        "history": "",
                    }
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
        "history": "",
    }


if "memory" not in st.session_state:
    memory = ConversationBufferMemory(return_messages=True)
    st.session_state["memory"] = memory
else:
    memory = st.session_state["memory"]

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            Cite sources and return the sources of the answers as they are, do not change them.
            Answers: {answers}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | streaming_llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nData:{answer['date']}\n"
        for answer in answers
    )

    return choose_chain.invoke(
        {"question": question, "answers": condensed, "history": get_history({})}
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    nav = soup.find("nav")
    aside = soup.find("aside")
    astro_breadcrumbs = soup.find("astro-breadcrumbs")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    if nav:
        nav.decompose()
    if aside:
        aside.decompose()
    if astro_breadcrumbs:
        astro_breadcrumbs.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace(
            "Cloudflare DashboardDiscordCommunityLearning CenterSupport Portal  Cookie Settings",
            "",
        )
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        loader = SitemapLoader(
            url,
            filter_urls=[
                r"^(.*\/ai-gateway\/).*",
                r"^(.*\/vectorize\/).*",
                r"^(.*\/workers-ai\/).*",
            ],
            parsing_function=parse_page,
        )
        loader.requests_per_second = 5
        # Set a realistic user agent
        loader.headers = {"User-Agent": ua.random}

        docs = loader.load_and_split(text_splitter=splitter)
        cache_dir = LocalFileStore(f"./.cache/sitegpt/{url.split('/')[2]}/embeddings/")
        embeddings = OpenAIEmbeddings()
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_dir
        )
        vector_store = FAISS.from_documents(docs, cached_embeddings)
        return vector_store
    except Exception as e:
        return []


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


all_questions, all_answers = [], []


def restore_memory(pathname):
    global memory
    with open(pathname, "r") as f:
        memory_json = json.load(f)
        messages = messages_from_dict(memory_json)
        for i in range(0, len(messages), 2):
            if type(messages[i]) == HumanMessage:
                save_message(messages[i].content, "human")
                all_questions.append(messages[i].content)
            if type(messages[i + 1]) == AIMessage:
                save_message(messages[i + 1].content, "ai")
            memory.save_context(
                {"input": messages[i].content},
                {"output": messages[i + 1].content},
            )


def find_history(query):
    histories = get_history({})
    temp = []
    for idx in range(len(histories) // 2):
        temp.append(
            {
                "input": histories[idx * 2].content,
                "output": histories[idx * 2 + 1].content,
            }
        )

    docs = [
        Document(page_content=f"input:{item['input']}\noutput:{item['output']}")
        for item in temp
    ]
    if len(docs) < 10:
        return None
    try:
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
        found_docs = vector_store.similarity_search_with_relevance_scores(
            query,
            k=1,
            score_threshold=0.98,
        )
        candidate = found_docs[0].page_content.split("\n")[1]
        return candidate.replace("output:", "")
    except IndexError:
        return None


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )

if url:

    pathname = f"./.cache/sitegpt/{url.split('/')[2]}/"
    if not os.path.exists(pathname):
        os.mkdir(pathname)

    if (
        os.path.exists(f"{pathname}memories/memory.json")
        and not st.session_state["messages"]
    ):
        restore_memory(f"{pathname}memories/memory.json")
    paint_history()
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL")
    else:
        query = st.chat_input("Ask a question to the website.")
        vector_store = load_website(url)
        if query:
            send_message(query, "human")
            found = find_history(query)
            if found:
                send_message(found, "ai")
                memory.save_context({"input": query}, {"output": found})
            else:
                retriever = vector_store.as_retriever()
                chain = (
                    {
                        "docs": retriever,
                        "question": RunnablePassthrough(),
                    }
                    | RunnablePassthrough.assign(history=get_history)
                    | RunnableLambda(get_answers)
                    | RunnableLambda(choose_answer)
                )
                with st.chat_message("ai"):
                    result = chain.invoke(query)
                memory.save_context({"input": query}, {"output": result.content})
            if not os.path.exists(f"{pathname}memories/"):
                os.mkdir(f"{pathname}memories/")
            with open(f"{pathname}memories/memory.json", "w") as f:
                messages = messages_to_dict(get_history({}))
                json.dump(messages, f)
else:
    st.session_state["messages"] = []
