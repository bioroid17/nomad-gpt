from langchain.document_loaders import SitemapLoader
from fake_useragent import UserAgent
import streamlit as st
import asyncio

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


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    try:
        loader = SitemapLoader(url)
        loader.requests_per_second = 1
        # Set a realistic user agent
        loader.headers = {"User-Agent": ua.random}
        docs = loader.load()
        return docs
    except Exception as e:
        return []


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL")
    else:
        docs = load_website(url)
        st.write(docs)
