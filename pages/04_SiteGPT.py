from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    try:
        loader = SitemapLoader(
            url,
            filter_urls=[
                r"^(.*\/encryption\/).*",
            ],
            parsing_function=parse_page,
        )
        loader.requests_per_second = 5
        # Set a realistic user agent
        loader.headers = {"User-Agent": ua.random}
        docs = loader.load_and_split(text_splitter=splitter)
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
