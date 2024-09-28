import json
import streamlit as st
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from typing_extensions import override
from openai import AssistantEventHandler
import openai as client
import yfinance
import re

st.set_page_config(
    page_title="AssistantAPI",
    page_icon="🧰",
)

st.title("AssistantAPI")


class EventHandler(AssistantEventHandler):

    message = ""

    @override
    def on_text_created(self, text) -> None:
        self.message_box = st.empty()

    def on_text_delta(self, delta, snapshot):
        self.message += delta.value
        self.message_box.markdown(self.message.replace("$", "\$"))

    @override
    def on_message_done(self, message) -> None:
        # print a citation to the file searched
        message_content = message.content[0].text
        annotations = message_content.annotations
        citations = []
        for index, annotation in enumerate(annotations):
            message_content.value = message_content.value.replace(
                annotation.text, f"[{index}]"
            )
            if file_citation := getattr(annotation, "file_citation", None):
                cited_file = client.files.retrieve(file_citation.file_id)
                citations.append(
                    f"[{index}] {cited_file.filename}({annotation.start_index}-{annotation.end_index})"
                )

        matches = len(re.findall(r"【[^】]*】", self.message))
        for n in range(matches):
            self.message = re.sub(
                r"【[^】]*】",
                f"[{n}]",
                self.message,
                1,
            )
        self.message += f"\n\nSources: {citations}"
        self.message_box.markdown(self.message)

    def on_event(self, event):

        if event.event == "thread.run.requires_action":
            submit_tool_outputs(event.data.id, event.data.thread_id)


# Tools
def get_ticker(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    company_name = inputs["company_name"]
    return ddg.run(f"Ticker symbol of {company_name}")


def get_income_statement(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.income_stmt.to_json())


def get_balance_sheet(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.balance_sheet.to_json())


def get_daily_stock_performance(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.history(period="3mo").to_json())


functions_map = {
    "get_ticker": get_ticker,
    "get_income_statement": get_income_statement,
    "get_balance_sheet": get_balance_sheet,
    "get_daily_stock_performance": get_daily_stock_performance,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "get_ticker",
            "description": "Given the name of a company returns its ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The name of the company",
                    }
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_income_statement",
            "description": "Given a ticker symbol (i.e AAPL) returns the company's income statement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_balance_sheet",
            "description": "Given a ticker symbol (i.e AAPL) returns the company's balance sheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_stock_performance",
            "description": "Given a ticker symbol (i.e AAPL) returns the performance of the stock for the last 100 days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
]


#### Utilities
def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    return messages


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    with client.beta.threads.runs.submit_tool_outputs_stream(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()


def insert_message(message, role):
    with st.chat_message(role):
        st.markdown(message)


def paint_history(thread_id):
    messages = get_messages(thread_id)
    for message in messages:

        if (
            message.assistant_id
            and client.beta.assistants.retrieve(message.assistant_id).name
            == "Book Assistant"
        ):
            message_content = message.content[0].text
            annotations = message_content.annotations
            citations = []
            for index, annotation in enumerate(annotations):
                message_content.value = message_content.value.replace(
                    annotation.text, f"[{index}]"
                )
                if file_citation := getattr(annotation, "file_citation", None):
                    cited_file = client.files.retrieve(file_citation.file_id)
                    citations.append(
                        f"[{index}] {cited_file.filename}({annotation.start_index}-{annotation.end_index})"
                    )

            matches = len(re.findall(r"【[^】]*】", message.content[0].text.value))
            for n in range(matches):
                message.content[0].text.value = re.sub(
                    r"【[^】]*】",
                    f"[{n}]",
                    message.content[0].text.value,
                    1,
                )
            message.content[0].text.value += f"\n\nSources: {citations}"

        insert_message(
            message.content[0].text.value,
            message.role,
        )


@st.cache_data(show_spinner="Uploading file...")
def upload_file(uploaded, assistant_id, thread_id):
    if "files" not in st.session_state:
        st.session_state["files"] = [uploaded.name]
    else:
        if uploaded.name in st.session_state["files"]:
            return
        else:
            st.session_state["files"].append(uploaded.name)
    print(st.session_state["files"])
    file = client.files.create(
        file=uploaded,
        purpose="assistants",
    )
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="Please refer to the uploaded file for my question.",
        attachments=[
            {
                "file_id": file.id,
                "tools": [
                    {
                        "type": "file_search",
                    }
                ],
            }
        ],
    )
    insert_message("Please refer to the uploaded file for my question.", "user")
    with st.chat_message("assistant"):
        with client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant_id,
            event_handler=EventHandler(),
        ) as stream:
            stream.until_done()


if "thread" not in st.session_state:
    thread = client.beta.threads.create()
    st.session_state["thread"] = thread
else:
    thread = st.session_state["thread"]

with st.sidebar:
    ASSISTANT_NAME = st.selectbox(
        "Choose what you want to use.",
        (
            "Investor Assistant",
            "Book Assistant",
        ),
    )
    if ASSISTANT_NAME == "Investor Assistant":
        instruction = "You help users do research on publicly traded companies and you help users decide if they should buy the stock or not."
        md_quote = "Ask a question about a company and our Assistant will do the research for you."
    elif ASSISTANT_NAME == "Book Assistant":
        instruction = "You help users with their question on the files they upload."
        md_quote = (
            "Upload your files on the sidebar and our Assistant will answer your query."
        )
        uploaded = st.file_uploader(
            "Upload a .txt .pdf or .docx file.", type=["pdf", "txt", "docx"]
        )

st.markdown(
    f"""
    Welcome to AssistantAPI.
            
    {md_quote}
"""
)

assistants = client.beta.assistants.list(limit=10)
for a in assistants:
    if a.name == ASSISTANT_NAME:
        assistant = client.beta.assistants.retrieve(a.id)
        break
else:
    assistant = client.beta.assistants.create(
        name=ASSISTANT_NAME,
        instructions=instruction,
        model="gpt-4o-mini",
        tools=functions,
    )


paint_history(thread.id)
content = st.chat_input("What do you want to search?")
if ASSISTANT_NAME == "Book Assistant":
    if uploaded:
        if uploaded not in st.session_state.get("files", []):
            upload_file(uploaded, assistant.id, thread.id)


if content:
    send_message(thread.id, content)
    insert_message(content, "user")

    with st.chat_message("assistant"):
        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            event_handler=EventHandler(),
        ) as stream:
            stream.until_done()
