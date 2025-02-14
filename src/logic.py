from typing import Hashable, List, Optional, Tuple, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from typing import Hashable, List, Tuple
import pandas as pd

from fastapi.templating import Jinja2Templates

from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

from langchain_chroma import Chroma


class State(TypedDict):
    messages: List[BaseMessage]
    user_preference: Optional[str]


STATE = None
LLM = None
EMBEDDING_MODEL = None
VECTOR_DB = None
AI_TEMPLATES = None


async def init():
    init_ai_templates()
    init_models()
    await init_vector_db()
    init_state()


def init_ai_templates():
    global AI_TEMPLATES
    AI_TEMPLATES = Jinja2Templates(directory="ai_templates")


def init_models():
    global LLM, EMBEDDING_MODEL

    LLM = init_chat_model("gpt-4o-mini")
    LLM = LLM.bind_tools([update_preference_tool, retrieve_cocktails_tool])

    EMBEDDING_MODEL = OpenAIEmbeddings()


@tool(parse_docstring=True)
async def update_preference_tool(preference: str):
    """Update user preference of the cocktails they like.

    Args:
        preference (str): User preference of the cocktails they like. Use system message preference too.
    """

    state = get_state()
    print(f"NEW PREFERENCE: {preference}")
    state["user_preference"] = preference
    state["messages"][0] = SystemMessage(fill_template("system.message", state))  # type: ignore


@tool(parse_docstring=True)
async def retrieve_cocktails_tool(query: str) -> List[Document]:
    """Retrieve cocktail information from the database.

    Args:
        query (str): Query to the database.
    """

    assert VECTOR_DB
    return await VECTOR_DB.asimilarity_search(query)


async def init_vector_db():
    global VECTOR_DB

    df = pd.read_csv("assets/final_cocktails.csv")
    df = df[df["category"] == "Cocktail"].drop(
        columns=[
            "glassType",
            "drinkThumbnail",
            "ingredientMeasures",
        ]
    )

    VECTOR_DB = Chroma(embedding_function=EMBEDDING_MODEL)

    await VECTOR_DB.aadd_documents(list(map(convert_cocktail, df.iterrows())))


def convert_cocktail(id_row: Tuple[Hashable, pd.Series]) -> Document:
    row = id_row[1]

    metadata = {
        "id": row["id"],
        "name": row["name"],
        "instructions": row["instructions"],
        "ingredients": row["ingredients"],
        "alcoholic": row["alcoholic"] == "Alcoholic",
    }

    text = f"{metadata['name']} (Alcoholic: {metadata['alcoholic']}): {metadata['ingredients']}"

    return Document(page_content=text, metadata=metadata)


def init_state():
    messages = [
        SystemMessage(fill_template("system.message")),
        AIMessage(fill_template("first.message")),
    ]

    global STATE
    STATE = State(messages=messages, user_preference=None)


async def send_message(input: str):
    if not input:
        return

    state = get_state()
    state["messages"].append(HumanMessage(content=input))
    await call_llm()


async def call_llm():
    state = get_state()

    result = await get_llm().ainvoke(state["messages"])
    state["messages"].append(result)

    if result.tool_calls:  # type: ignore
        for tool_call in result.tool_calls:  # type: ignore
            match tool_call["name"]:
                case "update_preference_tool":
                    res = await update_preference_tool.ainvoke(tool_call)
                    state["messages"].append(res)
                case "retrieve_cocktails_tool":
                    res = await retrieve_cocktails_tool.ainvoke(tool_call)
                    state["messages"].append(res)

        await call_llm()


def fill_template(name: str, context: dict = {}) -> str:
    # Do not change default parameter.
    assert AI_TEMPLATES
    return AI_TEMPLATES.get_template(name).render(context)


def get_state() -> State:
    assert STATE
    return STATE


def get_messages() -> List[BaseMessage]:
    return get_state()["messages"]


def get_llm() -> BaseChatModel:
    assert LLM
    return LLM  # type: ignore
