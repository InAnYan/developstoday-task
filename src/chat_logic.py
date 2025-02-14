from typing import List, Optional, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from typing import Annotated, Hashable, List, Tuple
import pandas as pd
import markdown2

from fastapi import Depends, FastAPI, Form, Request
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import starlette.status as status

from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain.chat_models import init_chat_model

from langchain_chroma import Chroma

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    messages: Annotated[list, add_messages]

    user_preference: Optional[str]

    query: Optional[str]
    cocktails: Optional[List[Document]]


LLM = None
EMBEDDING_MODEL = None
VECTOR_DB = None
AI_TEMPLATES = None
GRAPH = None


async def init():
    init_models()
    await init_vector_db()
    init_ai_templates()
    init_graph()


def init_models():
    global LLM, EMBEDDING_MODEL

    LLM = init_chat_model("gpt-4o-mini")
    EMBEDDING_MODEL = OpenAIEmbeddings()


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


def init_ai_templates():
    global AI_TEMPLATES
    AI_TEMPLATES = Jinja2Templates(directory="ai_templates")


def init_graph():
    graph_builder = StateGraph(State)

    graph_builder.add_node("update_preference", update_preference)
    graph_builder.add_node("gen_query", gen_query)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("answer_with_documents", answer_with_documents)

    graph_builder.add_edge(START, "update_preference")
    graph_builder.add_conditional_edges("update_preference", chat_or_retrieve)
    graph_builder.add_edge("gen_query", "retrieve")
    graph_builder.add_edge("retrieve", "answer")
    graph_builder.add_edge("answer", END)

    global GRAPH
    memory = MemorySaver()
    GRAPH = graph_builder.compile(checkpointer=memory)


async def update_preference(state: State):
    update_preferences_message = fill_template("update_preference.message", state=state)

    result = await llm_answer(update_preferences_message)

    if result.strip().startswith("None"):
        return {}
    else:
        return {"user_preference": result}


async def chat_or_retrieve(state: State) -> str:
    pass


async def gen_query(state: State):
    gen_query_message = fill_template("gen_query.message", state=state)
    result = await llm_answer(gen_query_message)
    return {"query": result}


async def retrieve(state: State):
    if state["query"]:
        result = await vector_db_search(state["query"])
        return {"cocktails": result}
    else:
        return {}


async def answer_with_documents(state: State):
    answer_message = fill_template("answer_with_documents.message", state=state)
    result = await llm_answer(answer_message)
    return {"final_answer": result}


async def llm_answer(input: str) -> str:
    assert LLM
    res = str((await LLM.ainvoke(input)).content)
    # print("LLM ANSWER: ")
    # print(res)
    # print()
    return res


def get_embedding_model():
    assert EMBEDDING_MODEL
    return EMBEDDING_MODEL


async def vector_db_search(query: str) -> List[Document]:
    assert VECTOR_DB
    print(f"QUERY: {query}")
    res = await VECTOR_DB.asimilarity_search(query)
    print(f"RESULTS: {res}")
    print()
    return res


def get_graph():
    assert GRAPH
    return GRAPH


def fill_template(name: str, state: State) -> str:
    assert AI_TEMPLATES
    res = AI_TEMPLATES.get_template(name).render(state=state)
    # print("TEMPLATE: ")
    # print(res)
    # print()
    return res
