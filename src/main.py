# 1. Create database
# 2. Load data
# 3. Start server.

# Data abstraction:
# Langchain documentsa
# Document type: cocktail-info, user-info.

# Server:
# Get form
# Post form


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
from langgraph.graph.state import CompiledStateGraph

from langchain_chroma import Chroma


import logic


MESSAGES = None
TEMPLATES = None


async def init():
    global MESSAGES, TEMPLATES

    TEMPLATES = Jinja2Templates(directory="html_templates")
    MESSAGES = [AIMessage("Hello! How can I help you today?")]

    await logic.init()


def get_messages():
    return MESSAGES


def get_templates():
    assert TEMPLATES
    return TEMPLATES


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def get_form(
    request: Request,
    templates: Annotated[Jinja2Templates, Depends(get_templates)],
    messages: Annotated[List[BaseMessage], Depends(get_messages)],
):
    processed_messages = map(
        lambda m: (m.type, markdown2.markdown(str(m.content))),
        filter(lambda m: m.type in ["ai", "human"], messages),
    )

    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"messages": list(processed_messages)},
    )


@app.post("/send_message")
async def post_form(
    message: Annotated[str, Form()],
    messages: Annotated[List[BaseMessage], Depends(get_messages)],
    graph: Annotated[CompiledStateGraph, Depends(logic.get_graph)],
):
    if not message:
        return

    messages.append(HumanMessage(message))
    config = {"configurable": {"user_id": "1", "thread_id": "1"}}
    result = await graph.ainvoke({"user_message": message}, config=config)
    messages.append(AIMessage(result["final_answer"]))

    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
