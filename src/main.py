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

TEMPLATES = None


async def init():
    global TEMPLATES
    TEMPLATES = Jinja2Templates(directory="html_templates")

    await logic.init()


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
    messages: Annotated[List[BaseMessage], Depends(logic.get_messages)],
):
    processed_messages = [
        (message.type, markdown2.markdown(str(message.content)))
        for message in messages
        if message.content and message.type in ["ai", "human"]
    ]

    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"messages": list(processed_messages)},
    )


@app.post("/send_message")
async def post_form(
    message: Annotated[str, Form()],
):
    await logic.send_message(message)

    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
