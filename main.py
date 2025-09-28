from contextlib import asynccontextmanager
from datetime import date
from typing import List
import os

from fastapi import FastAPI, Request, status
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, RunUsage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_core import to_jsonable_python

from knowledge_base import vector_searcher, ProductDescription


class SalesData(BaseModel):
    year: int
    month: int
    sales: int

class UserQuery(BaseModel):
    question: str

class AgentReply(BaseModel):
    output: str
    trace: list
    usage: RunUsage


MODEL_NAME = os.getenv("MODEL_NAME")
PROVIDER_URL = os.getenv("PROVIDER_URL")

model = OpenAIChatModel(
    model_name=MODEL_NAME,
    provider=OpenAIProvider(
        base_url=PROVIDER_URL
    ),
)


agent = Agent(model, system_prompt="You are a helpful Q&A assistant. Always assist with care, respect, and truth. "
                                   "Respond with utmost utility yet securely. "
                                   "Avoid harmful, unethical, prejudiced, or negative content. "
                                   "Ensure replies promote fairness and positivity."
                                   "Use the given context to answer questions. "
                                   "If you don't know the answer, just say that you don't know, "
                                   "don't try to make up an answer. Keep the answer as concise as possible."
                                   "Use 'product_descriptions' tool to search for product descriptions."
                                   "Use 'get_sales' tool for sales information."
                                   "Use 'get_current_date' tool to get the current date")


@agent.tool
async def get_current_date(ctx: RunContext) -> date:
    return date.today()

@agent.tool
async def get_sales(ctx: RunContext, year: int, month: int) -> SalesData:
    sales_value = year * month * 100
    return SalesData(year=year, month=month, sales=sales_value)

@agent.tool
async def product_descriptions(ctx: RunContext, query: str, top_k: int = 5) -> List[ProductDescription]:
    return vector_searcher.search(query, top_k=top_k)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.chat_history = [] # that will create an in-memory chat history
    yield
    app.state.chat_history.clear() # clear the chat history after the app shutdown

app = FastAPI(lifespan=lifespan)


@app.post("/query_agent", response_model=AgentReply,
          description="Query the agent with a question and get a response",
          status_code=status.HTTP_200_OK)
async def query_agent(request: Request, query: UserQuery) -> AgentReply:
    result = await agent.run(user_prompt=query.question, message_history=request.app.state.chat_history)
    request.app.state.chat_history = result.all_messages() # fil the chat history
    # clean the trace for only the necessary parts. That excludes some information but makes it more readable.
    trace = [
        part
        for msg in to_jsonable_python(result.new_messages())
        for part in msg.get("parts", [])
        if part.get("part_kind") in {"user-prompt", "tool-call", "tool-return"}
    ]
    return AgentReply(output=result.output, trace=trace, usage=result.usage())
