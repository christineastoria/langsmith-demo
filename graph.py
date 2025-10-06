# graph.py
import operator, os
from typing import List
from typing_extensions import TypedDict, Annotated

from langchain_core.messages import HumanMessage, AnyMessage, get_buffer_string
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver  # optional, for persistence

# LangChain docs + retriever
from langchain.docstore.document import Document
from langchain_community.retrievers import WikipediaRetriever  # needs `pip install wikipedia`

print("Tracing V2:", os.getenv("LANGCHAIN_TRACING_V2"))
print("Project:", os.getenv("LANGCHAIN_PROJECT"))
print("Key set:", bool(os.getenv("LANGCHAIN_API_KEY")))

# ---- config ----
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2
WIKI_MAX_DOCS = 2

retriever = WikipediaRetriever(top_k_results=WIKI_MAX_DOCS)  # emits 'retriever' runs
llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE)  # emits 'llm' runs

MUSEUM_PROMPT = """You are a museum label writer.

Write a concise (150–200 words), public-facing label for the given artwork using only the provided context.
Be concise, scholarly, and elegant; avoid jargon or define it briefly. If context is insufficient, say so briefly.

Artwork: {question}

Context (from Wikipedia):
{context}

Write the label. If any details are uncertain, qualify them briefly.
"""

# -------- Graph state (matches your clean example shape) --------
class GraphState(TypedDict):
    question: str
    messages: Annotated[List[AnyMessage], operator.add]
    documents: List[Document]

# -------- Nodes --------
def retrieve_documents(state: GraphState):
    messages = state.get("messages", [])
    question = state["question"]
    query = f"{get_buffer_string(messages)} {question}".strip()
    documents = retriever.invoke(query)
    return {"documents": documents}

def generate_response(state: GraphState):
    question = state["question"]
    messages = state.get("messages", [])
    documents = state.get("documents", [])

    context = "\n\n".join(doc.page_content for doc in documents) if documents else \
              f"No reliable context found for '{question}'."
    prompt_text = MUSEUM_PROMPT.format(context=context, question=question)
    generation = llm.invoke([HumanMessage(content=prompt_text)])
    return {"documents": documents, "messages": [HumanMessage(question), generation]}

# -------- Build & compile the graph --------
builder = StateGraph(GraphState)
builder.add_node("retrieve_documents", retrieve_documents)
builder.add_node("generate_response", generate_response)
builder.add_edge(START, "retrieve_documents")
builder.add_edge("retrieve_documents", "generate_response")
builder.add_edge("generate_response", END)

# Optional: persist threads/checkpoints across restarts
checkpointer = SqliteSaver.from_conn_string("studio.db")
graph = builder.compile(checkpointer=checkpointer)  # <-- Studio will look for `graph`
