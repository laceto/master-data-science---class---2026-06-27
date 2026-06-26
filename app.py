"""
LangGraph RAG Chatbot — Streamlit App

Features:
- LangGraph StateGraph with MemorySaver (persistent multi-turn memory)
- Hybrid BM25 + FAISS retrieval via EnsembleRetriever (Reciprocal Rank Fusion)
- Live token streaming with st.write_stream
- Configurable retriever weights and k via sidebar
- Retrieved context viewer per message
"""

import uuid
from typing import Annotated, Generator

import streamlit as st
from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# Knowledge Base
# ══════════════════════════════════════════════════════════════════════════════

_RAW_TEXTS = [
    """LangChain is an open-source framework for building applications powered by large language
    models. It provides composable building blocks — prompts, chains, agents, tools, memory, and
    retrievers — that developers assemble to create complex AI applications. LangChain supports
    integrations with over 60 LLM providers including OpenAI, Anthropic, and Cohere.""",

    """LangChain Expression Language (LCEL) is the recommended way to compose LangChain components.
    It uses the pipe operator (|) to chain Runnables together: prompt | llm | output_parser.
    LCEL supports streaming, async, batch, and parallel execution out of the box, and provides
    an elegant way to build both simple chains and complex agent workflows.""",

    """LangGraph is a library for building stateful, multi-step agentic applications using a graph
    abstraction. Unlike a simple chain, LangGraph represents your application as a directed graph
    where nodes are functions (or LLM calls) and edges define the flow of data. It supports cycles,
    branching, human-in-the-loop, and persistent state — making it ideal for complex agent
    architectures.""",

    """LangGraph StateGraph is the core primitive in LangGraph. You define a TypedDict as the
    graph state, add nodes (Python functions), connect them with edges, and compile the graph.
    The state is passed between nodes and accumulated using reducer functions. The special
    add_messages reducer appends messages to the conversation history without overwriting them.""",

    """LangGraph supports three streaming modes: 'values' streams the full graph state after each
    step, 'updates' streams only the node output deltas, and 'messages' streams individual LLM
    token chunks for real-time token-level output. You can combine modes by passing a list:
    stream_mode=['updates', 'messages'].""",

    """Retrieval-Augmented Generation (RAG) is a technique that grounds LLM responses in external
    knowledge. Instead of relying solely on the model's parametric memory, RAG retrieves relevant
    documents at query time and injects them into the prompt as context. This reduces hallucinations
    and lets the model answer questions about private or up-to-date information.""",

    """Dense retrieval uses neural embedding models to encode documents and queries into vector
    space. Semantic similarity is measured with cosine similarity or dot product. FAISS (Facebook
    AI Similarity Search) is a popular library for efficient approximate nearest-neighbor search
    over millions of embeddings. OpenAI's text-embedding-3-small model produces 1536-dimensional
    vectors at low cost.""",

    """BM25 (Best Match 25) is a classical sparse retrieval algorithm based on TF-IDF with document
    length normalization. It excels at exact keyword matching and handles out-of-vocabulary terms
    that embedding models might miss. BM25 is fast, requires no GPU, and needs no training data —
    making it a strong baseline for any retrieval system.""",

    """Hybrid retrieval combines sparse retrieval (BM25) and dense retrieval (embeddings) to get
    the best of both worlds. BM25 handles exact keyword matches well; dense retrieval captures
    semantic similarity. The EnsembleRetriever in LangChain merges results using Reciprocal Rank
    Fusion (RRF), a rank-aggregation method that is robust to score distribution differences.""",

    """Reciprocal Rank Fusion (RRF) scores each document as the sum of 1/(k + rank_i) across
    retrievers, where k=60 by default. Documents appearing at the top of multiple retriever result
    lists receive the highest combined score. RRF is parameter-light and outperforms linear score
    interpolation in most benchmarks without requiring score normalization.""",

    """FAISS from_documents() takes a list of LangChain Documents and an embedding model, embeds
    all document texts, and stores them in an index. The as_retriever() method wraps it in a
    LangChain retriever interface. You can persist the index to disk with save_local() and reload
    with load_local() to avoid re-embedding on every run.""",

    """GPT-4o is OpenAI's flagship multimodal model. It accepts text and image inputs and returns
    text output. GPT-4o is faster and cheaper than GPT-4 Turbo while matching or exceeding its
    performance on most benchmarks. It supports a 128k token context window and function calling
    with parallel tool use.""",

    """OpenAI's streaming API returns Server-Sent Events (SSE) with delta chunks as the model
    generates tokens. In LangChain, setting streaming=True on ChatOpenAI and calling .stream()
    yields AIMessageChunk objects one token at a time. This enables responsive UIs that show text
    appearing incrementally rather than waiting for the full response.""",

    """A LangGraph agent typically alternates between an LLM call (which may decide to use tools)
    and a tool execution step. The graph loops until the LLM produces a response without requesting
    a tool call. This ReAct-style loop (Reasoning + Acting) is the foundation of most LangGraph
    agent implementations and can be extended with memory, human approval steps, and custom tools.""",

    """LangGraph checkpointers enable persistent conversation memory across sessions. When you
    compile a graph with a checkpointer (e.g., MemorySaver or PostgresSaver), the graph state is
    saved after each node execution. Passing a thread_id in the config lets you resume a
    conversation exactly where it left off, even across process restarts.""",
]

DOCUMENTS = [Document(page_content=t.strip()) for t in _RAW_TEXTS]


# ══════════════════════════════════════════════════════════════════════════════
# Cached resources  (created once per Streamlit session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading embedding model…")
def _load_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


@st.cache_resource(show_spinner="Building FAISS vector index…")
def _load_vectorstore() -> FAISS:
    return FAISS.from_documents(DOCUMENTS, _load_embeddings())


def _build_hybrid_retriever(bm25_weight: float, k: int) -> EnsembleRetriever:
    """Rebuild the hybrid retriever with current sidebar settings (cheap)."""
    bm25 = BM25Retriever.from_documents(DOCUMENTS)
    bm25.k = k
    dense = _load_vectorstore().as_retriever(search_kwargs={"k": k})
    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[bm25_weight, round(1.0 - bm25_weight, 2)],
    )


# ── Mutable holder so the graph node always uses the latest retriever ─────────
class _RetrieverHolder:
    retriever: EnsembleRetriever | None = None

_holder = _RetrieverHolder()


# ── Graph (cached per model name; settings flow through _holder) ───────────────
@st.cache_resource(show_spinner="Initialising LangGraph chatbot…")
def _build_chatbot(model_name: str):
    llm = ChatOpenAI(model=model_name, temperature=0, streaming=True)
    memory = MemorySaver()

    SYSTEM = (
        "You are a knowledgeable AI assistant specializing in LangChain, LangGraph, and RAG. "
        "Use the retrieved context below to answer accurately and concisely. "
        "If the context is insufficient, say so and answer from your own knowledge.\n\n"
        "Retrieved context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        MessagesPlaceholder("messages"),
    ])
    chain = prompt | llm

    class _State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        context: str

    def retrieve_node(state: _State) -> dict:
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        query = last_human.content if last_human else ""
        retriever = _holder.retriever or _build_hybrid_retriever(0.4, 4)
        docs = retriever.invoke(query)
        context = "\n\n".join(
            f"[Document {i + 1}]\n{doc.page_content}" for i, doc in enumerate(docs)
        )
        return {"context": context}

    def generate_node(state: _State) -> dict:
        response = chain.invoke({
            "context": state["context"],
            "messages": state["messages"],
        })
        return {"messages": [response]}

    g = StateGraph(_State)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile(checkpointer=memory)


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="LangGraph RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    model_name = st.selectbox(
        "LLM model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        help="Changing the model starts a fresh conversation.",
    )

    st.divider()
    st.subheader("🔍 Hybrid Retriever")

    bm25_weight = st.slider(
        "BM25 weight",
        min_value=0.0, max_value=1.0, value=0.4, step=0.05,
        help="Weight given to keyword (BM25) results. Dense gets 1 − this value.",
    )
    dense_weight = round(1.0 - bm25_weight, 2)
    st.caption(f"Dense (FAISS) weight: **{dense_weight}**")

    k_docs = st.slider(
        "Documents to retrieve (k)",
        min_value=2, max_value=8, value=4,
        help="Number of documents returned by each sub-retriever.",
    )

    st.divider()
    show_context = st.toggle("Show retrieved context", value=True)

    st.divider()
    turns = len([m for m in st.session_state.get("display_messages", []) if m["role"] == "user"])
    st.metric("Conversation turns", turns)

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.pop("thread_id", None)
        st.session_state.pop("display_messages", None)
        st.rerun()

    st.divider()
    st.caption(
        "Built with "
        "[LangGraph](https://langchain-ai.github.io/langgraph/) · "
        "[LangChain](https://python.langchain.com/) · "
        "[OpenAI](https://openai.com/)"
    )

# ── Session state bootstrap ───────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "display_messages" not in st.session_state:
    # display_messages stores dicts: {role, content, context}
    st.session_state.display_messages = []

# Update retriever holder with current sidebar settings on every rerun
_holder.retriever = _build_hybrid_retriever(bm25_weight, k_docs)

# Build (or retrieve cached) chatbot for the selected model
chatbot = _build_chatbot(model_name)
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🤖 LangGraph RAG Chatbot")
st.caption(
    f"Model: **{model_name}** · "
    f"Retriever: BM25 **{bm25_weight:.0%}** + FAISS **{dense_weight:.0%}** · "
    f"k = {k_docs} · "
    f"Thread: `{st.session_state.thread_id[:8]}…`"
)

# ── Render existing conversation ──────────────────────────────────────────────
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_context and msg.get("context"):
            with st.expander("📄 Retrieved context", expanded=False):
                st.text(msg["context"])

# ── Chat input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask something about LangChain, LangGraph, or RAG…"):

    # Show user message immediately
    st.session_state.display_messages.append({"role": "user", "content": user_input, "context": ""})
    with st.chat_message("user"):
        st.markdown(user_input)

    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "context": "",
    }

    # Stream assistant response token by token
    with st.chat_message("assistant"):

        def _token_generator() -> Generator[str, None, None]:
            for chunk, metadata in chatbot.stream(
                initial_state,
                config=config,
                stream_mode="messages",
            ):
                if (
                    isinstance(chunk, AIMessageChunk)
                    and metadata.get("langgraph_node") == "generate"
                    and chunk.content
                ):
                    yield chunk.content

        # st.write_stream renders tokens live and returns the full string
        full_response: str = st.write_stream(_token_generator())

        # Fetch the context the retrieve node stored in the checkpoint
        snapshot = chatbot.get_state(config)
        retrieved_context: str = snapshot.values.get("context", "")

        if show_context and retrieved_context:
            with st.expander("📄 Retrieved context", expanded=False):
                st.text(retrieved_context)

    # Persist to display history for next rerun
    st.session_state.display_messages.append({
        "role": "assistant",
        "content": full_response,
        "context": retrieved_context,
    })
