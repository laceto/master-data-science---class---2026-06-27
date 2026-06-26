"""
LangChain Demo App: Pitti Fashion Exhibitor Search & Analysis
Showcases: embeddings, RAG, semantic search, summarization, classification
"""

import streamlit as st
import pandas as pd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# Set page config
st.set_page_config(
    page_title="Pitti LangChain Demo",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Pitti Exhibitor Search with LangChain")
st.markdown("""
This demo showcases LangChain capabilities:
- **Vector Embeddings** - Convert text to embeddings
- **Semantic Search** - Find similar exhibitors by meaning
- **RAG (Retrieval-Augmented Generation)** - Answer questions using context
- **Text Summarization** - Summarize company descriptions
""")

# Load data with caching
@st.cache_resource
def load_data():
    df = pd.read_excel("pitti.xlsx")
    return df

@st.cache_resource
def create_vector_store(df):
    """Create FAISS vector store from exhibitor descriptions"""
    # Prepare documents
    documents = []
    for idx, row in df.iterrows():
        if pd.notna(row['description']):
            doc_content = f"Company: {row.get('tag', 'Unknown')}\nDescription: {row['description']}"
            documents.append(doc_content)

    # Split text for better embedding
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = splitter.create_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# Load data
df = load_data()
st.sidebar.success(f"✅ Loaded {len(df)} Pitti exhibitors")

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ OPENAI_API_KEY not set. Set it in your environment to use AI features.")
    st.info("""
    To run this demo, set your OpenAI API key:
    ```bash
    export OPENAI_API_KEY='sk-...'
    streamlit run pitti_langchain_demo.py
    ```
    """)
else:
    # Create vector store
    vector_store = create_vector_store(df)

    # Tabs for different demos
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Semantic Search",
        "💬 Q&A with RAG",
        "📊 Stats & Tags",
        "ℹ️ About"
    ])

    with tab1:
        st.header("Semantic Search")
        st.markdown("Find exhibitors by meaning, not keywords")

        query = st.text_input(
            "What kind of fashion company are you looking for?",
            placeholder="e.g., luxury leather goods, sustainable fashion, vintage style"
        )

        if query:
            # Semantic search
            results = vector_store.similarity_search(query, k=5)
            st.subheader(f"Top 5 Matches for: '{query}'")

            for i, doc in enumerate(results, 1):
                with st.expander(f"📦 Result {i}"):
                    st.write(doc.page_content)

    with tab2:
        st.header("Question & Answer with RAG")
        st.markdown("Ask questions and get answers based on exhibitor data")

        # Create QA chain
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        qa_prompt = PromptTemplate(
            template="""You are a helpful assistant for the Pitti Immagine fashion fair.
Answer the question based on the exhibitor information provided.
If you don't know the answer, say "I don't have that information in the exhibitor database."

Context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": qa_prompt}
        )

        question = st.text_input(
            "Ask a question about exhibitors:",
            placeholder="e.g., Which companies make sustainable footwear?"
        )

        if question:
            with st.spinner("🤔 Thinking..."):
                answer = qa_chain.invoke({"query": question})
                st.subheader("Answer:")
                st.write(answer["result"])

    with tab3:
        st.header("Dataset Overview")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Exhibitors", len(df))
        with col2:
            st.metric("With Descriptions", df['description'].notna().sum())
        with col3:
            st.metric("Unique Tags", df['tag'].nunique())

        # Top tags
        st.subheader("Top Exhibitor Categories")
        tag_counts = df['tag'].value_counts().head(15)
        st.bar_chart(tag_counts)

        # Sample exhibitors
        st.subheader("Sample Exhibitors")
        sample_df = df[['tag', 'description']].dropna().head(10)
        st.dataframe(sample_df, use_container_width=True)

    with tab4:
        st.header("About This Demo")
        st.markdown("""
        ### LangChain Features Demonstrated:

        1. **Text Embeddings** (Tab 1)
           - Converts text descriptions into numerical vectors
           - Uses OpenAI's embedding model for semantic understanding

        2. **Vector Search** (Tab 1)
           - FAISS for fast similarity search
           - Finds conceptually similar exhibitors

        3. **Retrieval-Augmented Generation** (Tab 2)
           - Retrieves relevant context from vector store
           - Feeds to LLM for intelligent answers
           - Grounds responses in actual exhibitor data

        4. **Prompt Engineering** (Tab 2)
           - Custom prompt templates for RAG
           - Structured input/output formats

        ### Data Source:
        - **Dataset**: Pitti Immagine Exhibition exhibitors
        - **Records**: 797 fashion companies
        - **Fields**: Description, Tags, Links

        ### Stack:
        - LangChain for orchestration
        - OpenAI API for embeddings and LLM
        - FAISS for vector similarity
        - Streamlit for UI
        """)

# Footer
st.divider()
st.caption("LangChain Demo | Pitti Exhibitor Analysis")
