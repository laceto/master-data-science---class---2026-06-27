# 🏗️ LangChain Demo: Pitti Fashion Exhibitor Analysis

A hands-on demonstration of core LangChain capabilities using real Pitti Immagine fashion exhibitor data.

## 📚 What You'll Learn

This demo showcases essential LangChain patterns:

### 1. **Text Processing**
- Loading and splitting documents with `RecursiveCharacterTextSplitter`
- Preparing text for embeddings with proper chunk sizing

### 2. **Embeddings & Vector Search**
- Converting text to numerical embeddings using OpenAI's API
- Building and querying FAISS vector stores
- Semantic search (finding by meaning, not keywords)

### 3. **Retrieval-Augmented Generation (RAG)**
- Combining retrieval with language models
- Building Q&A systems that reference actual data
- Custom prompt templates with context injection

### 4. **Chain Composition**
- Creating multi-step workflows with `RetrievalQA`
- LLMChain for classification and transformation
- Prompt templates with variables

### 5. **UI Development**
- Building interactive apps with Streamlit
- Multi-tab interfaces for different features

## 📦 Files Included

```
├── pitti.xlsx                          # Data: 797 fashion exhibitors
├── pitti_langchain_demo.py             # Interactive Streamlit app
├── pitti_langchain_guide.ipynb         # Educational Jupyter notebook
├── requirements_langchain_demo.txt     # Python dependencies
└── LANGCHAIN_DEMO_README.md           # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- (Optional) OpenAI API key for full AI features

### Installation

```bash
# Install dependencies
pip install -r requirements_langchain_demo.txt

# Set OpenAI API key (optional, for advanced features)
export OPENAI_API_KEY='sk-...'  # macOS/Linux
set OPENAI_API_KEY=sk-...        # Windows cmd
$env:OPENAI_API_KEY='sk-...'     # Windows PowerShell
```

### Run the Demo

**Option 1: Interactive Streamlit App**
```bash
streamlit run pitti_langchain_demo.py
```

Opens a web UI with:
- 🔍 Semantic Search tab
- 💬 Q&A with RAG tab
- 📊 Dataset Stats tab

**Option 2: Jupyter Notebook (Educational)**
```bash
jupyter notebook pitti_langchain_guide.ipynb
```

Interactive walkthrough with explanations and code examples.

## 📊 Dataset Overview

**Pitti Immagine Exhibition Exhibitors**
- **Records**: 797 fashion companies
- **Columns**:
  - `tag`: Company name/category
  - `description`: Company description (769 entries)
  - `link`: URL to exhibitor page

## 🎯 Feature Walkthrough

### 1. Semantic Search (Tab 1)

Find exhibitors by meaning, not keywords.

```python
vector_store.similarity_search(
    "sustainable and eco-friendly fashion",
    k=5  # Top 5 results
)
```

**Examples you can try:**
- "luxury leather goods"
- "vintage retro style"
- "sustainable fabrics"
- "innovative modern design"

### 2. Q&A with RAG (Tab 2)

Ask natural language questions and get answers grounded in exhibitor data.

```python
qa_chain.invoke({
    "query": "Which companies make sustainable footwear?"
})
```

**How it works:**
1. Your question is converted to an embedding
2. Similar exhibitor descriptions are retrieved (3 most relevant)
3. Retrieved context + question → ChatGPT → Answer
4. Answer is grounded in real data

### 3. Dataset Stats (Tab 3)

Explore the Pitti dataset:
- Total exhibitor count
- Distribution by category
- Sample data

## 🏗️ Architecture & Components

```
┌─────────────────────────────────────────┐
│         Pitti Exhibitor Data            │
│         (797 companies)                 │
└────────────────┬────────────────────────┘
                 │
                 ▼
        ┌────────────────────┐
        │   Text Splitting   │
        │ (chunk_size=500)   │
        └────────┬───────────┘
                 │
                 ▼
        ┌────────────────────┐
        │  Embeddings        │
        │  (OpenAI API)      │
        └────────┬───────────┘
                 │
                 ▼
        ┌────────────────────┐
        │   FAISS Index      │
        │  (Vector Store)    │
        └────────┬───────────┘
                 │
      ┌──────────┴──────────┐
      │                     │
      ▼                     ▼
  Semantic Search      Retriever
      │                     │
      │                ┌────▼────────┐
      │                │ RetrievalQA  │
      │                │ + ChatOpenAI │
      │                └────┬────────┘
      │                     │
      ▼                     ▼
   Results           Grounded Answers
```

## 💡 Key Concepts

### Text Splitting
Breaks long documents into overlapping chunks for better embeddings:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Characters per chunk
    chunk_overlap=50     # Overlap between chunks
)
```

### Vector Embeddings
Converts text to 1536-dimensional vectors:

```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embedding = embeddings.embed_query("luxury fashion")
# → [0.123, -0.456, 0.789, ...]
```

### FAISS Vector Store
Fast similarity search in embedded space:

```python
vector_store = FAISS.from_documents(docs, embeddings)
results = vector_store.similarity_search(query, k=5)
```

### Retrieval-Augmented Generation
Combines retrieval + generation for accurate answers:

```python
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(...),
    retriever=vector_store.as_retriever(),
    chain_type="stuff"  # Inject all context in one prompt
)
```

## 🔄 Prompt Templates

Reusable prompts with variables:

```python
prompt = PromptTemplate(
    template="Classify this: {text}",
    input_variables=["text"]
)
```

## 📊 Modes

### Demo Mode (No API Key)
- ✓ Text loading & splitting
- ✓ Mock embeddings (deterministic)
- ✓ Semantic search structure
- ✗ Real embeddings
- ✗ RAG/Q&A (requires LLM)

### Full Mode (With OpenAI API Key)
- ✓ All features above
- ✓ Real embeddings
- ✓ RAG with ChatGPT
- ✓ Question answering

## 🛠️ Troubleshooting

**"OPENAI_API_KEY not set"**
- Set your API key: `export OPENAI_API_KEY='sk-...'`
- Demo mode still works without it

**FAISS installation issues**
- macOS/Linux: `pip install faiss-cpu` or `pip install faiss-gpu`
- Windows: Use `faiss-cpu` (pre-built wheels)

**Out of memory**
- Reduce `chunk_size` in `RecursiveCharacterTextSplitter`
- Use `k=3` instead of `k=5` in similarity_search
- Process data in batches

## 📚 Learning Path

1. **Start**: Run `pitti_langchain_guide.ipynb`
   - Understand data flow step-by-step
   - See all concepts in action

2. **Explore**: Try `streamlit run pitti_langchain_demo.py`
   - Interactive experimentation
   - See UI best practices

3. **Extend**:
   - Modify prompts in `pitti_langchain_demo.py`
   - Try different chunk sizes
   - Experiment with retrieval parameters (k, search_type)

4. **Advanced**:
   - Switch to `text-embedding-3-large` for better quality
   - Add memory for multi-turn conversations
   - Implement multi-step agents
   - Use different LLMs (Claude, Llama)

## 🔗 Resources

- **LangChain Docs**: https://python.langchain.com
- **OpenAI API**: https://platform.openai.com
- **FAISS**: https://github.com/facebookresearch/faiss
- **Streamlit**: https://streamlit.io

## 📝 License & Attribution

- **Data**: Pitti Immagine Exhibition (pitti.it)
- **Code**: Educational demo
- **Stack**: LangChain, OpenAI, FAISS, Streamlit

---

**Happy learning! 🚀**
