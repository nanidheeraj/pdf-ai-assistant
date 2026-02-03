

# ⚡ PDF Genius Pro

**Live App:** 👉 [https://flashpdf.streamlit.app/](https://flashpdf.streamlit.app/)

PDF Genius Pro is an **AI-powered PDF intelligence platform** that allows users to **summarize**, **visualize**, **quiz**, and **chat** with their PDF documents using **modern LLM + vector search techniques**.

It transforms static PDFs into an **interactive learning and analysis experience**.

---

## 🚀 Features

* 📄 **Multi-PDF Upload & Processing**
* 📝 **AI-generated Professional Summary**
* 🧠 **Mind Map Visualization of Concepts**
* 🎓 **Auto-generated MCQ Quiz with Explanations**
* 💬 **Smart Chatbot (RAG-based Q&A)**
* ⚡ Fast, responsive, and deployed on **Streamlit Cloud**

---

## 🛠️ Tech Stack & Tools Used

### 1. **Frontend & App Framework**

* **Streamlit**

  * Rapid UI development
  * Built-in state management
  * Perfect for ML/AI apps
  * Native deployment support

**Why Streamlit?**
Minimal boilerplate, fast iteration, and excellent support for AI workflows.

---

### 2. **PDF Processing**

* **PyPDF2**

  * Extracts raw text from uploaded PDFs
  * Lightweight and reliable

---

### 3. **LLM (Large Language Model)**

* **Groq + LLaMA 3.1 (8B Instant)**

  * Ultra-low latency inference
  * Strong reasoning & summarization
  * Cost-efficient for production use

**Why Groq?**
Much faster than traditional cloud LLM APIs with competitive quality.

---

### 4. **Embeddings & Vector Search**

* **FastEmbed Embeddings**

  * Model: `BAAI/bge-small-en-v1.5`
* **FAISS Vector Store**

  * Fast similarity search
  * In-memory vector indexing

**Why FAISS?**
Industry-standard for vector similarity search with excellent performance.

---

### 5. **RAG (Retrieval-Augmented Generation)**

* **LangChain**

  * `load_qa_chain`
  * Retrieval + LLM reasoning
* Ensures:

  * Factual answers
  * Reduced hallucinations
  * Context-aware responses

---

### 6. **Visualization**

* **streamlit-markmap**

  * Converts markdown hierarchies into interactive mind maps
  * Enhances conceptual understanding

---

### 7. **Security & Config**

* **python-dotenv**
* **Streamlit Secrets**

  * API keys never hardcoded
  * Secure deployment-ready configuration

---

## 🧠 How the System Works (End-to-End)

### Step 1: PDF Upload

* User uploads one or multiple PDFs
* PDFs are read using `PdfReader`
* All extracted text is combined into a single knowledge base

---

### Step 2: Text Chunking

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
```

**Why chunking?**

* Improves embedding quality
* Prevents token overflow
* Maintains semantic continuity

---

### Step 3: Vector Store Creation

* Text chunks → embeddings
* Stored in FAISS index
* Cached using `@st.cache_resource` for performance

---

### Step 4: Feature Modules

#### 📝 Summary

* LLM generates **professional bullet-point summaries**
* Optimized prompt design
* Only top content used (token-safe)

---

#### 🧠 Mind Map

* LLM converts document into **markdown hierarchy**
* Rendered using Markmap
* Interactive & visually intuitive

---

#### 🎓 Quiz Generator

* LLM generates **5 MCQs**
* Output strictly parsed as JSON
* Includes:

  * Question
  * Options
  * Correct answer
  * Explanation
* Multi-stage quiz flow:

  * Generate → Attempt → Evaluate → Review

---

#### 💬 Smart Chat (RAG)

1. User asks a question
2. FAISS retrieves top-k relevant chunks
3. LangChain feeds context + question to LLM
4. Answer generated with **source grounding**

---

## ⚙️ Optimization Techniques Used

### ✅ Caching

* `@st.cache_data` → PDF text extraction
* `@st.cache_resource` → Vector store creation
* Prevents recomputation on reruns

---

### ✅ Token Optimization

* Limited text passed to LLM (`[:6000]`, `[:10000]`)
* Prevents API overload
* Faster responses

---

### ✅ Temperature Control

* Summary & Chat: `temp=0.1–0.3`
* Quiz Generation: `temp=0.7`
* Balances accuracy vs creativity

---

### ✅ Session State Management

* Persistent quiz stages
* Chat history retained
* Clean UI reruns without data loss

---

### ✅ Error Handling

* Safe PDF reading
* Regex-based JSON extraction
* Graceful fallback for malformed LLM responses





---

## 🌐 Deployment

* Deployed on **Streamlit Cloud**
* Secure API key handling via secrets
* Scales automatically
* Accessible worldwide

🔗 **Live URL:** [https://flashpdf.streamlit.app/](https://flashpdf.streamlit.app/)

---

## 🎯 Use Cases

* Students & exam prep
* Research paper analysis
* Corporate reports
* Technical documentation
* Interview preparation
* Knowledge revision

---

## 🚧 Future Enhancements

* 🔍 Page-level citations
* 🗂️ PDF section-wise summaries
* 🎤 Voice-based Q&A
* 🧠 Concept linking across PDFs
---

## 👨‍💻 Author

**Dheeraj Reddy**


