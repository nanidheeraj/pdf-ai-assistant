import streamlit as st
import os
import json
import time
import re
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# --- AI IMPORTS ---
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit_markmap import markmap

load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(page_title="PDF Genius Pro", page_icon="⚡", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header { font-size: 2.8rem; font-weight: 800; text-align: center; background: linear-gradient(45deg, #FF4B4B, #FF914D); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 5px; }
    .sub-text { text-align: center; color: #666; margin-bottom: 30px; }
    .quiz-card { background-color: #f0f2f6; padding: 20px; border-radius: 12px; margin-bottom: 15px; border-left: 5px solid #FF4B4B; color: #1f1f1f !important; }
    .stButton>button { width: 100%; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

@st.cache_data
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content: text += content + "\n"
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text

@st.cache_resource
def get_vectorstore(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(text)
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

def get_llm(temp=0.3):
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant", 
        temperature=temp, 
        max_retries=5
    )

def generate_quiz(text):
    llm = get_llm(temp=0.7)
    prompt = f"Generate 5 MCQs from this text. Return ONLY a JSON array with 'question', '3 or 4 options' (list), 'answer', and 'explanation'.\n\nText: {text[:7000]}"
    try:
        response = llm.invoke(prompt)
        match = re.search(r'\[\s*\{.*\}\s*\]', response.content, re.DOTALL)
        return json.loads(match.group(0)) if match else []
    except: return []

# --- SESSION INITIALIZATION ---
if 'messages' not in st.session_state: st.session_state.messages = []
if 'quiz_stage' not in st.session_state: st.session_state.quiz_stage = 0

# --- SIDEBAR ---
with st.sidebar:
    st.image("data:image/png;base64,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", width=80)
    st.title("PDF")
    pdf_docs = st.file_uploader("Upload Knowledge Base", type="pdf", accept_multiple_files=True)
    
    if st.button("🚀 Process Documents", type="primary"):
        if pdf_docs:
            with st.spinner("Crunching data..."):
                raw_text = get_pdf_text(pdf_docs)
                st.session_state.raw_text = raw_text
                st.session_state.vectorstore = get_vectorstore(raw_text)
                st.session_state.messages = [] # Clear chat on new upload
                st.success("Analysis Complete!")
        else:
            st.warning("Please upload a file first.")

    if 'raw_text' in st.session_state:
        st.divider()
        st.metric("Total Characters", len(st.session_state.raw_text))
        st.metric("Est. Reading Time", f"{max(1, len(st.session_state.raw_text)//3000)} mins")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

# --- MAIN UI ---
st.markdown('<div class="main-header">⚡ PDF Genius Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Interact, Visualize, and Master your documents.</div>', unsafe_allow_html=True)

if 'raw_text' in st.session_state:
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🧠 Mind Map", "🎓 Quiz", "💬 Smart Chat"])

    with tab1:
        if st.button("✨ Generate AI Summary"):
            with st.spinner("Synthesizing..."):
                llm = get_llm()
                res = llm.invoke(f"Summarize the key takeaways in professional bullet points:\n{st.session_state.raw_text[:10000]}")
                st.session_state.summary = res.content
        if 'summary' in st.session_state:
            st.info("Key Takeaways")
            st.markdown(st.session_state.summary)

    with tab2:
        if st.button("🗺️ Visualize Concepts"):
            with st.spinner("Building Map..."):
                llm = get_llm()
                res = llm.invoke(f"Create a Markdown hierarchy (using -) for the main topics of: {st.session_state.raw_text[:6000]}")
                st.session_state.mindmap = res.content.replace("```markdown", "").replace("```", "")
        if 'mindmap' in st.session_state:
            markmap(st.session_state.mindmap, height=500)

    with tab3:
                st.subheader("🎓 Knowledge Check")
                
                # --- STAGE 0: INITIAL STATE ---
                if st.session_state.quiz_stage == 0:
                    st.write("Click below to generate 5 questions from your document.")
                    if st.button("🎯 Generate New Quiz", key="gen_new_quiz_start"):
                        with st.spinner("Analyzing text..."):
                            data = generate_quiz(st.session_state.raw_text)
                            if data:
                                st.session_state.quiz_data = data
                                st.session_state.quiz_stage = 1
                                st.rerun()

                # --- STAGE 1: TAKING THE QUIZ ---
                elif st.session_state.quiz_stage == 1:
                    with st.form("quiz_form"):
                        user_answers = {}
                        for i, q in enumerate(st.session_state.quiz_data):
                            st.markdown(f'<div class="quiz-card"><b>Question {i+1}:</b> {q["question"]}</div>', unsafe_allow_html=True)
                            # We use the full string of the option as the key
                            user_answers[i] = st.radio("Select the correct option:", q["options"], key=f"ans_{i}")
                        
                        if st.form_submit_button("Submit Exam"):
                            st.session_state.final_answers = user_answers
                            st.session_state.quiz_stage = 2
                            st.rerun()

                # --- STAGE 2: RESULTS & REVIEW ---
                elif st.session_state.quiz_stage == 2:
                    # Calculate Score
                    score = sum(1 for i, q in enumerate(st.session_state.quiz_data) 
                            if st.session_state.final_answers.get(i) == q["answer"])
                    
                    # 1. SCORE DISPLAY
                    st.markdown(f"""
                        <div style="background: linear-gradient(45deg, #FF4B4B, #FF914D); padding:20px; border-radius:12px; margin-bottom:25px; text-align:center;">
                            <h2 style="color:white; margin:0;">🏆 Final Score: {score} / {len(st.session_state.quiz_data)}</h2>
                        </div>
                    """, unsafe_allow_html=True)

                    # 2. DETAILED QUESTION REVIEW
                    for i, q in enumerate(st.session_state.quiz_data):
                        u_ans = st.session_state.final_answers.get(i)
                        is_correct = (u_ans == q["answer"])
                        
                        st.markdown(f"#### Question {i+1}")
                        st.write(q["question"])
                        
                        # Custom display for Your Answer vs Correct Answer
                        if is_correct:
                            st.success(f"**Your Answer:** {u_ans}")
                        else:
                            st.error(f"**Your Answer:** {u_ans}")
                            st.info(f"✅ **Correct Answer:** {q['answer']}")
                        
                        st.write(f"*Explanation:* {q.get('explanation', 'Refer to the document sections for more details.')}")
                        st.divider()

                    # 3. NAVIGATION BUTTONS
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Retake Same Quiz", use_container_width=True):
                            # Goes back to Stage 1 with existing data
                            st.session_state.quiz_stage = 1
                            st.rerun()
                    with col2:
                        if st.button("✨ Generate New Quiz", type="primary", use_container_width=True):
                            # Resets to Stage 0 to fetch fresh questions
                            st.session_state.quiz_stage = 0
                            st.rerun()

    with tab4:
            st.subheader("💬 Smart Chat")

            # Container for the chat history
            chat_placeholder = st.container()

            # Render history in the container
            with chat_placeholder:
                for m in st.session_state.messages:
                    with st.chat_message(m["role"]):
                        st.markdown(m["content"])
                        if "sources" in m:
                            with st.expander("View Reference Context"):
                                for doc in m["sources"]:
                                    st.caption(f"Source: ...{doc.page_content[:250]}...")

            # Input box always stays at the bottom
            if prompt := st.chat_input("Ask a question about the PDF contents..."):
                # Display user message immediately
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_placeholder:
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        # Process RAG
                        docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
                        llm = get_llm(temp=0.1)
                        chain = load_qa_chain(llm, chain_type="stuff")
                        
                        # Generate response
                        response = chain.run(input_documents=docs, question=prompt)
                        st.markdown(response)
                        
                        # Save to state
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": docs
                        })
                
                # Force rerun to ensure the history container updates correctly
                st.rerun()
else:
    st.info("👈 Please upload and 'Process' a PDF to begin your journey.")
