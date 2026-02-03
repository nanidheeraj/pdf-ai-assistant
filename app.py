import streamlit as st
import os
import json
import random
import re
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# --- IMPORTS ---
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit_markmap import markmap

# Load environment variables
load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(page_title="PDF Genius", page_icon="⚡", layout="wide")

# --- CUSTOM CSS (Fixed for Visibility) ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 800; text-align: center;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF914D);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    /* Fixed the quiz card text color to be visible in all modes */
    .quiz-card {
        background-color: #f0f2f6; 
        padding: 20px; 
        border-radius: 12px;
        margin-bottom: 15px; 
        border-left: 5px solid #FF4B4B;
        color: #1f1f1f !important; /* Force dark text for visibility */
    }
    .quiz-card b {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- BACKEND FUNCTIONS ---

@st.cache_data
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content: text += content + "\n"
    return text

@st.cache_resource
def get_vectorstore(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_llm(temperature=0.3):
    return ChatGroq(model_name="llama-3.1-8b-instant", temperature=temperature, max_retries=3)

def generate_quiz(text):
    llm = get_llm(temperature=0.6)
    # Refined prompt for stricter JSON output
    prompt = f"""
    Based on the following text, generate 5 Multiple Choice Questions.
    Return ONLY a valid JSON array. Do not include any conversational text.
    
    JSON Structure:
    [
        {{
            "question": "The question text",
            "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
            "answer": "The exact string of the correct option",
            "explanation": "Brief explanation"
        }}
    ]
    
    Text: {text[:7000]}
    """
    try:
        response = llm.invoke(prompt)
        # Regex to find the JSON array in case the LLM adds chatter
        match = re.search(r'\[\s*\{.*\}\s*\]', response.content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return []
    except Exception as e:
        st.error(f"Quiz Generation Error: {e}")
        return []

# --- SESSION STATE ---
if 'quiz_stage' not in st.session_state: st.session_state.quiz_stage = 0
if 'messages' not in st.session_state: st.session_state.messages = []
if 'quiz_data' not in st.session_state: st.session_state.quiz_data = []

# --- MAIN APP ---
def main():
    st.markdown('<div class="main-header">⚡ PDF Genius</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("📂 Upload")
        pdf_docs = st.file_uploader("Upload PDF", accept_multiple_files=True)
        if st.button("🚀 Process Document", type="primary"):
            if pdf_docs:
                with st.spinner("Analyzing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    st.session_state.raw_text = raw_text
                    st.session_state.vectorstore = get_vectorstore(raw_text)
                    st.session_state.quiz_stage = 0 
                    st.session_state.messages = []
                    st.success("Ready!")

    if 'raw_text' in st.session_state:
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🧠 Mind Map", "🎓 Quiz", "💬 Chat"])

        with tab1:
            if st.button("Generate Summary"):
                with st.spinner("Summarizing..."):
                    llm = get_llm()
                    st.session_state.summary = llm.invoke(f"Summarize in 5 bullet points:\n{st.session_state.raw_text[:8000]}").content
            if 'summary' in st.session_state:
                st.markdown(st.session_state.summary)

        with tab2:
            if st.button("Generate Map"):
                with st.spinner("Mapping..."):
                    llm = get_llm()
                    res = llm.invoke(f"Create a hierarchical Markdown list (use -) for: {st.session_state.raw_text[:6000]}")
                    st.session_state.mindmap = res.content.replace("```markdown", "").replace("```", "")
            if 'mindmap' in st.session_state:
                markmap(st.session_state.mindmap, height=500)

        with tab3:
            st.subheader("Knowledge Check")
            
            if st.session_state.quiz_stage == 0:
                if st.button("Generate 5 Questions", key="gen_quiz"):
                    with st.spinner("Creating Quiz..."):
                        data = generate_quiz(st.session_state.raw_text)
                        if data:
                            st.session_state.quiz_data = data
                            st.session_state.quiz_stage = 1
                            st.rerun()
                        else:
                            st.error("Failed to generate quiz. Try again.")

            elif st.session_state.quiz_stage == 1:
                with st.form("quiz_form"):
                    user_answers = {}
                    for i, q in enumerate(st.session_state.quiz_data):
                        # The CSS now forces this text to be dark and visible
                        st.markdown(f'<div class="quiz-card"><b>Q{i+1}:</b> {q["question"]}</div>', unsafe_allow_html=True)
                        user_answers[i] = st.radio("Select answer:", q["options"], key=f"ans_{i}")
                    
                    if st.form_submit_button("Submit Answers"):
                        st.session_state.final_answers = user_answers
                        st.session_state.quiz_stage = 2
                        st.rerun()

            elif st.session_state.quiz_stage == 2:
                score = 0
                for i, q in enumerate(st.session_state.quiz_data):
                    u_ans = st.session_state.final_answers.get(i)
                    if u_ans == q["answer"]:
                        score += 1
                        st.success(f"Q{i+1}: Correct! ✔️")
                    else:
                        st.error(f"Q{i+1}: Incorrect. Correct answer: {q['answer']}")
                        st.info(f"💡 {q.get('explanation', '')}")
                
                st.metric("Total Score", f"{score} / 5")
                if st.button("Start New Quiz"):
                    st.session_state.quiz_stage = 0
                    st.rerun()

        with tab4:
            chat_container = st.container()
            if prompt := st.chat_input("Ask about the PDF..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                docs = st.session_state.vectorstore.similarity_search(prompt)
                llm = get_llm()
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})

            with chat_container:
                for m in st.session_state.messages:
                    with st.chat_message(m["role"]): st.markdown(m["content"])
    else:
        st.info("Upload a PDF to start.")

if __name__ == "__main__":
    main()