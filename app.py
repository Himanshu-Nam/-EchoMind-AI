import tempfile
import streamlit as st
import datetime
from io import BytesIO

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from core.transcriber import transcribe_all
from utils.audio_processor import download_youtube_audio, process_input


# =====================================================
# CONFIG
# =====================================================

load_dotenv()

st.set_page_config(
    page_title="EchoMind AI",
    page_icon="🎙️",
    layout="wide"
)

# =====================================================
# LLM
# =====================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)

# =====================================================
# PDF GENERATOR (Transcript / QA)
# =====================================================

def create_pdf(text: str):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()
    content = []

    for line in text.split("\n"):
        content.append(Paragraph(line, styles["Normal"]))
        content.append(Spacer(1, 5))

    doc.build(content)
    buffer.seek(0)
    return buffer


# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "ready" not in st.session_state:
    st.session_state.ready = False

if "transcript" not in st.session_state:
    st.session_state.transcript = ""


# =====================================================
# UI HEADER
# =====================================================

st.title("🎙️ EchoMind AI")
st.subheader("Chat with YouTube Videos & Audio Files")


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("📥 Content Source")

    youtube_url = st.text_input("YouTube URL")

    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=["wav", "mp3", "m4a"]
    )

    process_btn = st.button("🚀 Process Content", use_container_width=True)

    # =================================================
    # DOWNLOAD SECTION
    # =================================================

    if st.session_state.ready:

        st.success("Transcript Ready")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # ---------------- TXT DOWNLOAD ----------------
        st.download_button(
            "📥 Download Transcript (TXT)",
            data=st.session_state.transcript,
            file_name=f"transcript_{timestamp}.txt",
            mime="text/plain"
        )

        # ---------------- PDF DOWNLOAD ----------------
        pdf_buffer = create_pdf(st.session_state.transcript)

        st.download_button(
            "📄 Download Transcript (PDF)",
            data=pdf_buffer,
            file_name=f"transcript_{timestamp}.pdf",
            mime="application/pdf"
        )

        # ---------------- QA SUMMARY ----------------
        if st.button("🧠 Generate QA Summary"):
            qa_text = ""

            for msg in st.session_state.chat_history:
                if isinstance(msg, HumanMessage):
                    qa_text += f"Q: {msg.content}\n"
                else:
                    qa_text += f"A: {msg.content}\n\n"

            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", "Summarize the following Q&A into clean notes."),
                ("human", qa_text)
            ])

            summary_chain = summary_prompt | llm
            summary = summary_chain.invoke({}).content

            st.session_state.qa_summary = summary

        # QA DOWNLOAD
        if "qa_summary" in st.session_state:

            st.download_button(
                "📥 Download QA Summary (TXT)",
                data=st.session_state.qa_summary,
                file_name=f"qa_summary_{timestamp}.txt",
                mime="text/plain"
            )


# =====================================================
# PROCESS CONTENT
# =====================================================

if process_btn:

    transcript = None

    with st.spinner("Generating transcript..."):

        try:

            if youtube_url:
                transcript = download_youtube_audio(youtube_url)

            elif uploaded_file:

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(uploaded_file.read())
                    file_path = tmp.name

                chunks = process_input(file_path)
                transcript = transcribe_all(chunks, language="english")

            else:
                st.warning("Provide YouTube URL or upload audio.")
                st.stop()

            # ---------------- Chunking ----------------
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            docs = splitter.split_text(transcript)

            # ---------------- Embeddings ----------------
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )

            vectorstore = Chroma.from_texts(
                texts=docs,
                embedding=embeddings
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.transcript = transcript
            st.session_state.ready = True

            st.success("Transcript processed successfully!")

        except Exception as e:
            st.error(str(e))


# =====================================================
# CHAT HISTORY UI
# =====================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =====================================================
# CHAT ENGINE
# =====================================================

if st.session_state.ready:

    query = st.chat_input("Ask anything about the transcript...")

    if query:

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        docs = retriever.invoke(query)

        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are EchoMind AI. Answer ONLY using transcript context."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human",
             "Context:\n{context}\n\nQuestion:\n{question}")
        ])

        chain = prompt | llm

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = chain.invoke({
                    "context": context,
                    "question": query,
                    "chat_history": st.session_state.chat_history
                })

                answer = response.content
                st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        st.session_state.chat_history.append(HumanMessage(content=query))
        st.session_state.chat_history.append(AIMessage(content=answer))

        st.session_state.chat_history = st.session_state.chat_history[-20:]


# =====================================================
# TRANSCRIPT VIEWER
# =====================================================

if st.session_state.ready:
    with st.expander("📄 View Transcript"):
        st.write(st.session_state.transcript)