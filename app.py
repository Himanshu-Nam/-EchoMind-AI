import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from core.transcriber import transcribe_all
from utils.audio_processor import (
    download_youtube_audio,
    process_input
)

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
# CSS
# =====================================================

st.markdown("""
<style>

.main-title{
    text-align:center;
    font-size:3rem;
    font-weight:700;
}

.subtitle{
    text-align:center;
    color:#888;
    margin-bottom:20px;
}

.block-container{
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 class='main-title'>🎙️ EchoMind AI</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Chat with YouTube Videos & Audio Files</div>",
    unsafe_allow_html=True
)

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
# LLM
# =====================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("📥 Content Source")

    youtube_url = st.text_input(
        "YouTube URL"
    )

    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=["wav", "mp3", "m4a"]
    )

    process_btn = st.button(
        "🚀 Process Content",
        use_container_width=True
    )

    if st.session_state.ready:
        st.success("Transcript Ready")

# =====================================================
# PROCESS CONTENT
# =====================================================

if process_btn:

    transcript = None

    with st.spinner("Generating transcript..."):

        try:

            # -------------------------
            # YouTube
            # -------------------------

            if youtube_url:

                transcript = download_youtube_audio(
                    youtube_url
                )

            # -------------------------
            # Audio File
            # -------------------------

            elif uploaded_file:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav"
                ) as tmp:

                    tmp.write(
                        uploaded_file.read()
                    )

                    file_path = tmp.name

                chunks = process_input(
                    file_path
                )

                transcript = transcribe_all(
                    chunks,
                    language="english"
                )

            else:

                st.warning(
                    "Provide YouTube URL or upload audio."
                )
                st.stop()

            # -------------------------
            # Chunking
            # -------------------------

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            docs = splitter.split_text(
                transcript
            )

            # -------------------------
            # Embeddings
            # -------------------------

            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )

            vectorstore = Chroma.from_texts(
                texts=docs,
                embedding=embeddings
            )

            st.session_state.vectorstore = (
                vectorstore
            )

            st.session_state.transcript = (
                transcript
            )

            st.session_state.ready = True

            st.success(
                "Transcript processed successfully!"
            )

        except Exception as e:

            st.error(str(e))

# =====================================================
# SHOW CHAT HISTORY
# =====================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

# =====================================================
# CHAT
# =====================================================

if st.session_state.ready:

    query = st.chat_input(
        "Ask anything about the transcript..."
    )

    if query:

        # -------------------------
        # Show User Message
        # -------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):
            st.markdown(query)

        # -------------------------
        # Semantic Search
        # -------------------------

        retriever = (
            st.session_state.vectorstore
            .as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 5
                }
            )
        )

        docs = retriever.invoke(
            query
        )

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        # -------------------------
        # Prompt
        # -------------------------

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are EchoMind AI.

                    Answer ONLY using the
                    provided transcript context.

                    Rules:

                    - Use transcript context only.
                    - Use chat history for follow-up questions.
                    - Never make up facts.
                    - If answer is not available say:

                    'I could not find that information in the transcript.'
                    """
                ),

                MessagesPlaceholder(
                    variable_name="chat_history"
                ),

                (
                    "human",
                    """
                    Context:
                    {context}

                    Question:
                    {question}
                    """
                )
            ]
        )

        chain = prompt | llm

        # -------------------------
        # LLM Call
        # -------------------------

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Thinking..."
            ):

                response = chain.invoke(
                    {
                        "context": context,
                        "question": query,
                        "chat_history":
                            st.session_state.chat_history
                    }
                )

                answer = response.content

                st.markdown(answer)

        # -------------------------
        # Save Chat
        # -------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.session_state.chat_history.append(
            HumanMessage(
                content=query
            )
        )

        st.session_state.chat_history.append(
            AIMessage(
                content=answer
            )
        )

        # Keep last 10 exchanges
        st.session_state.chat_history = (
            st.session_state.chat_history[-20:]
        )

# =====================================================
# TRANSCRIPT VIEWER
# =====================================================

if st.session_state.ready:

    with st.expander(
        "📄 View Transcript"
    ):
        st.write(
            st.session_state.transcript
        )