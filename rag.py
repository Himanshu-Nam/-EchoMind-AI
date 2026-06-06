from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
load_dotenv()
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from core.transcriber import transcribe_all
from langchain_openai import OpenAIEmbeddings
from utils.audio_processor import (
    download_youtube_audio,
    process_input
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    max_completion_tokens=300
)

source = input("Enter YouTube URL or File Path: ").strip().strip('"').strip("'")
language = "english"
if source.startswith(("http://", "https://")):
    transcript = download_youtube_audio(source)
else:
    chunks = process_input(source)
    transcript = transcribe_all(chunks, language=language)

#create a chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(transcript)


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory="./db"
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that answers questions
            only from the provided transcript context.

            Rules:
            - Use only the information in the context.
            - Do not make up facts.
            - If the answer is not found in the context, say:
              "I could not find that information in the transcript."
            - Be concise and accurate.
            """,
        ),
        (
            "human",
            """
            Context:
            {context}

            Question:
            {question}
            """,
        ),
    ]
)

query = "What are the benefits of RAG?"

docs = retriever.invoke(query)

context = "\n\n".join(
    doc.page_content
    for doc in docs
)

response = llm.invoke(
    prompt.format(
        context=context,
        question=query
    )
)

print(response.content)