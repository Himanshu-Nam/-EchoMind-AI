# 🎙️ EchoMind AI

> Chat with any YouTube video or audio file using AI (RAG-powered conversational assistant)

---

## 🚀 Overview

**EchoMind AI** is an intelligent Retrieval-Augmented Generation (RAG) chatbot that allows users to interact with **YouTube videos and audio files** in natural language.

It converts multimedia content into text using **Whisper + yt-dlp**, processes it into embeddings using **OpenAI**, stores them in **ChromaDB**, and enables **semantic search + conversational Q&A** using GPT-4o-mini.

---

## ✨ Features

- 🎥 Chat with YouTube videos directly
- 🎧 Upload and analyze audio files
- 🧠 Semantic search using vector embeddings
- 💬 Conversational memory (context-aware chat)
- 🌍 Multi-language support (Hindi, English, Hinglish)
- ⚡ Whisper-based transcription pipeline
- 🗂️ ChromaDB vector database
- 🖥️ Modern Streamlit UI
- 🔍 Accurate context-based answers (RAG)

---

## 🏗️ Architecture
YouTube / Audio File
│
▼
yt-api / Upload
│
▼
Whisper Transcription
│
▼
Text Chunking
│
▼
OpenAI Embeddings
│
▼
ChromaDB (Vector Store)
│
▼
Semantic Search
│
▼
GPT-4o-mini (RAG Answering)
│
▼
Chat Response
---

## 🛠️ Tech Stack

- Python 🐍
- Streamlit 🎨
- LangChain 🧠
- OpenAI GPT-4o-mini 🤖
- Whisper 🎙️
- yt-dlp 📺
- ChromaDB 🗄️
- FFmpeg 🎧

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/echomind-ai.git
cd echomind-ai