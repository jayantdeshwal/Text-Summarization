🦜 LangChain: Summarize YouTube or Website Content

A Streamlit app built with LangChain, Groq LLM, and Unstructured/Youtube loaders to extract and summarize content from YouTube videos or websites — automatically translating non-English transcripts (e.g., Hindi) to English before summarization.

🌐 Demo

🚀 Try it live:https://text-summarization-bbptjrsutlnxx5najptvxd.streamlit.app/

📽️ Overview

This app allows you to:

Paste a YouTube video URL or website URL

Automatically extract transcripts or article text

Translate non-English (e.g., Hindi) transcripts into English

Summarize the content in around 300 words using Groq’s LLaMA 3.3 model

⚙️ Tech Stack
Component	Description
Frontend	Streamlit
LLM Provider	Groq API

Framework	LangChain
Models	llama-3.3-70b-versatile
Loaders	YoutubeLoader, UnstructuredURLLoader
Text Processing	RecursiveCharacterTextSplitter
Translation	Groq LLM (prompt-based translation)
🧠 How It Works

Input: User enters a YouTube or website URL.

Extraction:

If YouTube → fetches the transcript (supports Hindi/English).

If website → extracts readable text using Unstructured Loader.

Translation:

Detects non-English (e.g., Hindi) and translates to English using LLM.

Summarization:

Uses a LangChain summarization chain (map_reduce) to generate a concise summary.

Output:

Displays the English summary in the Streamlit UI

🧑‍💻 Author

Jayant Deshwal
💼 GitHub: https://github.com/jayantdeshwal

📧 Email: jayant.deshwal.56@gmail.com

📝 License

This project is licensed under the MIT License — free to use, modify, and distribute.
