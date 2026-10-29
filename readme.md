# AI Video Assistant

Turn any video or audio file into a transcript, summary, action items, key decisions and open questions, then chat with it. Answers come only from the transcript (RAG).

## Demo

<!-- Add your demo video link here -->

## Features

- Transcription with local Whisper (English, Hindi, and Hinglish translated to English)
- Auto-generated title and bullet-point summary
- Action items, key decisions and open questions
- RAG chat over the transcript (Chroma + Groq)
- Streamlit UI with downloadable report and transcript

## Tech stack

Python, Whisper, LangChain (LCEL), Groq, ChromaDB, Sentence Transformers, yt-dlp, pydub, Streamlit

## How it works

1. Audio is extracted, converted to WAV and split into chunks
2. Each chunk is transcribed with Whisper
3. LLM chains produce the title, summary, action items, decisions and questions
4. The transcript is embedded into Chroma for retrieval-based chat

## Project structure

```
uiapp.py          Streamlit UI
main.py           Pipeline and command-line entry point
core/             Transcription, summarizing, extraction and RAG logic
utils/            Audio download, conversion and chunking
requirements.txt  Dependencies
```

## Run locally

Requirements: Python 3.11 or 3.12 and ffmpeg installed on your system.

```bash
pip install -r requirements.txt
cp .env.example .env     # then add your Groq API key
streamlit run uiapp.py
```

You can also run the pipeline in the terminal with `python main.py`.

## Limitations

- Runs locally. The Whisper and embedding models need more RAM than free cloud tiers offer
- YouTube downloads can be blocked (HTTP 429), so uploading a file is more reliable
- Hindi accuracy depends on the Whisper model size (`small` or larger recommended)