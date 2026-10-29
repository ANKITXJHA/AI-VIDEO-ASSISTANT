import os

import torch
import whisper

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# Maps the language choice to Whisper's (language code, task) settings.
# - english:  speech is English, transcribe as is
# - hindi:    speech is Hindi, output stays in Hindi (Devanagari)
# - hinglish: speech is Hindi/Hinglish, output is translated to English
LANGUAGE_SETTINGS = {
    "english": {"language": "en", "task": "transcribe"},
    "hindi": {"language": "hi", "task": "transcribe"},
    "hinglish": {"language": "hi", "task": "translate"},
}

_model = None


def load_model():
    """Load the Whisper model once and reuse it for every chunk."""
    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")
    return _model


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """Transcribe one audio chunk with the local Whisper model."""
    settings = LANGUAGE_SETTINGS.get(language.lower())
    if settings is None:
        raise ValueError(
            f"Unsupported language '{language}'. Choose from: {', '.join(LANGUAGE_SETTINGS)}"
        )

    model = load_model()
    result = model.transcribe(
        chunk_path,
        language=settings["language"],
        task=settings["task"],
        # FP16 only works on NVIDIA GPUs; on CPU this avoids the warning
        fp16=torch.cuda.is_available(),
    )
    return result["text"].strip()


def transcribe_all(chunks: list, language: str = "english") -> str:
    """Transcribe all chunks in order and join them into one transcript."""
    print(f"Using local Whisper ({WHISPER_MODEL}) for '{language}' transcription.")

    parts = []
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)
        if text:
            parts.append(text)

    print("Transcription complete.")
    return " ".join(parts)