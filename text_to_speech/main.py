import time

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from TTS.api import TTS


device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


def text_to_speech(text):
    start = time.time()
    tts.tts_to_file(
        text=text,
        speaker_wav="voice_reference.wav",  # your voice reference file
        language="ru",
        file_path="output.wav",
    )
    end = time.time() - start
    return "output.wav", round(end, 4)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = "./"


@app.get("/tts/")
async def get_wav_file(text: str):
    filepath, time_tts = text_to_speech(text)
    wav_file_path = static_path + filepath
    return FileResponse(wav_file_path, media_type="audio/wav", filename="output.wav")
