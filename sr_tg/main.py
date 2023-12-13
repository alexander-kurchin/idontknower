import os

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from speech_recognition import whisper_inference
from text_generation import MODEL_INFO, inference


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists("files"):
    os.mkdir("files")
static_path = "./files/"


@app.post("/sr-tg/")
async def upload_wav_w(file: UploadFile = File(...)):
    file_path = static_path + file.filename
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    prompt, time_sr = whisper_inference(file_path)  # sr
    answer, time_tg = inference(prompt)  # tg

    if os.path.isfile(file_path):
        os.remove(file_path)

    return {
        "info": MODEL_INFO,
        "prompt": prompt,
        "response": answer,
        "time_sr": time_sr,
        "time_tg": time_tg,
    }
