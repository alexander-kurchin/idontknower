import base64
import os
import time

import nest_asyncio
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import ds.sr_tg_module as sr_tg
import ds.tts_module as tts


nest_asyncio.apply()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory="assets"), name="assets")
templates = Jinja2Templates(directory="templates")

if not os.path.exists("temp"):
    os.mkdir("temp")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat/")
async def chat(voice: UploadFile = File()):
    suffix = str(int(time.time()))
    input_file = f"temp/input_{suffix}.webm"
    try:
        with open(input_file, "wb") as f:
            f.write(voice.file.read())
        with open(input_file, "rb") as f:
            output = sr_tg.get_text(f)
        prompt = sr_tg.fix_sr_output(output[0])
        text = sr_tg.fix_tg_output(output[1])
        audio = tts.get_audio(text)
        audio_base64 = base64.b64encode(audio)
        if os.path.isfile(input_file):
            os.remove(input_file)
    except Exception as e:
        print(e)
        if os.path.isfile(input_file):
            os.remove(input_file)
        return {}
    else:
        return {
            "status": "ok",
            "response": {
                "prompt": prompt,
                "text": text,
                "audio_base64": b"data:audio/webm;base64," + audio_base64,
            },
        }
