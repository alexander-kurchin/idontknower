import os

import requests

tts_url = os.getenv("TTS_URL")
url = f"{tts_url}/tts/"


def get_audio(text="Привет!"):
    r = requests.get(url, params={"text": text}, timeout=25)
    return r.content
