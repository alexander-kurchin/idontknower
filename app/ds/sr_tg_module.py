import os
import re

import requests

sr_tg_url = os.getenv("SR_TG_URL")
url = f"{sr_tg_url}/sr-tg/"


def get_text(voice):
    r = requests.post(url, files={"file": voice}, timeout=25)
    prompt = r.json()["prompt"]
    text = r.json()["response"]
    time_sr = r.json()["time_sr"]
    time_tg = r.json()["time_tg"]
    if text.isdigit():
        text = "Чудесный сегодня денёк!"
    return prompt.strip(), text.strip(), time_sr, time_tg


def fix_tg_output(text):
    # cut answer
    pattern = "[/INST]"
    cut_point = text.find(pattern) + len(pattern)
    text = text[cut_point:]

    # clean answer
    pattern = r"[<>/\[\]]"
    text = re.sub(pattern, "", text)
    cut_point = text.find("INST")
    output = text[:cut_point]
    if output.find("INST") != -1:
        output = text[: output.find("INST")]

    # last punctuation
    last_punctuation_index = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last_punctuation_index != -1:
        output = output[: last_punctuation_index + 1]

    return output.strip()


def fix_sr_output(text):
    dunno_stupid_names = [
        "незнайка",
        "Не знай",
        "Не знай ка",
        "Не знаю ка",
        "Не знай-ка",
    ]
    for stupid_name in dunno_stupid_names:
        if text.find(stupid_name) != -1:
            t = text.replace(stupid_name, "Незнайка")
            return t
    return text
