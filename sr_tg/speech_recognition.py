import os
import time

import whisper


model_size = os.getenv("SR_MODEL")
model = whisper.load_model(model_size)


def whisper_inference(file: str):
    start = time.time()
    audio_file = whisper.load_audio(file)
    audio_file = whisper.pad_or_trim(audio_file)
    mel = whisper.log_mel_spectrogram(audio_file).to(model.device)
    options = whisper.DecodingOptions(language="ru")
    result = whisper.decode(model, mel, options)
    end = time.time() - start
    return result.text, round(end, 4)
