import os
import time

import torch
from huggingface_hub import login
from peft import PeftConfig, PeftModel
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig)
from transformers.generation import GenerationConfig


access_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
login(token=access_token)


device = "cuda" if torch.cuda.is_available() else "cpu"
# Model and tokenizer names
base_model_name = os.getenv("MODEL_NAME")
peft_model_id = os.getenv("LORA_ADAPTERS")

MODEL_INFO = f"model: {base_model_name}, adapters: {peft_model_id}, device: {device}"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map={"": 0}
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1
base_model.config.max_new_tokens = 60
base_model.config.min_length = 10

# inference congig
gen_cfg = GenerationConfig.from_model_config(base_model.config)
gen_cfg.max_new_tokens = 60
gen_cfg.min_length = 10

# Load the Lora model
if peft_model_id == "none":
    model = base_model
else:
    config = PeftConfig.from_pretrained(peft_model_id)
    model = PeftModel.from_pretrained(
        base_model,
        peft_model_id,
        device_map={"": 0}
    )


def inference(text):
    start = time.time()
    input_ids = tokenizer(
        f"Диалог между человеком и Незнайкой [INST] {text} [/INST]",
        return_tensors="pt"
    ).to(device)
    output_tokens = model.generate(
        **input_ids,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=64,
        do_sample=True,
        top_k=1,
    )
    text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    inference_time = time.time() - start
    return text.strip(), round(inference_time, 4)
