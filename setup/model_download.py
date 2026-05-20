
import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(token=os.environ['hf_token'])

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')

    return model, tokenizer, device