import os
import runpod
from typing import Union
from llama_cpp import Llama
import json
MODEL_PATH = os.getenv("MODEL_PATH", "/runpod-volume/models/promptday_0815_Q4_K_M.gguf")
HUMANWATER_TEST_ENV = os.getenv("HUMANWATER_TEST_ENV", "none")

_llm = None
_model_path = None
N_CTX = int(os.getenv("N_CTX", "2048"))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "1"))
def _resolved_model_path(path: str) -> Union[str, bool]:
    if path.endswith(".gguf") and os.path.isfile(path):
        return path
    return False

def _load_llm_once():
    global _llm, _model_path
    if _llm is not None:
        return
    _model_path = _resolved_model_path(MODEL_PATH)
    if _model_path:
        _llm = Llama(
            model_path=_model_path,
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            use_mmap=True,
            use_mlock=False,
        )
        print("Success to load model.")
    else:
        print("(!) Invalid model path")
        return {"message": "(!) Cannot find model."}


def get_prompt(user_input):

    SYSTEM_PROMPT = (
    "You are a professional scheduling assistant. "
    "Given an email about an event or meeting, extract structured fields as JSON. "
    "Always return valid JSON only."
    )

    INSTRUCTION_TEMPLATE = """Extract fields from the following email.
    
    Return JSON with this shape:
    
    {
    "title": "...",
    "location": "...",
    "organizer": "...",
    "tags": ["...", "..."],
    "dates": [{ "type": "event|deadline|sub_event", "text": "...", "iso": "..." }]
    }
    
    Rules:
    - Output must be valid JSON and contain ALL keys.
    - "tags" should be a short list of keywords (lowercase, concise).
    - "dates.iso" MUST be ISO-8601 (date or datetime, or date range 'YYYY-MM-DD/YYYY-MM-DD').
    - If a field is missing, put an empty string ("") or an empty array ([]).
    """

    prompt = (
    f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
    f"<|user|>\n{INSTRUCTION_TEMPLATE}\n\nEMAIL:\n{user_input}\n</s>\n<|assistant|>\n"
    )

    return prompt

def handler(event):
    # try:
    #     from llama_cpp.llama_cpp import llama_supports_gpu_offload
    #     offload = bool(llama_supports_gpu_offload())
    #     print(f"gpu_offload_supported: {offload}")
    # except ImportError:
    #     print("fail to check gpu available (ImportError...fuck...)")
    _load_llm_once()
    print(f"Worker Start")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"HUMANWATER_TEST_ENV: {HUMANWATER_TEST_ENV}")
    input = event['input']
    prompt = input.get('prompt', "")
    out = _llm(prompt=get_prompt(prompt), max_tokens=512, temperature=0.7, echo=False, stop=["</s>"])
    text = out.get("choices", [{}])[0].get("text", "")
    result = text
    try:
        result = json.loads(text)
    except:
        print('(!) Invalid json format')
        pass
    return result

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})


