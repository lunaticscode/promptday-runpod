import os
import runpod
from typing import Union
from llama_cpp import Llama
MODEL_PATH = os.getenv("MODEL_PATH", "/runpod-volume/models/promptday_Q4_K_M.gguf")
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
    prompt = f"""
            Extract below things from Input Email. And output should me JSON(parsable) format. 
            - `title`: Title of the main event (e.g., "SSN Trophy 2026 Opening Ceremony").
            - `location`: The venue or location mentioned.
            - `organizer`: Contextual cues about who is hosting or inviting, even if not named as the sender.
            - `tags`: Relevant keywords to classify the event (e.g., ["sports", "ceremony"]).
            - `dates`: A list of all date/time expressions(ISO string), classified into:
              - `"event"`: Main event start (e.g., opening ceremony, finals)
              - `"deadline"`: Submission, registration, or application deadlines
              - `"sub_event"`: Sub-events such as matches, rounds, rehearsals, or workshops

            Input Email: {user_input}
    """
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
    return text

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})


