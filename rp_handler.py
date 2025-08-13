import os
MODEL_PATH = os.getenv("MODEL_PATH", "/runpod-volume/models/your-model.gguf")
HUMANWATER_TEST_ENV = os.getenv("HUMANWATER_TEST_ENV", "none");
import runpod

def handler(event):
    try:

        from llama_cpp.llama_cpp import llama_supports_gpu_offload
        offload = bool(llama_supports_gpu_offload())
        return {"gpu_offload_supported": offload}
    except ImportError:
        print("fail to check gpu available (ImportError)")

    print(f"Worker Start")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"HUMANWATER_TEST_ENV: {HUMANWATER_TEST_ENV}")
    input = event['input']
    prompt = input.get('prompt')

    return prompt

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})


