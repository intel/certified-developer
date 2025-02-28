import uvicorn
import os
import requests

from tqdm import tqdm
from langchain.llms import GPT4All
from fastapi import FastAPI
from model import GenPayload
from PickerBot import PickerBot
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

SAFE_BASE_DIR = os.path.join(os.path.expanduser("~"), "mlops", "lab8")

app = FastAPI()


def load_gpt4allj(
    model_path: str,
    n_threads: int = 6,
    max_tokens: int = 50,
    repeat_penalty: float = 1.20,
    n_batch: int = 6,
    top_k: int = 1,
    timeout: int = 90,  # Timeout in seconds
):
    """
    Loads the GPT4All model, downloading it if necessary.

    Args:
        model_path (str): Path to the model file.
        n_threads (int, optional): Number of threads to use. Defaults to 6.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 50.
        repeat_penalty (float, optional): Penalty for repeated tokens. Defaults to 1.20.
        n_batch (int, optional): Batch size for processing. Defaults to 6.
        top_k (int, optional): Number of top tokens to consider. Defaults to 1.
        timeout (int, optional): Timeout for downloading the model in seconds. Defaults to 90.

    Returns:
        GPT4All: Loaded GPT4All model.
    """
    if not os.path.isfile(model_path):
        # download model
        url = "https://huggingface.co/nomic-ai/gpt4all-falcon-ggml/resolve/main/ggml-model-gpt4all-falcon-q4_0.bin"
        # send a GET request to the URL to download the file. Stream since it's large
        response = requests.get(url, stream=True, timeout=timeout)

        # open the file in binary mode and write the contents of the response to it in chunks
        # This is a large file, so be prepared to wait.
        with open(model_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=10000)):
                if chunk:
                    f.write(chunk)
    else:
        print("model already exists in path.")

    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]
    # Verbose is required to pass to the callback manager
    llm = GPT4All(
        model=model_path,
        callbacks=callbacks,
        verbose=True,
        n_threads=n_threads,
        n_predict=max_tokens,
        repeat_penalty=repeat_penalty,
        n_batch=n_batch,
        top_k=top_k,
    )

    return llm


gptj = load_gpt4allj(
    model_path="./models/pickerbot/ggml-model-gpt4all-falcon-q4_0.bin",
    n_threads=15,
    max_tokens=100,
    repeat_penalty=1.20,
    n_batch=15,
    top_k=1,
)


@app.get("/ping")
async def ping():
    """
    Ping the server to check its status.

    Returns:
        dict: A response indicating the server's health status.
    """
    return {"message": "Server is Running"}


@app.post("/predict")
async def predict(payload: GenPayload):
    """
    Prediction Endpoint

    Args:
        payload (GenPayload): Prediction endpoint payload model.

    Returns:
        dict: PickerBot response based on the inference result.
    """
    bot = PickerBot(payload.data, model=gptj, safe_root=SAFE_BASE_DIR)
    bot.data_proc()
    bot.create_vectorstore()
    response = bot.inference(user_input=payload.user_input)
    return {"msg": "Sim Search and Inference Complete", "PickerBot Response": response}


if __name__ == "__main__":
    """Main entry point for the server.

    This block runs the FastAPI application using Uvicorn.
    """
    uvicorn.run("serve:app", host="127.0.0.1", port=5000, log_level="info")
