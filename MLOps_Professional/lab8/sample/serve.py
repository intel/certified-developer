import uvicorn
import os
import requests

from tqdm import tqdm
from langchain.llms import GPT4All
from fastapi import FastAPI, Depends
from model import GenPayload
from SupportBot import PickerBot
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

app = FastAPI()

def load_gpt4allj(model_path: str = './models/pickerbot/ggml-gpt4all-j-v1.3-groovy.bin', n_threads: int=6, 
                  max_tokens: int=50, repeat_penalty: float = 1.20):
        
    if not os.path.isfile(model_path): 

        # download the commertial gpt4all-j model
        url = "https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin"
        # send a GET request to the URL to download the file. Stream since it's large
        response = requests.get(url, stream=True)

        # open the file in binary mode and write the contents of the response to it in chunks
        # This is a large file, so be prepared to wait.
        with open(model_path, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                if chunk:
                    f.write(chunk)
    else:
        print('model already exists in path.')
    
    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]
    # Verbose is required to pass to the callback manager
    llm = GPT4All(model=model_path, callbacks=callbacks, verbose=True, backend='gptj', n_threads=n_threads, n_predict=max_tokens, repeat_penalty=repeat_penalty)
    
    return llm


@app.get("/ping")
async def ping():
    """Ping server to determine status

    Returns
    -------
    API response
        response from server on health status
    """
    return {"message":"Server is Running"}


@app.post("/predict")
async def predict(payload:GenPayload, llm = Depends(load_gpt4allj)):
    bot = PickerBot(payload.data, model = llm)
    bot.data_proc()
    bot.create_vectorstore()
    response = bot.inference(user_input = payload.user_input)
    return {"msg": "Sim Search and Inference Complete", "SupportBot Response": response}

if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=5000, log_level="info")