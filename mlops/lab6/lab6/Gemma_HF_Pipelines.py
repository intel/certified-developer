from os import getenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import time

from lab6.__init__ import here

# Set the cache directory to avoid downloading the model multiple times
(cache_dir := here / "cache").mkdir(exist_ok=True)


def main():
    model = AutoModelForCausalLM.from_pretrained(getenv("MODEL"), cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        getenv("MODEL"), cache_dir=cache_dir, use_fast=True
    )

    generator = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    user_input = "start"

    while user_input != "stop":
        user_input = input(f"Provide Input to {model} parameter Falcon (not tuned): ")

        start = time.time()

        if user_input != "stop":
            sequences = generator(
                f""" {user_input}""",
                max_length=getenv("MAX_LENGTH"),
                do_sample=False,
                top_k=getenv("TOP_K"),
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
            )

        inference_time = time.time() - start

        for seq in sequences:
            print(f"Result: {seq['generated_text']}")

        print(f"Total Inference Time: {inference_time} seconds")


if __name__ == "__main__":
    print("Using model:", getenv("MODEL"))
    main()
