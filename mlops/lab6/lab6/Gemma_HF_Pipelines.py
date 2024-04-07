from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

from time import time

from lab6.__init__ import MAX_NEW_TOKENS, here, TOKEN, MODEL_NAME
import transformers.activations

# Set the cache directory to avoid downloading the model multiple times
(cache_dir := here / "cache").mkdir(exist_ok=True)


def main():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=cache_dir,
        token=TOKEN,
        hidden_activation="gelu_pytorch_tanh",

    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=cache_dir,
        token=TOKEN,

    )

    generator = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float32,
        device_map="auto",
    )

    user_input = "start"
    generator("you are a helpful and kind friend who enjoys conversations with others")

    while user_input != "stop":
        user_input = input("\nProvide input:\t\t ")

        start = time()

        if user_input != "stop":
            input_ids = tokenizer(user_input, return_tensors="pt")
            outputs = model.generate(**input_ids, max_new_tokens=MAX_NEW_TOKENS)
            inference_time = time() - start
            result = tokenizer.decode(outputs[0])
            print(f"Result:\t\t\t {result}")

        print(f"inference time: {inference_time:.3f} s\n")


if __name__ == "__main__":
    print("Using model:", MODEL_NAME)
    main()
