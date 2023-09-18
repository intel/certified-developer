from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import argparse
import time


def main(FLAGS):
    
    model = f"tiiuae/falcon-{FLAGS.falcon_version}"
    
    
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    
        
    generator = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    user_input = "start"

    while user_input != "stop":

        user_input = input(f"Provide Input to {model} parameter Falcon (not tuned): ")
        
        start = time.time()

        if user_input != "stop":
            sequences = generator( 
            f""" {user_input}""",
            max_length=FLAGS.max_length,
            do_sample=False,
            top_k=FLAGS.top_k,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,)

        inference_time = time.time() - start
        
        for seq in sequences:
         print(f"Result: {seq['generated_text']}")
         
        print(f'Total Inference Time: {inference_time} seconds')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-fv',
                        '--falcon_version',
                        type=str,
                        default="7b",
                        help="select 7b or 40b version of falcon")
    parser.add_argument('-ml',
                        '--max_length',
                        type=int,
                        default="25",
                        help="used to control the maximum length of the generated text in text generation tasks")
    parser.add_argument('-tk',
                        '--top_k',
                        type=int,
                        default="5",
                        help="specifies the number of highest probability tokens to consider at each step")
    
    FLAGS = parser.parse_args()
    main(FLAGS)