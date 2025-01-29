import subprocess
from mlx_lm import load, generate
import pandas


def start():
    # prompt format
    print("Start/n")
    intstructions_string = f"""I am insufferable know it all, ask away you little tyke!!"""
    print("prompt builder/n")
    prompt_builder = lambda comment: f'''{intstructions_string} \n{comment} \n'''

    model_path = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
    print("enter your question")
    input_prompt = input()
    prompt = prompt_builder({input_prompt})
    max_tokens = 256

    model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.2-4bit")
    response = generate(model, tokenizer, prompt=prompt, max_tokens = max_tokens,verbose=True)

    print("prompt end/n")

if __name__=="__main__":
    start()