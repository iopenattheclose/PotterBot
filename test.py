import subprocess
from mlx_lm import load, generate
import pandas


def start():
    # prompt format
    print("Start/n")
    intstructions_string = f"""ShawGPT, functioning as a virtual data science consultant on YouTube, communicates in clear, accessible language, escalating to technical depth upon request. \
    It reacts to feedback aptly and ends responses with its signature '–ShawGPT'. \
    ShawGPT will tailor the length of its responses to match the viewer's comment, providing concise acknowledgments to brief expressions of gratitude or feedback, \
    thus keeping the interaction natural and engaging.

    Please respond to the following comment.
    """
    print("prompt builder/n")
    prompt_builder = lambda comment: f'''<s>[INST] {intstructions_string} \n{comment} \n[/INST]\n'''

    model_path = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
    prompt = prompt_builder("name the 7 hocruxes in two lines?")
    max_tokens = 140

    model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.2-4bit")
    response = generate(model, tokenizer, prompt=prompt, max_tokens = max_tokens,verbose=True)

    print("prompt end/n")

if __name__=="__main__":
    start()