#!/bin/python

import sys
from ctransformers import AutoModelForCausalLM
import colorama
from colorama import Fore
import html2text
import chime
from time import time
import yaml
import argparse

def makePrompt(query):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request

    ### Instruction: {query}

    ### Response:"""

config = ""
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser()
parser.add_argument('-model_path', default="./", help="The base location of the models")
parser.add_argument('-model', type=int, default=0, choices=range(0, len(config["models"])),  help="The model to use")
args = parser.parse_args()


chime.theme('mario')

model = config['models'][args.model]
llm = AutoModelForCausalLM.from_pretrained(args.model_path + model['location'], 
                                           max_new_tokens=model['max_new_tokens'],
                                           model_type=model['model_type'], 
                                           gpu_layers=model['gpu_layers'])

colorama.init()  # initialize colorama

# Initialize HTML to text converter
h = html2text.HTML2Text()
h.ignore_links = True

print(Fore.BLUE + "#########################################")
print(Fore.BLUE + "# Welcome to Easy AI!                   #")
print(Fore.BLUE + f'''# Model: {model['name']} #''')
print(Fore.BLUE + "#########################################")
print("")
print(Fore.CYAN + "----------------------")
print(Fore.CYAN + "Enter next prompt: ")
for line in sys.stdin:
    if 'Exit' == line.rstrip():
        break
    input_tokens = len(llm.tokenize(line))
    print(Fore.LIGHTGREEN_EX + f"== Generating response from [{input_tokens}] tokens ==")
    print(Fore.WHITE)

    # response = llm(makePrompt(line))
    # print(renderResponse(response))

    response = ""
    start_time = time()
    for chunk in llm(makePrompt(line), stream=True):
        response += chunk
        print(chunk, end='', flush=True)
    end_time = time()

    output_tokens = len(llm.tokenize(response))
    elapsed_time = end_time - start_time
    print("")
    print(Fore.LIGHTGREEN_EX + f"== Generated [{output_tokens}] tokens in [{round(elapsed_time, 1)}s] ==")

    chime.success()

    print(Fore.CYAN + "----------------------")
    print(Fore.CYAN + "Enter next prompt: ")


