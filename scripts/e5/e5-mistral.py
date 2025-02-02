from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import time
import argparse
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
# Define the file path
file_path = os.path.join(current_directory, 'sentences.txt')

# Open the file and read lines into a list
with open(file_path, 'r') as file:
    lines = file.readlines()

# Strip newline characters from each line
lines = [line.strip() for line in lines]

def merge_every_N(strings, N):
    merged_strings = []
    for i in range(0, len(strings), N):
        # Merge three strings and add to the result list
        merged_strings.append(''.join(strings[i:i+N]))
    return merged_strings

lines = merge_every_N(lines, 16)

batch_size = 32 
input_texts = lines[:1000]
print(f"len: {len(input_texts)}")

parser = argparse.ArgumentParser("running LLM models")
parser.add_argument("gpu", type=int, help="the GPU to run LLM")
parser.add_argument("iteration", type=int, help='repeats')
args = parser.parse_args()
gpu_index = args.gpu
iteration = args.iteration
device = f"cuda:{gpu_index}"

tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct', torch_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained('intfloat/e5-mistral-7b-instruct', torch_dtype=torch.bfloat16).to(device)

max_length = 256
inputs = []
for i in range(0, 32 * 8, batch_size):
  batch_dict = tokenizer(input_texts[i:i+batch_size], max_length=max_length, truncation=True, return_tensors='pt').to(device)
  inputs.append(batch_dict)
  # for k in batch_dict:
  #   print(batch_dict[k].shape)

beg = time.time()
with torch.no_grad():
  for i in range(iteration):
    print(f"[GPU {gpu_index}] iteration {i}")
    outputs = []
    for batch_dict in inputs:
      outputs = [model(**batch_dict)]
      # print(outputs[-1][0].shape)

end = time.time()
print(f"[GPU {gpu_index} done], {end - beg}, {8 * 32 / (end - beg)}")