import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

import time
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

lines = merge_every_N(lines, 6)

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


# Each query must come with a one-sentence instruction that describes the task
task = 'Given a web search query, retrieve relevant passages that answer the query'
queries = [
    get_detailed_instruct(task, 'how much protein should a female eat'),
    get_detailed_instruct(task, 'summit define')
]
# No need to add instruction for retrieval documents
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
]
input_texts = queries + documents

batch_size = 32 
input_texts = lines[:1000]

gpu_index = 1

device = f'cuda:{gpu_index}'

tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct')

model.to(device)
def printFreeMem():
  print(torch.cuda.mem_get_info())


max_length = 256
# Tokenize the input texts
inputs = []
for i in range(0, 32 * 8, batch_size):
  batch_dict = tokenizer(input_texts[i:i+batch_size], max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(device)
  inputs.append(batch_dict)
  for k in batch_dict:
    print(batch_dict[k].shape)

with torch.no_grad():
  outputs = []
  beg = time.time()
  for batch_dict in inputs:
    outputs += [model(**batch_dict)]
    print(outputs[-1][0].shape)
  end = time.time()
  print(f"{gpu_index}, {end - beg}")
# embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# normalize embeddings
# embeddings = F.normalize(embeddings, p=2, dim=1)
# scores = (embeddings[:2] @ embeddings[2:].T) * 100
# print(scores.tolist())