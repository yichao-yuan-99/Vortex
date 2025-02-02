from huggingface_hub import login
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import transformers
import subprocess
import torch
import time
import argparse

parser = argparse.ArgumentParser("running LLM models")
parser.add_argument("gpu", type=int, help="the GPU to run LLM")
parser.add_argument("batchSize", type=int, help='text generation batch size')
parser.add_argument("iteration", type=int, help='the time LLM runs')

args = parser.parse_args()

gpu_index = args.gpu
iteration = args.iteration
batchSize = args.batchSize
model_id = 'meta-llama/Meta-Llama-3-8B-Instruct' 

device = f"cuda:{gpu_index}"


print("Loading Model for Use")

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=device
)
pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id

messages = []
for i in range(batchSize):
    messages.append([
        {"role": "system", "content": "You are a friendly assistant who complies with instructions"},
        {"role": "user", "content": "Tell me the longest story you know, keep talking"},
    ])

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = []
message_cnt = 0
start_time = time.time()


max_token = 512

beg = time.time()
for i in range(iteration):
  print(f"[GPU {gpu_index}] Generating message {i}")
  output = pipeline(
      messages,
      max_new_tokens=max_token,
      eos_token_id=terminators,
      num_return_sequences=1,
      do_sample=False,
      batch_size=batchSize
  )
end = time.time()

print(f"[GPU {gpu_index} done] time: {end - beg}, throughput: {batchSize * max_token * iteration / (end - beg)} tokens/s")