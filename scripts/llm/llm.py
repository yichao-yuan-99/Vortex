from huggingface_hub import login
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import transformers
import subprocess
import torch
import time
import argparse

def list_available_gpus():
    try:
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Available GPUs:")
            print(result.stdout)
        else:
            print("Error running rocm-smi:", result.stderr)
    except FileNotFoundError:
        print("rocm-smi not found. Please ensure ROCm is installed and rocm-smi is available in the PATH.")

def choose_gpu():
    gpu_index = int(input("Enter the GPU index you want to use (e.g., 0 for the first GPU): "))
    return gpu_index

parser = argparse.ArgumentParser("running LLM models")
parser.add_argument("gpu", type=int, help="the GPU to run LLM")
parser.add_argument("duration", type=int, help='the time LLM lasts')
parser.add_argument("tokenFile", type=str, help='the file contains huggingface token')
parser.add_argument("modelId", type=str, help="the LLM", default='meta-llama/Meta-Llama-3-8B-Instruct')

args = parser.parse_args()

gpu_index = args.gpu
duration = args.duration
model_id = args.modelId
with open(args.tokenFile, 'r') as f:
  token = f.read().strip()

# List available GPUs
# list_available_gpus()

# choose GPU
# gpu_index = choose_gpu()
# gpu_index = 1 
device = f"cuda:{gpu_index}" if torch.cuda.is_available() else f"cpu"

# Ask user for Hugging Face token
# token = input("Enter your Hugging Face token: ")
login(token)

# duration = int(input("Enter the duration (in seconds) for which you want the model to generate text: "))

# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

print("Loading Model for Use")
# device="cuda"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=device
    # device_map="auto"
)

messages = []
for i in range(duration * 60 * 30):
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


max_token = 1024

for message in messages:
    current_time = time.time()
    elapsed_time = (current_time - start_time)  # converting to mins
    if elapsed_time >= duration:
        break
    
    print("Generating message {}".format(message_cnt))
    outputs.append(pipeline(
        message,
        max_new_tokens=max_token,
        eos_token_id=terminators,
        do_sample=False,
    ))
    message_cnt += 1

tokenizer = AutoTokenizer.from_pretrained(model_id)
generated_text = outputs[-1][0]['generated_text'][-1]['content']
# print(f"Generated Text: {generated_text}")
tokens = tokenizer.tokenize(generated_text)
print(f"{gpu_index}, time: {elapsed_time}, token: {len(tokens)}, througput: {len(tokens) * message_cnt / elapsed_time} tokens/s")
