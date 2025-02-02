import torch
from diffusers import StableDiffusion3Pipeline
import time
import argparse

parser = argparse.ArgumentParser("running LLM models")
parser.add_argument("gpu", type=int, help="the GPU to run LLM")
args = parser.parse_args()

gpu_index = args.gpu

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
# pipe.enable_model_cpu_offload()
device = f'cuda:{gpu_index}' 
pipe.to(device)

beg = time.time()
for i in range(300):
  image = pipe(
      prompt="a photo of a cat holding a sign that says hello world",
      negative_prompt="",
      num_inference_steps=28,
      height=1024,
      width=1024,
      guidance_scale=7.0
  ).images[0]
end = time.time()

print(f"{gpu_index}, {end - beg}")

image.save(f"sd3_hello_world-{gpu_index}.png")