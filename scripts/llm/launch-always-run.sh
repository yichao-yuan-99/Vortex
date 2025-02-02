numactl --membind=1 python scripts/llm/llm.py 1 50000 bw-result/token meta-llama/Meta-Llama-3-8B-Instruct &
numactl --membind=1 python scripts/llm/llm.py 2 50000 bw-result/token meta-llama/Meta-Llama-3-8B-Instruct &
numactl --membind=1 python scripts/llm/llm.py 3 50000 bw-result/token meta-llama/Meta-Llama-3-8B-Instruct &

jobs