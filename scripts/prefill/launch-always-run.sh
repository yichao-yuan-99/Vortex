numactl --membind=1 python scripts/prefill/llama3.py 1 500 &
numactl --membind=1 python scripts/prefill/llama3.py 2 500 &
numactl --membind=1 python scripts/prefill/llama3.py 3 500 &

jobs