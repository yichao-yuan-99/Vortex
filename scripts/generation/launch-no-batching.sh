numactl --membind=1 python scripts/generation/llama3.py 1 1 100 &
numactl --membind=1 python scripts/generation/llama3.py 2 1 100 &
numactl --membind=1 python scripts/generation/llama3.py 3 1 100 &

jobs