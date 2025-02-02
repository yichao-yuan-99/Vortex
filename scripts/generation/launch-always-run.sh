numactl --membind=1 python scripts/generation/llama3.py 1 32 100 &
numactl --membind=1 python scripts/generation/llama3.py 2 32 100 &
numactl --membind=1 python scripts/generation/llama3.py 3 32 100 &

jobs