numactl --membind=1 python scripts/generation/llama3.py 1 32 1  2>&1 | grep done &
numactl --membind=1 python scripts/generation/llama3.py 2 32 1  2>&1 | grep done &
numactl --membind=1 python scripts/generation/llama3.py 3 32 1  2>&1 | grep done &

wait $(jobs -p)