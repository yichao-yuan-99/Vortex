numactl --membind=1 python scripts/prefill/llama3.py 1 1  2>&1 | grep done  &
numactl --membind=1 python scripts/prefill/llama3.py 2 1  2>&1 | grep done  &
numactl --membind=1 python scripts/prefill/llama3.py 3 1  2>&1 | grep done  &

wait $(jobs -p)