numactl --membind=1 python scripts/e5/e5-mistral.py 1 1  2>&1 | grep done &
numactl --membind=1 python scripts/e5/e5-mistral.py 2 1  2>&1 | grep done &
numactl --membind=1 python scripts/e5/e5-mistral.py 3 1  2>&1 | grep done &

wait $(jobs -p)