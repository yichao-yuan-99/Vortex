numactl --membind=1 python scripts/diffusion/sd.py 1 &
numactl --membind=1 python scripts/diffusion/sd.py 2 &
numactl --membind=1 python scripts/diffusion/sd.py 3 &

wait $(jobs -p)