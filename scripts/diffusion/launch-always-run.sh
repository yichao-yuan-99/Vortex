numactl --membind=1 python scripts/diffusion/sd-loop.py 1 &
numactl --membind=1 python scripts/diffusion/sd-loop.py 2 &
numactl --membind=1 python scripts/diffusion/sd-loop.py 3 &

wait $(jobs -p)