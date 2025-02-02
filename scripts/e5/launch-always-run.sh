numactl --membind=1 python scripts/e5/e5-mistral.py 1 500 &
numactl --membind=1 python scripts/e5/e5-mistral.py 2 500 &
numactl --membind=1 python scripts/e5/e5-mistral.py 3 500 &

jobs