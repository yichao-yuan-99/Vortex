lanuch background llm tasks to run it together with DB tasks on GPU 0

To understand the interference of LLM on GPU IO, source `launch-always-run.sh` in the background, and run `dynio` in foreground.
After dynio finish, kill the LLM processes.

To understand the interference of GPU IO on LLM, run `./build/src/examples/dynio -t 8000000000 -g 20000000 -r 5000 -f ./data/rand_uint32_4b.bin` in the background (need to comment out the checking code).
Then, run `source launch-short.sh` five times to get the throughput on each GPU
After collection all the resources, kill the `dynio`.

Use fraction 10 to measure single direction traffic