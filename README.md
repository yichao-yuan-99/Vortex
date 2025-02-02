# Vortex

source code for "Vortex: Overcoming Memory Capacity Limitations in
GPU-Accelerated Large-Scale Data Analytics"

If you find the repo useful, please cite the following paper
```
@inproceedings{yuan2024vortex,
  author    = {Yichao Yuan and
               Advait Iyer and
               Lin Ma and
               Nishil Talati 
               },
  title     = {{Vortex: Overcoming Memory Capacity Limitations in GPU-Accelerated Large-Scale Data Analytics}},
  booktitle = {{51th International Conference on Very Large Databases (VLDB 2025)}},
  pages     = {},
  publisher = {ACM},
  year      = {2024}
}
```

## build
To build Vortex
```
mkdir build && cd build
cmake ..
cmake --build .
```

Then the executables are placed in the subdirectory of `build/src`.

### Dependencies
this project is developed and tested on AMD GPUs (MI100).

All the packages for AMD GPU development should be installed to build this project.

The AI related code (for interference analysis) requires pytorch and huggingface installed

## Benchmarks
`build/src/example` contains `join`, `sort-alt`, and `dynio`, which benchmarks the performance of hash join, sort, and IO primitive of Vortex.

`build/src/crystal-ssb` is the SSB query implementation that integrates Vortex and Crystal.

`build/src/crystal-ssb-selectivity` is the SSB query implementation that integrates Vortex and Crystal.

`scripts` contains a couple of Deep Learning scripts that are used for interference analysis.
`scripts/diffusion` contains the scripts for Stable Diffusion 3, `scripts/e5` contains the scripts for text embedding generation, `scripts/prefill` contains the scripts for LLAMA3 prefill stage, and `scripts/generation` contains the scripts for LLAMA3 decode stage.
Those scripts are expected to run at either background or foreground to understand the interference on the target GPU and forwarding GPUs.

## Reproduce the numbers in the paper
Now we assume all the commands are carried out in the `build` directory.

### prepare input data

#### input data for sort and hash join
Both sort and hash join requires random numbers to populate the tables.

Frist create a directory `data`, then run `./src/tools/uniqueKeys` to generate random numbers.
All the required data can be generated using commands listed in the comments of `src/tools/uniqueKeys.cc`.
For example:
```
./src/tools/uniqueKeys -N 4000000000 -s 12138 -ps 12138 -o ../data/uniqueKeys_uint64_4b_12138_12138.bin
```

Besides, a file `rand_uint32_4b.bin` needs to be generated in `data`.
It should contains 4 billion unsigned integer in binary format.

Finally, run `./src/tools/permutation -m 1000000000 -s 12138 -o ../data/permutation_1b_12138.bin` and
`./src/tools/permutation -m 4000000000 -s 12138 -o ../data/permutation_4b_12138.bin` that will be used to randomly permute the tuples in the table.

#### input data for SSB queries
We reuse the data preparation step in [Crystal](https://github.com/anilshanbhag/crystal). 
After the data is populated, the address in `src/crystal-ssb*` need to be changed.
For example, in `src/crystal-ssb/q11.cc`, change the line 51 to
```
ssb::Dataset data("/path/to/crystal/test/ssb/data/s1000_columnar", ssb::Q1xDataConfig);
```




### Reproduce the numbers in the result sections
To reproduce the numbers in Figure 11, 
run `build/examples/dynio` with arguments like below:
```
./dynio  -t <totalTraffic> -g <granularity> -r <repeat>
```
To sweep through all the data points, we provide a script `python scripts/dynio/run.py`.
An example script to run this script is
```
python scripts/dynio/run.py ./build/src/examples/dynio ./data/rand_uint32_4b.bin ./results/dynio/expr0
```
It will generates all the results for different granularity and total transfer size in directory `results/dynio/expr0` directory. 

To reproduce the numbers in Figure 12 (about sort), run `./src/examples/sort-alt`.
This program will print the total amount of time as well as the time taken by each step.
A simple bash for loop can be used to repeat the experiment multiple times (we repeat 5 times).

To reproduce the numbers in Figure 13 (about hash join), run `./src/examples/join`.
Similar to `sort-alt`, it prints the total execution time as well as time for each step.

To reproduce the numbers in Figure 14 (about SSB benchmarks), run all the queries in `./src/crystal-ssb*`.

The queries in `./src/crystal-ssb` directory uses our enhanced IO primitive, but do not take advantage of late materialiation.
This correspond to the `IO redistribution (SDMA)` bars in Figure 14.
The queries in `./src/crystal-ssb-selectivity` takes advantage of late materialization, and the queries in `./src/crystal-ssb-zero-copy` only use zero-copy but does not use our enhanced IO primitive. 
They correspond to the `Vortex` bar and `zero-copy` bars respectively.

Finally, to reproduce Figure 15 (about interference), the AI inference scripts and the applications need to be run together.
Each AI inference application is shipped with a `launch-alwyas-run.sh` and `launch-short.sh`.
The former keeps running the AI applciation and the latter runs the application once.
To reproduce Figure 15 (a), run `dynio` program in the background with a very high `repeat`, then run the AI inference applications in the foreground by `source launch-short.sh`. 
To reproduce Figure 15 (b), run the AI inference applications in the background by `source launch-always-run.sh` and run the `sort-alt`, `join` and ssb quries in the foreground.


## Source Code Sructure
`src` contains all the source code.

`src/sched` contains all the code related to the IO scheduling.
This part of code correspond to Section 4 in the paper.

`src/execution` contains the code related to the IO-decoupled programming model, as well as the `ExKernel` definition.
This correspond to Section 5 of the paper.

`src/kernels` contains all the on-GPU kernels.
It wraps the kernel from AMD rocPRIM as well as contains our custom kernels for hash join.
This corresponds to Section 5 and 6.

`src/crystal-ssb*` contains code for SSB queries.
This corresponds to Section 6.3.

`src/hip` contains the wrapper code for HIP.
