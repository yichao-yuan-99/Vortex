import argparse
import subprocess
import datetime
import os

# change this to change the expr setup
def exprArgs():
  gran = [i * 1000_000 for i in [20]] 
  tsize = [i * 1000_000_000 for i in [2, 4, 8, 12, 16]]
  repeat = 5
  # use the default path

  r = []
  for t in tsize:
    for g in gran:
      r.append((t, g, repeat))
  return r

# def exprArgs():
#   return [(16_000_000_000, 40_000_000, 1), (8_000_000_000, 20_000_000, 1)]

# format: <t>-<g>-<r>
def nameTuple(t, g, r):
  return f'{t}-{g}-{r}'

def toCommand(bin, t, g, r, file):
  cmd = [bin, '-t', t, '-g', g, '-r', r, '-f', file]
  cmd = [str(i) for i in cmd]
  return cmd
  


def getCommandAndFilename(bin, file):
  args = exprArgs()
  r = []
  for arg in args:
    r.append((nameTuple(*arg), toCommand(bin, *arg, file)))
  return r
    

def runCommand(outpath, cmd):
  print(f"running: {' '.join(cmd)}")
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

  stdout, _ = process.communicate()

  if process.returncode != 0:
    print(f"Error with return code {process.returncode}")
  else:
    with open(outpath, 'w') as f:
      f.write(stdout.decode())



def main():
  parser = argparse.ArgumentParser(description='run multiple dynio test and collect results.')

  parser.add_argument("bin", help="the path to dynio binary")
  parser.add_argument("file", help="the ground truth file used by dynio")
  parser.add_argument("outdir", help="path of the directory that stores the output")

  args = parser.parse_args()

  cmds = getCommandAndFilename(args.bin, args.file)


  outdir = args.outdir
  if not os.path.exists(outdir):
    os.makedirs(outdir)

  with open(os.path.join(outdir, 'metainfo'), 'w') as f:
    f.write(f"UTC timestamp: {datetime.datetime.utcnow().timestamp()}")

  for outfile, cmd in cmds:
    outpath = os.path.join(outdir, outfile)
    # print(outpath, cmd)
    runCommand(outpath, cmd)

# python scripts/dynio/run.py ./build/src/examples/dynio ./data/rand_uint32_4b.bin ./results/dynio/expr0
if __name__ == "__main__":
  main()
      