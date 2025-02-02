import os
import argparse

def splitOut(out):
  split = []
  for i, line in enumerate(out):
    if line[:3] == '@@@':
      split.append((i, line[3:].strip()))

  r = {}
  split.append((len(out), 'End'))
  for i in range(len(split) - 1):
    beg, name = split[i]
    end, _ = split[i + 1]
    r[name] = out[beg + 1:end]

  return r

def transformFiles(filePath):
  with open(filePath, 'r') as f:
    lines = f.readlines()
    s = splitOut(lines)
  
  if len(s) != 1:
    print(f"Error at {filePath}")
  else:
    result = s['Result']
    with open(filePath + '.csv', 'w') as f:
      for l in result:
        f.write(l)
  

# python ./scripts/dynio/tocsv.py ./results/dynio/expr0/ 
def main():
  parser = argparse.ArgumentParser(description='run multiple dynio test and collect results.')

  parser.add_argument("outdir", help="path of the directory that stores the output")

  args = parser.parse_args()

  files = os.listdir(args.outdir)
  files.remove('metainfo')
  filePaths = [os.path.join(args.outdir, f) for f in files]

  for f in filePaths:
    transformFiles(f)

if __name__ == "__main__":
  main()

