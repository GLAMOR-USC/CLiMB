import json
import numpy as np
import sys

assert len(sys.argv)==2, "input .json file"

fn = sys.argv[1]

with open(fn, "r") as f:
    rdict = json.load(f)

print(fn)
for k in rdict.keys():
    scores = np.array(list(rdict[k].values()))
    assert len(scores) == 3, "missing experiments"
    print(f'{k}: {scores.mean():.1f} ±{scores.std():.1f}')
