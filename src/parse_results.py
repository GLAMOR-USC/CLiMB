import json
import os
import pdb
import numpy as np
import sys
import glob

assert len(sys.argv)==2, "input task name"
task_name = sys.argv[1]

fns = glob.glob(f"/data/experiments/MCL/vision_only/{task_name}_*")
for fn in fns:
    with open(fn, "r") as f:
        rdict = json.load(f)

    print("-"*30)
    name = os.path.basename(fn).split('_')[:-1]
    print(f'# {name}:')

    for k in rdict.keys():
        scores = np.array(list(rdict[k].values()))
#        print(rdict[k])
#        print(scores)
        test_scores, dev_scores = scores[:, 0], scores[:, 1]
        print(f'{k}-test: {test_scores.mean():.1f} ±{test_scores.std():.1f}')
#        print(f'{k}-dev: {dev_scores.mean():.1f} ±{dev_scores.std():.1f}')
#        print(f'{k}-test: {test_scores.mean():.1f}')
        if np.abs(test_scores.mean() - dev_scores.mean()) > 5:
            print(rdict[k])
            pdb.set_trace()
    print("-"*30)
