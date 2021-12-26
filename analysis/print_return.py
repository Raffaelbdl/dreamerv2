import argparse
import pickle as pkl
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--output", "-o", type=str, default="dreamer.out")
parser.add_argument("--window", "-w", type=int, default=10)
args = parser.parse_args()

with open(args.output, 'rb') as f:
    data = pkl.load(f)
    config = data['config']
    returns = data['returns']
    termination_times = data['termination_times']

returns = np.convolve(returns,np.ones(args.window)/args.window, mode='valid')
termination_times = np.convolve(termination_times,np.ones(args.window)/args.window, mode='valid')

for t,r in zip(termination_times, returns):
    print(str(int(t))+": "+str(r))
