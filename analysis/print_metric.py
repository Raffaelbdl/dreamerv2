import argparse
import pickle as pkl


parser = argparse.ArgumentParser()
parser.add_argument("--output", "-o", type=str, default="dreamer.out")
parser.add_argument("--metric", "-m", type=str)
args = parser.parse_args()

with open(args.output, 'rb') as f:
    data = pkl.load(f)
    config = data['config']
    values = data['evaluation_metrics'][args.metric]

eval_frequency = config['eval_frequency']

for i, v in enumerate(values):
    print(str((i + 1) * eval_frequency) + " :" + str(v))
