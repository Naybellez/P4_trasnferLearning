import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--GPU', type=int, help="INPUT THE GPU NUMBER YOU'D LIKE TO USE")
args = parser.parse_args()

print(f"P1 REVISITED.     GPU CHOSEN  {args.GPU}")

from Dir_learning.P4_transferLearning.P1_Returns.src.Setup import setup
setup(args.GPU)