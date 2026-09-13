import argparse
from src.Setup import setup

parser = argparse.ArgumentParser()

parser.add_argument('--GPU', type=int, help="INPUT THE GPU NUMBER YOU'D LIKE TO USE")
args = parser.parse_args()

print(f"P1 REVISITED.     GPU CHOSEN  {args.GPU}")


setup(args.GPU)