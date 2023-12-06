from model import ToxicClassifier
from data_util import *
import argparse


def main(args):
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate stock autoencoder on a test data set for benchmarking")

    # shared arguments
    parser.add_argument("--run_id", type=str, default=0, help="id / name for models to be stored as")
    args = parser.parse_args()

    # run main
    main(args)
