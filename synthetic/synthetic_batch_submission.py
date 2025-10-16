import os
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CPIC experiments.')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--start_seed', type=int, default=None)
    parser.add_argument('--end_seed', type=int, default=None)
    parser.add_argument('--method', type=str, default="CPIC")
    args = parser.parse_args()
    for i in tqdm(range(args.start_seed, args.end_seed)):
        print("seed: {}".format(i))
        if args.method == "CPIC":
            os.system('python code/synthetic_experiment.py --seed {} --config lorenz_stochastic_infonce_exploration --device {}'
                      .format(i, args.device))
        if args.method == "CPIC_obs":
            os.system(
                'python code/synthetic_experiment.py --seed {} --config lorenz_stochastic_infonce_obs_exploration --device {}'
                .format(i, args.device))
        if args.method == "CPIC_deterministic":
            os.system('python code/synthetic_experiment.py --seed {} --config lorenz_deterministic_infonce_exploration --device {}'
                      .format(i, args.device))
        if args.method == "CPIC_deterministic_obs":
            os.system('python code/synthetic_experiment.py --seed {} --config lorenz_deterministic_infonce_obs_exploration --device {}'
                      .format(i, args.device))
