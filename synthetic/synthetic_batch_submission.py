import os
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    # run from script dir so config and experiment paths work
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='CPIC experiments.')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--start_seed', type=int, default=None)
    parser.add_argument('--end_seed', type=int, default=None)
    parser.add_argument('--method', type=str, default="CPIC")
    args = parser.parse_args()

    device_arg = "" if args.device is None else " --device {}".format(args.device)

    for i in tqdm(range(args.start_seed, args.end_seed)):
        print("seed: {}".format(i))
        if args.method == "CPIC":
            os.system('python synthetic_experiment.py --seed {} --config lorenz_stochastic_infonce_exploration{}'
                      .format(i, device_arg))
        elif args.method == "CPIC_obs":
            os.system('python synthetic_experiment.py --seed {} --config lorenz_stochastic_infonce_obs_exploration{}'
                      .format(i, device_arg))
        elif args.method == "CPIC_deterministic":
            os.system('python synthetic_experiment.py --seed {} --config lorenz_deterministic_infonce_exploration{}'
                      .format(i, device_arg))
        elif args.method == "CPIC_deterministic_obs":
            os.system('python synthetic_experiment.py --seed {} --config lorenz_deterministic_infonce_obs_exploration{}'
                      .format(i, device_arg))
        elif args.method == "CPIC_conv_s":
            os.system('python synthetic_experiment.py --seed {} --config lorenz_stochastic_infonce_obs_exploration_conv_s{}'
                      .format(i, device_arg))
        elif args.method == "CPIC_conv_st":
            os.system('python synthetic_experiment.py --seed {} --config lorenz_stochastic_infonce_obs_exploration_conv_st{}'
                      .format(i, device_arg))
        elif args.method == "CPIC_conv_t":
            os.system('python synthetic_experiment.py --seed {} --config lorenz_stochastic_infonce_obs_exploration_conv_t{}'
                      .format(i, device_arg))
        else:
            raise ValueError("Unknown method: {}".format(args.method))
