
import multiprocessing
import click
import numpy as np
from tqdm import tqdm
from src.synthetic_data_generator import SyntethicDataGenerator, flatten_dict, create_combinations_limited, unflatten_dict, merge_two_params_comb
from src.synthetic_data_noiser import SyntheticDataNoiser


all_params = {
    "rps": list(np.arange(2, 60, 1, dtype=np.int_)),
    "fres": [1],
    "fmax": [6000],
    # GROUP 1
    "inner_race": {
        "fundamental": list(np.arange(3.141, 14, 0.261, dtype=np.float_)),
        "amplitude": list(np.arange(0.2, 3.0, 0.6, dtype=np.float_)),
        "harcount": list(np.arange(5, 12, 3, dtype=np.int_)),
        "harslope": [0.8],
        "sideband_ratio": list(np.arange(0.1, 0.7, 0.3, dtype=np.float_)),
        "sideband_slope": [0.8],
        "sideband_freq": [1],
        "sideband_count": list(np.arange(1, 5, 2, dtype=np.int_)),
        "pattern_noise": [0.2],
    },
    "unbalance": {
        "fundamental": [1.0],
        "amplitude": [7],
        "harcount": [1],
        "harslope": [0]
    },
    # GROUP 2
    # "gearmesh": {
    #     "fundamental": list(np.arange(11, 51, 4, dtype=np.float_)),
    #     "amplitude": list(np.arange(0.1, 3.0, 0.6, dtype=np.float_)),
    #     "harcount": list(np.arange(3, 5, 1, dtype=np.int_)),
    #     "harslope": [0.8],
    #     "sideband_ratio": list(np.arange(0.1, 0.7, 0.3, dtype=np.float_)),
    #     "sideband_slope": [0.8],
    #     "sideband_freq": [1],
    #     "sideband_count": list(np.arange(1, 3, 1, dtype=np.int_)),
    #     "pattern_noise": [0.2]
    # },
    # "looseness": {
    #     "fundamental": [1.0],
    #     "amplitude": [7],
    #     "harcount": list(np.arange(3, 12, 2, dtype=np.int_)),
    #     "harslope": [0.8]
    # },
    # GROUP 3
    # "vpf": {
    #     "fundamental": list(np.arange(1, 7, 1, dtype=np.float_)),
    #     "amplitude": list(np.arange(1.0, 7.0, 1.0, dtype=np.float_)),
    #     "harcount": list(np.arange(3, 5, 1, dtype=np.int_)),
    #     "harslope": [0.8],
    # },
    # "misalignement": {
    #     "fundamental": [1.0],
    #     "amplitude": [7],
    #     "harcount": [2],
    #     "harslope":  list(np.arange(0.1, 1.0, 0.2, dtype=np.float_)),
    # }
}


def init_worker(output_path_val, n_peaks_to_split_val):
    global glob_output_path
    glob_output_path = output_path_val
    global glob_n_peaks_to_split
    glob_n_peaks_to_split = n_peaks_to_split_val


def single_run(recipe_params):
    synthetic_data = SyntethicDataGenerator(
        glob_n_peaks_to_split).run(recipe_params)

    synthetic_data_noiser = SyntheticDataNoiser()
    synthetic_data["frequencies"], synthetic_data["amplitudes"] = synthetic_data_noiser.create_random_peaks(
        synthetic_data["frequencies"], synthetic_data["amplitudes"], synthetic_data["fmax"])
    synthetic_data["frequencies"], synthetic_data["amplitudes"] = synthetic_data_noiser.create_white_noise(
        synthetic_data["frequencies"], synthetic_data["amplitudes"], synthetic_data["fmax"])
    synthetic_data["frequencies"], synthetic_data["amplitudes"] = synthetic_data_noiser.sort(
        synthetic_data["frequencies"], synthetic_data["amplitudes"])
    synthetic_data_noiser.save(synthetic_data, glob_output_path)


@ click.command(help="Script for running synthetic data generator.")
@ click.option("--output_path", type=str, required=True, help="Directory to save json files.")
@ click.option("--n_peaks_to_split", type=int, required=True, default=5, help="Number of closest neighbors to split peak between.")
@ click.option("--n_scenarios", type=int, required=True, default=100, help="Number of scenarios to generate.")
def run_script(output_path, n_peaks_to_split, n_scenarios):

    flat_params = flatten_dict(all_params)
    flat_combinations = create_combinations_limited(flat_params, n_scenarios)

    all_combinations = [unflatten_dict(combination)
                        for combination in flat_combinations]

    all_combinations = merge_two_params_comb(all_combinations, n_scenarios)

    with multiprocessing.Pool(initializer=init_worker, initargs=(output_path, n_peaks_to_split,)) as pool:
        list(tqdm(pool.imap(single_run, all_combinations), total=len(all_combinations)))


if __name__ == '__main__':
    run_script()
