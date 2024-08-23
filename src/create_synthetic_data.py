import multiprocessing
import random
import logging
import click
import uuid
import json
import os
from datetime import datetime
import numpy as np
from copy import deepcopy
from src.synthetic_data_generator import SyntethicDataGenerator, flatten_dict, create_combinations_limited, unflatten_dict, merge_two_params_comb, flatten_values
from src.synthetic_data_noiser import SyntheticDataNoiser


@ click.command(help="Script for running synthetic data generator.")
@ click.option("--output_path", type=str, required=True, help="Directory to save json files.")
@ click.option("--n_peaks_to_split", type=int, required=True, default=5, help="Number of closest neighbors to split peak between.")
@ click.option("--n_spectra", type=int, required=True, default=100, help="Number of spectra to generate in single scenario (single tag).")
@ click.option("--n_scenarios", type=int, required=True, default=5, help="Number of scenarios to generate (one scenario - one set of spectra).")
def run_script(output_path, n_peaks_to_split, n_spectra, n_scenarios):
    inputs = [(os.path.join(output_path, str(uuid.uuid4())), n_peaks_to_split, n_spectra, scenario_id)
              for scenario_id in range(1, n_scenarios+1)]
    with multiprocessing.Pool() as pool:
        pool.starmap(prepare_spectra, inputs)
    # [prepare_spectra(output_path, n_peaks_to_split, n_spectra, 1)
    #  for i in range(n_scenarios)]


def prepare_spectra(output_path, n_peaks_to_split, n_spectra, scenario_id):
    os.mkdir(output_path)
    os.mkdir(os.path.join(output_path, "spectra"))
    os.mkdir(os.path.join(output_path, "parameters"))
    logging.info(
        f"Preparing {scenario_id} scenario and saving to {output_path}")
    fixed_params = {
        "rps": random.choice(np.arange(5, 60, 0.3, dtype=np.float_)),
        "fres": 1,
        "fmax": 6000,
        "unbalance": {
            "fundamental": 1.0,
            "harcount": 1,
            "harslope": 0
        },
        "inner_race": {
            "fundamental": random.choice(np.arange(3.141, 14, 0.261, dtype=np.float_)),
            "harslope": 0.8,
            "sideband_ratio": random.choice(np.arange(0.1, 0.7, 0.3, dtype=np.float_)),
            "sideband_slope": 0.8,
            "sideband_freq": 1,
            "sideband_count": random.choice(np.arange(1, 5, 2, dtype=np.int_)),
            "pattern_noise": 0.2,
        }
    }
    fixed_params["unbalance"]["amplitude"] = random.choice(np.arange(
        fixed_params["rps"]/10, fixed_params["rps"]/2, 1.0, dtype=np.float_))
    for n_spectrum in range(n_spectra):
        spectrum_params = update_spectrum_params(
            deepcopy(fixed_params), n_spectrum, n_spectra)

        spectrum_params, synthetic_data = SyntethicDataGenerator(
            n_peaks_to_split).run(spectrum_params)

        synthetic_data_noiser = SyntheticDataNoiser(
            spectrum_params["fmax"], spectrum_params["fres"])
        synthetic_data["frequencies"], synthetic_data["amplitudes"] = synthetic_data_noiser.create_random_peaks(
            synthetic_data["frequencies"], synthetic_data["amplitudes"])
        synthetic_data["frequencies"], synthetic_data["amplitudes"] = synthetic_data_noiser.create_white_noise(
            synthetic_data["frequencies"], synthetic_data["amplitudes"])
        synthetic_data["frequencies"], synthetic_data["amplitudes"] = synthetic_data_noiser.sort(
            synthetic_data["frequencies"], synthetic_data["amplitudes"])
        synthetic_data_noiser.save(
            synthetic_data, spectrum_params, output_path, n_spectrum)


def update_spectrum_params(spectrum_params: dict, n_spectrum: int, n_spectra: int):
    progress = n_spectrum/n_spectra
    if random.expovariate(10) < progress:
        spectrum_params["inner_race"] = {
            **spectrum_params["inner_race"],
            **{
                "amplitude": min(3.0, max(0.2, random.gauss((progress**2)*2.5, 1))),
                "harcount": min(12, max(2, int(random.gauss((progress**2)*10, 4)))),
            }
        }
    else:
        del spectrum_params["inner_race"]
    return spectrum_params


if __name__ == '__main__':
    run_script()
