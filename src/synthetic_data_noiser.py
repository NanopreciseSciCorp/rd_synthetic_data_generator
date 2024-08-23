import numpy as np
import random
import json
import os
from datetime import datetime
from src.synthetic_data_generator import flatten_dict


class SyntheticDataNoiser():
    def __init__(self, fmax, fres) -> None:
        self.random_count = random.choice(
            np.arange(10, 50, 20, dtype=np.int_))
        self.random_max_amplitude = 1.0  # 2.0
        self.random_min_amplitude = 0.1  # 0.2
        # random.choice(np.arange(0.1, 1.0, 0.25, dtype=np.float_))
        self.noise = 0.1
        self.freq_grid = [float(round(x*fres, 3))
                          for x in range(int(fmax/fres))]

    def create_random_peaks(self, frequencies: list, amplitudes: list):
        count = 0
        while count < self.random_count:
            random_frequency = random.choice(self.freq_grid)
            if random_frequency not in frequencies:
                frequencies.append(random_frequency)
                amplitudes.append(random.uniform(
                    self.random_min_amplitude, self.random_max_amplitude))
                count += 1
        return frequencies, amplitudes

    def create_white_noise(self, frequencies: list, amplitudes: list):
        white_noise_frequencies = [
            i for i in self.freq_grid if i not in frequencies]
        white_noise_amplitudes = [random.uniform(
            0, self.noise) for _ in range(len(white_noise_frequencies))]

        frequencies += white_noise_frequencies
        amplitudes += white_noise_amplitudes
        return frequencies, amplitudes

    def sort(self, frequencies: list, amplitudes: list):
        sorted_pairs = sorted(zip(frequencies, amplitudes))
        frequencies = [item[0] for item in sorted_pairs]
        amplitudes = [item[1] for item in sorted_pairs]
        return frequencies, amplitudes

    @staticmethod
    def save(synthetic_data, spectrum_params, output_path, n_spectrum):
        synthetic_data["timestamp"] = datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S')
        flat_dict = flatten_dict(spectrum_params)
        tag_id = '_'.join(str(val) for key, val in flat_dict.items(
        ) if 'amplitude' not in key and 'harcount' not in key)
        synthetic_data["tag_id"] = tag_id
        spectrum_filepath = os.path.join(output_path, "spectra", tag_id +
                                         '_' + str(n_spectrum) + ".json")
        with open(spectrum_filepath, 'w') as fp:
            json.dump(synthetic_data, fp)

        params_filepath = os.path.join(output_path, "parameters", tag_id +
                                       '_' + str(n_spectrum) + ".json")
        with open(params_filepath, 'w') as fp:
            json.dump(spectrum_params, fp)
