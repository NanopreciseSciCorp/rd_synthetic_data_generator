import numpy as np
import random
import json
import os
import uuid
from datetime import datetime
from src.synthetic_data_generator import flatten_values


class SyntheticDataNoiser():
    def __init__(self) -> None:
        self.random_count = random.choice(
            np.arange(10, 50, 20, dtype=np.int_))
        self.random_max_amplitude = 1.0  # 2.0
        self.random_min_amplitude = 0.1  # 0.2
        # random.choice(np.arange(0.1, 1.0, 0.25, dtype=np.float_))
        self.noise = 0.1

    def update_data_description(self, description: str):
        updated_description = description + \
            "_".join((str(self.random_count), str(self.random_max_amplitude),
                     str(self.random_min_amplitude), str(self.noise)))
        return updated_description

    def create_random_peaks(self, frequencies: list, amplitudes: list, max_frequency: int):
        count = 0
        while count < self.random_count:
            random_frequency = random.randint(1, max_frequency)
            if random_frequency not in frequencies:
                frequencies.append(random_frequency)
                amplitudes.append(random.uniform(
                    self.random_min_amplitude, self.random_max_amplitude))
                count += 1
        return frequencies, amplitudes

    def create_white_noise(self, frequencies: list, amplitudes: list, max_frequency: int):
        white_noise_frequencies = [i for i in range(
            0, max_frequency+1) if i not in frequencies]
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
    def save(synthetic_data, output_path):
        synthetic_data["starting_timestamp"] = datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S')
        # This UUID is to avoid overwriting files with repeated names
        data_filename = {key: value for key, value in synthetic_data.items() if key not in [
            "frequencies", "amplitudes", "fault_labels"]}
        filename = flatten_values(data_filename) + "_" + str(uuid.uuid4())
        synthetic_data["tagId"] = filename
        filepath = os.path.join(output_path, filename + ".json")
        with open(filepath, 'w') as fp:
            json.dump(synthetic_data, fp)
