import itertools
import random
import math
import numpy as np
import logging
from copy import deepcopy
from src.common import calculate_energy

AMPLITUDES_TO_GENERATE = ["unbalance", "looseness", "misalignement"]
# Generated amplitudes below this threshold are ignored (MaxAmp * AMPLITUDE_THRESHOLD)
AMPLITUDE_THRESHOLD = 0.05


class SyntethicDataGenerator:
    """
    The SyntethicDataGenerator class is designed to generate synthetic data based on different types of faults.
    The synthetic data is created by generating harmonics and sidebands for each fault type.

    Attributes:
    ----------
    output_path : str
        The path where the output data will be saved.
    n_peaks_to_split : int
        The number of peaks to split.
    synthetic_faults_function : dict
        A dictionary mapping fault types to the corresponding function used to generate the synthetic data.

    Methods:
    -------
    run(params: dict):
        Executes the synthetic data creation process based on the provided parameters.
    _create_only_harmonics(params: dict, fmax: int, rps: float, fres: float) -> tuple[list[float], list[float], dict]:
        Creates harmonics for a given label.
    _create_harmonics_and_sidebands(params: dict, fmax: int, rps: float, fres: float) -> tuple[list[float], list[float], dict]:
        Creates harmonics and sidebands for a given label.
    _generate_harmonics(fundamental: float, n_harmonics: int, harmonics_ratio: float, max_amplitude: float, frequency_range: float, rps: float) -> tuple[list[float], list[float]]:
        Generates a list of harmonic frequencies and amplitudes.
    _generate_sidebands(harmonics_frequencies: list[float], harmonics_amplitudes: list[float], no_of_sidebands_el: int, sideband_step: float, sideband_ratio: float, sideband_slope: float) -> tuple[list[float], list[float]]:
        Generates a list of sideband frequencies and amplitudes.
    _randomize_amplitudes(harmonics_amplitudes: list[float], noise_factor: float) -> list[float]:
        Randomizes the amplitudes of the harmonics.
    _calculate_energy_statistics(amplitudes_peaks: list[float], amplitudes_peaks_sidebands: list[float]) -> tuple[float, float]:
        Calculates the energy statistics of the harmonics and sidebands.
    round_to_resolution(harmonics_frequencies: list[float], harmonics_amplitudes: list[float], frequency_resolution: float) -> tuple[list[float], list[float]]:
        Rounds the frequencies and amplitudes to a certain resolution.
    _merge_patterns(harm_freq: dict, harm_ampl: dict) -> tuple[list, list, list]:
        Merges the frequency and amplitude patterns of the different faults.
    _format(harm_freq: list[float], harm_ampl: list[float], fault_labels: list[list[str]], params: dict, precision: int = 3) -> tuple[list[float], list[float], list[list[str]], dict]:
        Formats the frequencies, amplitudes, fault labels, and parameters with a certain precision.
    _save(params: dict, harm_freq: list[float], harm_ampl: list[float], fault_labels: list[list[str]]):
        Saves the synthetic data to a file.
    _find_amplitude(frequencies: list[float], look_frequency: float, amplitudes: list[float]) -> float:
        Finds the amplitude corresponding to a certain frequency.
    format_nested_dict(params: dict, precision: int) -> dict:
        Formats a nested dictionary with a certain precision.
    """

    def __init__(self, n_peaks_to_split: int) -> None:
        self.n_peaks_to_split = n_peaks_to_split
        self.synthetic_faults_function = {
            "unbalance": self._create_only_harmonics,
            "looseness": self._create_only_harmonics,
            "misalignement": self._create_only_harmonics,
            "vpf": self._create_only_harmonics,
            "inner_race": self._create_harmonics_and_sidebands,
            "outer_race": self._create_only_harmonics,
            "gearmesh": self._create_harmonics_and_sidebands
        }

    def run(self, params):
        harm_freq_per_fault = {}
        harm_ampl_per_fault = {}
        fault_types = []

        for fault_name, fault_params in params.items():
            if fault_name in self.synthetic_faults_function:
                harm_freq_per_fault[fault_name], harm_ampl_per_fault[fault_name], updated_fault_params = self.synthetic_faults_function[fault_name](
                    fault_params, params["fmax"], params["rps"], params["fres"])
                params[fault_name].update(updated_fault_params)
                fault_types.append(fault_name)

        harm_freq_unified, harm_ampl_unified, fault_labels = self._merge_patterns(
            harm_freq_per_fault, harm_ampl_per_fault)
        harm_freq_unified, harm_ampl_unified, fault_labels, params = self._format(
            harm_freq_unified, harm_ampl_unified, fault_labels, params, precision=3)
        params["fault_type"] = ",".join(fault_types)
        final_params = {**params, **
                        {"frequencies": harm_freq_unified, "amplitudes": harm_ampl_unified, "fault_labels": fault_labels}}
        return final_params

    def _create_only_harmonics(self, params: dict, fmax: int, rps: float, fres: float) -> tuple[list[float], list[float], dict]:
        """
        Create harmonics for a given label. Returns a list of frequencies, a list of amplitudes, and the updated input parameters.
        """

        harm_freq, harm_ampl = self._generate_harmonics(params["fundamental"], params["harcount"],
                                                        params["harslope"], params["amplitude"], fmax, rps)
        params["her"], params["ser"] = self._calculate_energy_statistics(
            harm_ampl, [])
        harm_freq, harm_ampl = self.round_to_resolution(
            harm_freq, harm_ampl, fres, fmax)
        return harm_freq, harm_ampl, params

    def _create_harmonics_and_sidebands(self, params: dict, fmax: int, rps: float, fres: float) -> tuple[list[float], list[float], dict]:
        """
        Create harmonics and sidebands for a given label. Returns a list of frequencies, a list of amplitudes, and the updated input parameters.
        """

        harm_freq, harm_ampl = self._generate_harmonics(params["fundamental"], params["harcount"],
                                                        params["harslope"], params["amplitude"], fmax, rps)
        peaks_sideband_freq, peaks_sideband_ampl = self._generate_sidebands(
            harm_freq, harm_ampl, params["sideband_count"], params["sideband_freq"]*rps, params["sideband_ratio"], params["sideband_slope"])
        params["her"], params["ser"] = self._calculate_energy_statistics(
            harm_ampl, peaks_sideband_ampl)
        rand_ampl = self._randomize_amplitudes(
            peaks_sideband_ampl, params["pattern_noise"])
        harm_freq, harm_ampl = self.round_to_resolution(
            peaks_sideband_freq, rand_ampl, fres, fmax)
        return harm_freq, harm_ampl, params

    def _generate_harmonics(self, fundamental, n_harmonics, harmonics_ratio, max_amplitude, frequency_range, rps):
        harmonics_frequencies = []
        harmonics_amplitudes = []
        for i in range(n_harmonics):
            freq = fundamental * (i + 1) * rps
            if freq > frequency_range:
                break
            if i == 0:
                amp = max_amplitude
            else:
                amp = harmonics_amplitudes[-1] * harmonics_ratio
            harmonics_frequencies.append(freq)
            harmonics_amplitudes.append(amp)

        return harmonics_frequencies, harmonics_amplitudes

    def _generate_sidebands(self, harmonics_frequencies, harmonics_amplitudes, no_of_sidebands_el, sideband_step, sideband_ratio, sideband_slope):
        new_harmonics_frequencies = []
        new_harmonics_amplitudes = []
        for har_index in range(len(harmonics_frequencies)):
            harmonics_frequency = harmonics_frequencies[har_index]
            new_harmonics_frequencies.append(harmonics_frequency)
            harmonics_amplitude = harmonics_amplitudes[har_index]
            new_harmonics_amplitudes.append(harmonics_amplitude)
            for i in range(no_of_sidebands_el):
                new_harmonics_frequencies.append(
                    harmonics_frequency-(i+1)*sideband_step)
                new_harmonics_frequencies.append(
                    harmonics_frequency+(i+1)*sideband_step)
                new_harmonics_amplitudes.append(
                    harmonics_amplitude*sideband_ratio*(sideband_slope**i))

                new_harmonics_amplitudes.append(
                    harmonics_amplitude*sideband_ratio*(sideband_slope**i))

        return new_harmonics_frequencies, new_harmonics_amplitudes

    def _randomize_amplitudes(self, harmonics_amplitudes, noise_factor):
        noise = [1+noise_factor*random.uniform(-1, 1)
                 for _ in range(len(harmonics_amplitudes))]
        harmonics_amplitudes = [a * b for a,
                                b in zip(harmonics_amplitudes, noise)]
        return harmonics_amplitudes

    def _calculate_energy_statistics(self, amplitudes_peaks, amplitudes_peaks_sidebands):
        harmonic_energy = calculate_energy(amplitudes_peaks)

        if len(amplitudes_peaks_sidebands) > 0:
            sideband_energy = calculate_energy(amplitudes_peaks_sidebands)
        else:
            sideband_energy = 0
        return harmonic_energy, sideband_energy

    def round_to_resolution(self, harmonics_frequencies, harmonics_amplitudes, frequency_resolution, maximum_frequency):
        def calculate_neighbour_weight(neighbour, harmonics_frequency, close_far_border=1.5):
            close_neighbours_dist_power = 0.7
            far_neighbours_dist_power = 0.5
            if abs(neighbour-harmonics_frequency) < close_far_border:
                return 1/(abs(neighbour-harmonics_frequency)**close_neighbours_dist_power)
            else:
                return 1/(abs(neighbour-harmonics_frequency)**far_neighbours_dist_power)

        # TODO - generalize for different frequency_resolution
        if frequency_resolution != 1:
            logging.error(
                f"Working only for resolution equal to 1. This resolution is {frequency_resolution}.")
            raise ValueError

        new_harmonics_frequencies = []
        new_harmonics_amplitudes = []
        for har_index in range(len(harmonics_frequencies)):
            harmonics_frequency = harmonics_frequencies[har_index]
            harmonics_amplitude = harmonics_amplitudes[har_index]
            decimal_harmonic, integer_harmonic = math.modf(
                harmonics_frequency)
            if decimal_harmonic == 0:
                if integer_harmonic in new_harmonics_frequencies:
                    index = new_harmonics_frequencies.index(integer_harmonic)
                    new_harmonics_amplitudes[index] += harmonics_amplitude
                else:
                    new_harmonics_frequencies.append(integer_harmonic)
                    new_harmonics_amplitudes.append(harmonics_amplitude)
                continue

            sorted_frequencies = sorted(range(maximum_frequency), key=lambda elem: abs(
                elem - harmonics_frequency))
            n_closest_neigbours = sorted_frequencies[:self.n_peaks_to_split]
            neighbours_weights = [calculate_neighbour_weight(
                neighbour, harmonics_frequency) for neighbour in n_closest_neigbours]
            neighbours_shares = [
                neighbour_weight/sum(neighbours_weights) for neighbour_weight in neighbours_weights]

            for i in range(self.n_peaks_to_split):
                integer_harmonic = n_closest_neigbours[i]
                integer_share = neighbours_shares[i]
                if integer_harmonic in new_harmonics_frequencies:
                    index = new_harmonics_frequencies.index(integer_harmonic)
                    new_harmonics_amplitudes[index] += harmonics_amplitude * \
                        integer_share
                else:
                    new_harmonics_frequencies.append(integer_harmonic)
                    new_harmonics_amplitudes.append(
                        harmonics_amplitude*integer_share)
        return new_harmonics_frequencies, new_harmonics_amplitudes

    def _merge_patterns(self, harm_freq: dict, harm_ampl: dict) -> tuple[list, list, list]:
        # Get a set of unique frequencies
        harmonic_frequencies = list(
            set([freq for fault_freq in harm_freq.values() for freq in fault_freq]))

        # Obtain the corresponding fault labels and amplitudes for each frequency
        harmonic_amplitudes = []
        fault_labels = []

        for freq in harmonic_frequencies:
            # Fault types associated to a sigle frequency
            freq_label = []

            # Search for the exact `freq` among the faults
            for fault_name in harm_freq.keys():
                if freq in harm_freq[fault_name]:
                    freq_label.append(fault_name)

            # Append the found faults to the general list
            fault_labels.append(freq_label)

            # Get the corresponding amplitudes for each frequency
            found_ampl = [self._find_amplitude(
                harm_freq[fault_name], freq, harm_ampl[fault_name]) for fault_name in harm_freq.keys()]
            harmonic_amplitudes.append(max(found_ampl))

        return harmonic_frequencies, harmonic_amplitudes, fault_labels

    # -> tuple[list[float], list[float], list[list[str]], dict]:
    def _format(self, harm_freq, harm_ampl, fault_labels, params, precision=3):
        # round
        harm_freq = [float(round(x, precision)) for x in harm_freq]
        harm_ampl = [float(round(x, precision)) for x in harm_ampl]
        # remove zeros amplitudes
        zeros_idx = [i for i, x in enumerate(
            harm_ampl) if x < max(harm_ampl)*AMPLITUDE_THRESHOLD]
        harm_freq = [i for j, i in enumerate(harm_freq) if j not in zeros_idx]
        harm_ampl = [i for j, i in enumerate(harm_ampl) if j not in zeros_idx]
        fault_labels = [i for j, i in enumerate(
            fault_labels) if j not in zeros_idx]

        params = self.format_nested_dict(params, precision)
        return harm_freq, harm_ampl, fault_labels, params

    def _find_amplitude(self, frequencies, look_frequency, amplitudes):
        try:
            return amplitudes[frequencies.index(look_frequency)]
        except ValueError:
            return 0

    def format_nested_dict(self, params, precision):
        for k, v in params.items():
            if isinstance(v, dict):
                self.format_nested_dict(v, precision)
            elif isinstance(v, np.int64):
                params[k] = int(v)
            elif isinstance(v, float):
                params[k] = float(round(v, precision))
        return params


def unflatten_dict(dictionary, sep="&"):
    def insert(dictionary, keys, value):
        for key in keys[:-1]:
            dictionary = dictionary.setdefault(key, {})
        dictionary[keys[-1]] = value

    result_dict = {}
    for key, value in dictionary.items():
        parts = key.split(sep)
        insert(result_dict, parts, value)

    return result_dict


def merge_two_params_comb(params_comb: list, max_combinations: int):
    fault_types = [k for k in params_comb[0].keys()
                   if k in AMPLITUDES_TO_GENERATE]

    result_combinations = []
    for single_params_comb in params_comb:
        ampl_combinations = {}
        for fault_type in fault_types:
            ampl_combinations.setdefault(fault_type, {})
            ampl_combinations[fault_type]["amplitude"] = list(np.arange(
                single_params_comb["rps"]/10, single_params_comb["rps"]/2, 1.0, dtype=np.float_))
        amp_flat_params = flatten_dict(ampl_combinations)
        amp_flat_combinations = create_combinations(amp_flat_params)
        amp_all_combinations = [unflatten_dict(combination)
                                for combination in amp_flat_combinations]
        for amp_dict in amp_all_combinations:
            new_comb = deepcopy(single_params_comb)
            for fault_type, amp in amp_dict.items():
                new_comb[fault_type] = {**new_comb[fault_type], **amp}
            result_combinations.append(new_comb)
    result_combinations = random.sample(result_combinations, max_combinations)
    logging.info(
        f"{len(result_combinations)} of combinations with amplitudes created.")
    return result_combinations


def create_combinations_limited(d, max_combinations: int = 1_000_000):
    keys = d.keys()
    params_values = [i if isinstance(i, list) else [i] for i in d.values()]

    combinations = [list(comb) for comb in itertools.product(
        params_values[0], params_values[1])]
    for param_values in params_values[2:]:
        combinations = [combination + [val]
                        for val in param_values for combination in combinations]
        print(f"combinations len: {len(combinations)}")
        if len(combinations) > max_combinations:
            combinations = random.sample(combinations, max_combinations)
            print(f"downsampled -- combinations len: {len(combinations)}")

    combinations = [dict(zip(keys, combination))
                    for combination in combinations]

    return combinations


def create_combinations(d):
    keys = d.keys()
    values = [i if isinstance(i, list) else [i] for i in d.values()]
    combinations = [dict(zip(keys, combination))
                    for combination in itertools.product(*values)]
    return combinations


def create_combinations_limited(d, max_combinations: int = 5_000):
    keys = d.keys()
    params_values = [i if isinstance(i, list) else [i] for i in d.values()]

    combinations = [list(comb) for comb in itertools.product(
        params_values[0], params_values[1])]
    for param_values in params_values[2:]:
        combinations = [combination + [val]
                        for val in param_values for combination in combinations]
        if len(combinations) > max_combinations:
            combinations = random.sample(combinations, max_combinations)
    combinations = [dict(zip(keys, combination))
                    for combination in combinations]
    logging.info(f"{len(combinations)} of basic combinations created.")

    return combinations


def flatten_values(nested_dict):
    flat_dict = flatten_dict(nested_dict)
    flat_values = '_'.join(str(val) for val in flat_dict.values())
    return flat_values


def flatten_dict(d, parent_key='', sep='&'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
