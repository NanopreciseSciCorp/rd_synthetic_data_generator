from pathlib import Path
import os
import numpy as np


def get_project_root_dir():
    """
    This function is assumed to be placed in src/<project_name>/common.py.
    :return: Absolute path to project_root dir.
    """
    return Path(__file__).parent.parent.parent


def get_tests_data_dir():
    """
    This function is assumed to be placed in src/<project_name>/common.py.
    :return: Absolute path to testing datasets dir.
    """
    return os.path.join(get_project_root_dir(), "tests", "data")


def calculate_energy(amplitudes_peaks: list) -> float:
    """
    This function calculates the energy of a signal represented by its amplitudes.

    Parameters:
    amplitudes_peaks (list): A list of amplitudes of the signal peaks.

    Returns:
    float: The energy of the signal calculated as the square root of the sum of the squares of its amplitudes.
    """
    return np.round(np.sqrt(np.sum(np.square(amplitudes_peaks))), 2)
