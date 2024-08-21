import numpy as np
from src.synthetic_data_generator import SyntethicDataGenerator


# Test round_to_resolution
def test_round_to_resolution():
    harmonics_frequencies = [5.2, 14]
    harmonics_amplitudes = [10, 6.66]
    frequency_resolution = 1
    maximum_frequency = 20
    ex = SyntethicDataGenerator(5)
    ex.freq_grid = [float(round(x*frequency_resolution, 3))
                    for x in range(int(maximum_frequency/frequency_resolution))]

    new_harmonics_frequencies, new_harmonics_amplitudes = ex.round_to_resolution(
        harmonics_frequencies, harmonics_amplitudes, frequency_resolution)

    # Check results
    assert np.array_equal(new_harmonics_frequencies, [5, 6, 4, 7, 3, 14.0])
    assert np.allclose(
        new_harmonics_amplitudes,
        [4.707329706546124, 1.783744405059624, 1.3429775183282653,
            1.1372589471768424, 1.0286894228891446, 6.66]
    )
