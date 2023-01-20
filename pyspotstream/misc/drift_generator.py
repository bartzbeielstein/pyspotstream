import numpy as np


def generate_drift(df, drift_list=[1.1, 10.0, 0.1, 1.1]):
    drift_array = np.array(drift_list)
    quotient, remain = np.divmod(df.shape[0], len(drift_array))

    quotient_array = np.repeat(drift_list, quotient)
    remain_array = np.repeat(drift_list[-1], remain)

    drift = np.append(quotient_array, remain_array)
    return drift
