import numpy as np


def temporal_statistics(values):

    values = np.asarray(values)

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "start": float(values[0]),
        "end": float(values[-1]),
        "absolute_change": float(
            values[-1] - values[0]
        ),
        "relative_change": float(
            (values[-1] - values[0])
            / max(abs(values[0]), 1e-8)
        ),
    }


def temporal_slope(values, times):

    values = np.asarray(values)
    times = np.asarray(times)

    slope = np.polyfit(
        times,
        values,
        1
    )[0]

    return float(slope)
