import numpy as np

from .log import logger

class MetricCDF:
    """Empirical CDF of a metric."""
    def __init__(self, sample, direction="upper", value_range=(None, None)):
        """
        Parameters
        ----------
        sample : numpy.ndarray
        direction : {"lower", "upper"}, optional
            The direction in which the metric improves. Defaults to "upper".
        value_range : pair, optional
            The value range of the metric used to interpolate the CDF for
            values outside the sampled range. Use `None` to prevent
            interpolating (default).
        """
        if direction not in ("lower", "upper"):
            raise ValueError("direction must be 'lower' or 'upper'")

        self.sample = sample.copy()
        self.sample.sort()
        self.direction = direction
        self.value_range = value_range


    def __call__(self, x):
        """Calculate probabiltiy to find a metric value equal to or better than `x`.

        Parameters
        ----------
        x : float
            The observed metric.

        Return
        ------
        prob : float
            The probability to find a metric better than `x`.
        """
        i = self.index(x)
        lower_tail = i / len(self.sample)
        if self.direction == "lower":
            return lower_tail
        else:
            return 1 - lower_tail

    def index(self, x):
        if self.direction == "upper":
            i = np.searchsorted(self.sample, x, side="right")
        else:
            i = np.searchsorted(self.sample, x, side="left")

        # if i == 0 and x < self.sample[0] and self.value_range[0] is not None:
            # logger.debug("Interpolating lower-tail cdf")
            # i = (x - self.value_range[0]) / (self.sample[0] - self.value_range[0])
        # elif i == len(self.sample) and x > self.sample[-1] and self.value_range[1] is not None:
            # logger.debug("Interpolating upper-tail cdf")
            # i = i - 1 + (self.value_range[1] - x) / (self.value_range[1] - self.sample[-1])
        return i


class InterpolatingMetricCDF(MetricCDF):
    """Empirical interpolating CDF for the probalility to find worse values for a metric."""
    def index(self, x):
        i = super().index(x)
        if isinstance(i, int) and x != self.sample[i]:
            logger.debug("Interpolating CDF")
            i += (x - self.sample[i-1]) / (self.sample[i] - self.sample[i-1])
        return i


def percentile_weighted(a, q, weights=None, sorted_data=False):
    if weights is None:
        return np.percentile(a, q)

    a = np.asanyarray(a)
    weights = np.asanyarray(weights)

    if a.shape != weights.shape:
        raise ValueError("shape mismatch: a: {!r}  weights: {!r}".format(a.shape, weights.shape))

    assert np.all(weights >= 0), "All weights must be >=0"

    if not sorted_data:
        idx = a.argsort()
        a = a[idx]
        weights = weights[idx]

    cum_weights = weights.cumsum()

    return np.interp(cum_weights[0] + q * 0.01 * (cum_weights[-1] - cum_weights[0]), cum_weights, a)
