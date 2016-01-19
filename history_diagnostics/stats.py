import numpy as np

from .log import logger

class MetricCDF:
    """Empirical CDF for the probalility to find worse values for a metric."""
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

        if i == 0 and x < self.sample[0]:
            if self.value_range[0] is not None:
                logger.debug("Interpolating lower-tail cdf")
                i = (x - self.value_range[0]) / (self.sample[0] - self.value_range[0])
            # else:
                # i = 1
        elif i == len(self.sample) and x > self.sample[-1]:
            if self.value_range[1] is not None:
                logger.debug("Interpolating upper-tail cdf")
                i = i - 1 + (self.value_range[1] - x) / (self.value_range[1] - self.sample[-1])
            # else:
                # i = len(self.sample) - 1
        return i


class InterpolatingMetricCDF(MetricCDF):
    """Empirical interpolating CDF for the probalility to find worse values for a metric."""
    def index(self, x):
        i = super().index(x)
        if isinstance(i, int) and x != self.sample[i]:
            logger.debug("Interpolating CDF")
            i += (x - self.sample[i-1]) / (self.sample[i] - self.sample[i-1])
        return i
