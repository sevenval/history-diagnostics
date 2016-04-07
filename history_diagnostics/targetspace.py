from itertools import chain
import numpy as np
from scipy.stats import mannwhitneyu

from .stats import MetricCDF
from .log import logger


def group_indices(sarr, window_size=None, start=None):
    """Get list with indices of entries of sorted `sarr` grouped in windows.
    Parameters
    ----------
    sarr : ndarray
        Sorted array.
    window_size : float, optional
        The window size (optional).
    start : float, optional
        The start of the first window. If omitted, `sarr[0]` is used.
    Returns
    -------
    window_size : float
        The used window size.
    group : list of ndarrays
        The groups of windowed indices pointing to `sarr`.
    Examples
    --------
    >>> sarr = np.asarray([0, 0, 1, 1.5, 2, 3, 3.1])
    >>> group_indices(sarr, 1)
    1, [array([0, 1]), array([2, 3]), array([4]), array([5, 6])]
    """
    if start is None:
        start = sarr[0]
    if window_size is None:
        window_size = (sarr[-1] - start) / sarr.size

    window_idx = ((sarr - start) / window_size).astype(int)
    first_idx = np.unique(window_idx, return_index=True)[1]
    groups = []
    for i, j in enumerate(first_idx):
        if i < len(first_idx) - 1:
            end = first_idx[i+1]
        else:
            end = len(window_idx)
        groups.append(np.arange(j, end))

    return window_size, groups


class SampleGenerator:
    """Draw bootstrap samples from a request time series.

    Necessary inputs:

    sample : Sample
    windows : list
        List with indices contained in time-windows.
    n_empty : int
        The number of empty windows.

    A sample is drawn by

    * determine the number of non-emtpy windows to draw, `n_sample`, by
      drawing from a binomial distribution with number of trials given by
      the total number of windows (empty and full) scaled by the duration
      ratio,

      .. math::

          n_{trials} = r_{duration} * N_{windows}
                     = `duration_ratio * (len(windows) + self.n_empty)`

      and the probability to hit a non-empty window

      .. math::

          P = N_{full windows} / N_{windows}
            = `len(windows) /  (len(windows) + self.n_empty)`


    * draw `n_sample` windows from `self.windows`, sort and pull the request data from `self.sample.reqests`.
    """
    def __init__(self, sample, windows, n_empty):
        self.sample = sample
        self.windows = windows
        self.n_empty = n_empty

    def __call__(self, duration_ratio=1.0, traffic_ratio=1.0):
        n_windows = len(self.windows)
        n_output = int(duration_ratio * (n_windows + self.n_empty))

        while True:
            # Number of windows to draw.
            n_sample = np.random.binomial(n_output, n_windows / (n_windows + self.n_empty))
            sampled_windows = np.random.choice(n_windows, n_sample)
            sampled_idx = sorted(chain.from_iterable(self.windows[i] for i in sampled_windows))
            requests_gen = (self.sample.requests[i] for i in sampled_idx)
            sampled_requests = np.fromiter(requests_gen, dtype=self.sample.requests.dtype).view(np.recarray)

            s = Sample(sampled_requests,
                       beginning=self.sample.beginning,
                       end=self.sample.beginning + duration_ratio * self.sample.duration,
                       is_sorted=True,
                       traffic_boost=1/traffic_ratio)
            yield s


class Sample:
    """Container for a metric sample."""
    def __init__(self, requests, request_dtype=None, beginning=None, end=None, is_sorted=False, traffic_boost=1.0):
        """
        Parameters
        ----------
        requests : list of tuples, array, or DataFrame
            The request data. If passed as `recarray` or `DataFrame` it must
            have a "time" field.
        request_dtype : list of pairs, optional
            dtype specification of `requests`. Must be specified if
            `requests` is a list of tuples. Must have a numeric "time"
            field.
        beginning, end : float, optional
            The beginning and end of the history.
            If `None` (default) use the minimum and maximum of `requests.time`.
        is_sorted : bool, optional
            Pass `True` if `requests` is sorted by time.
        """
        if isinstance(requests, np.recarray):
            self.requests = requests
        elif hasattr(requests, "to_records"):
            self.requests = requests.to_records(index=False)
        else:
            if request_dtype is None:
                raise ValueError("request_dtype must be specified!")
            self.requests = np.array(requests,
                                     dtype=request_dtype).view(np.recarray)

        if "time" not in self.requests.dtype.names:
            raise ValueError("requests or request_dtype must have a 'time' field!")

        if not is_sorted:
            self.requests.sort(order="time")

        if beginning is None:
            beginning = self.requests.time[0]
        if end is None:
            end = self.requests.time[-1]

        self.beginning = beginning
        self.end = end
        self.traffic_boost = traffic_boost

        self.__min_distances = {}
        self.sample_generators = {}

    def __len__(self):
        return len(self.requests)

    def min_interval(self, n_contained):
        """The minimal interval size containing `n_contained` requests.

        Parameters
        ----------
        n_contained : int, > 0
        """
        if n_contained < 0:
            raise ValueError("n_contained must be >= 0")

        if n_contained not in self.__min_distances:
            t = self.requests.time
            dist = (t[n_contained:] - t[:-n_contained]).min()
            self.__min_distances[n_contained] = dist
        return self.__min_distances[n_contained]

    @property
    def duration(self):
        """The duration of the history."""
        return self.end - self.beginning

    def get_bootstrap_sample_generator(self, window_size, ratio):
        """Generator for bootstrap samples with data grouped in intervals of `interval_size`.

        Parameters
        ----------
        interval_size : float
            The targeted time window to group data into.

        Returns
        -------
        gen : generator
            Generator
        """
        n_windows = int(self.duration / window_size)
        prec_window_size = self.duration / n_windows
        logger.info("Asked and possible window sizes differ by {}".format(prec_window_size / window_size - 1))

        if n_windows not in self.sample_generators:
            windows, n_empty = self.get_windows(prec_window_size, n_windows)
            # win2, n_empty2 = self.get_windows_old(prec_window_size, n_windows)
            # assert len(windows) == len(win2), "Window-list lengths differ: {} vs. {} (prec_window_size: {})".format(len(windows), len(win2), prec_window_size)
            # assert n_empty == n_empty2, "Number of empty windows differ: {} vs. {} (prec_window_size: {})".format(n_empty, n_empty2, prec_window_size)
            self.sample_generators[n_windows] = SampleGenerator(self, windows, n_empty)
        return self.sample_generators[n_windows](ratio)

    def get_windows(self, prec_window_size, n_windows):
        groups = group_indices(self.requests.time, prec_window_size, self.beginning)[1]
        return groups, n_windows - len(groups)

    def get_windows_old(self, prec_window_size, n_windows):
        t = self.requests.time
        windows = []
        idx_last = 0
        num_requests = 0
        n_empty = 0
        for i_window in range(1, n_windows + 1):
            idx_included = np.searchsorted(t[idx_last:] , self.beginning + i_window * prec_window_size, "right")
            if idx_included == 0:
                n_empty += 1
            else:
                windows.append(np.arange(idx_last, idx_last + idx_included))
                idx_last += idx_included
                num_requests += len(windows[-1])
        assert num_requests == len(t)
        assert n_windows == len(windows) + n_empty

        return windows, n_empty

    def samples(self, n_samples, window_size, ratio):
        """Generate a list of bootstrap samples.

        Parameters
        ----------
        n_samples : int
            The number of samples to draw.
        window_size : float
            The target window size to group data into.
        ratio : float
            The ratio of requests, that the samples should contain.

        Returns
        -------
        samples : list of Sample objects
            List of bootstrap samples drawn from this sample.
        """
        gen = self.get_bootstrap_sample_generator(window_size, ratio)
        logger.info("Starting sample generation")
        samples = [s for _, s in zip(range(n_samples), gen)]
        logger.info("Finished sample generation")
        return samples

    def split(self, where):
        """Split the sample into two sub-samples.

        Parameters
        ----------
        where : float
            The time where the sample is split.

        Returns
        -------
        s1, s2 : Sample
            The new sub-samples.
        """
        if not self.beginning < where < self.end:
            raise ValueError("where outside of history")

        i = np.searchsorted(self.requests.time, where, "right")
        s1 = Sample(self.requests[:i], self.requests.dtype, self.beginning, where, True, self.traffic_boost)
        s2 = Sample(self.requests[i:], self.requests.dtype, where, self.end, True, self.traffic_boost)
        return s1, s2

    def merge(self, other):
        """Merge another sample in place.

        Parameters
        ----------
        other : Sample
        """
        if other.end <= self.beginning:
            self.requests = np.concatenate((other.requests, self.requests)).view(np.recarray)
            self.beginning = other.beginning
        elif self.end <= other.beginning:
            self.requests = np.concatenate((other.requests, self.requests)).view(np.recarray)
            self.end = other.end
        else:
            self.requests = np.concatenate((self.requests, other.requests)).view(np.recarray)
            self.requests.sort(order="time")
            self.beginning = min(self.beginning, other.beginning)
            self.end = max(self.end, other.end)
        self.__min_distances.clear()
        self.sample_generators.clear()

    @classmethod
    def generate(cls, beginning, end, data_rate, **generators):
        """Generate a sample.

        Parameters
        ----------
        beginning, end : float
            The time interval this sample spans.
            See `__init__`.
        data_rate : float
            The rate at which data occurs.
        kwargs :
            Definitions of random generators.
        """
        n_requests = int((end - beginning) * data_rate)
        times = beginning + (end - beginning) * np.random.rand(n_requests)
        times.sort()
        dtypes = [("time", float)]
        gen_data = [times]
        for name, (dtype, generator) in generators.items():
            dtypes.append((name, dtype))
            gen_data.append(generator(times))

        requests = list(zip(*gen_data))
        return cls(requests, dtypes, beginning, end, True)


def metric_directions(*dirs):
    """Annotate function with the direction (+/-1) of the returned metrics.

    `+1`, "upper" -> The larger, the better.
    `-1`, "lower" -> The smaller, the better.
    """
    def decorator(f):
        t = []
        for d in dirs:
            if d == +1:
                d = "upper"
            elif d == -1:
                d = "lower"
            elif d not in ("upper", "lower"):
                raise ValueError("Metric direction must be 'lower', 'upper', 1, or -1")
            t.append(d)
        f.directions = tuple(t)
        return f
    return decorator


def metric_ranges(*ranges):
    """Annotate function with ranges.
    """
    def decorator(f):
        t = []
        for r in ranges:
            if r is None:
                r = (None, None)
            elif len(r) != 2:
                raise ValueError("Range definition must be None or pair")
            t.append(tuple(r))
        f.ranges = tuple(t)
        return f
    return decorator


class TargetSpace:
    def __init__(self, metric_estimator, history, cdf_type=None):
        """
        Parameters
        ----------
        metric_estimator : callable
            Called to calculate metric values from a `History` object. Can
            have `ranges` and `directions` attributes which are used to
            contruct CDFs (see e.g. :class:`~targetspace.stats.MetricCDF`).
        cdf_cls : class, optional
            The CDF implementation. See :module:`targetspace/stats.py` for
            implementations. Defaults to
            :class:`~targetspace.stats.MetricCDF`.
        """
        self.metric_estimator = metric_estimator
        if cdf_type is None:
            self.cdf_type = MetricCDF
        else:
            self.cdf_type = cdf_type

        self.history = history

        #: The bootstrap CDFs for the metrics.
        self._bs_metric_cdfs = {}

    def combine_evaluated_metrics(self, evaluated):
        raise NotImplementedError()

    def calibrate(self, duration_ratio, n_bs_samples, bs_window_size, recalibrate=False):
        """Calibrate the target space via dynamic base lining.

        Parameters
        ----------
        duration_ratio : float
            The duration ratio to calibrate for.
        n_bs_samples : int
            The number of bootstrap samples used for dynamic baselining.
        bs_window_size : float
            The time window to generate bootstrap samples from `self.history`.
        recalibrate : bool, optional
        """
        if duration_ratio in self._bs_metric_cdfs:
            if not recalibrate:
                logger.warn("Target space for {!r} already calibrated".format(duration_ratio))
                return

        logger.info("Calibrating for {!r}: generating samples".format(duration_ratio))
        samples = self.history.samples(n_bs_samples, bs_window_size, duration_ratio)
        logger.info("Calibrating for {!r}: estimating metrics".format(duration_ratio))

        metric_bs_data = tuple(np.empty(n_bs_samples) for _ in range(len(self.metric_estimator.directions)))

        for i, s in enumerate(samples):
            metrics = self.metric_estimator(s)
            for j, m in enumerate(metrics):
                metric_bs_data[j][i] = m

        md_dir_range = zip(metric_bs_data, self.metric_estimator.directions, self.metric_estimator.ranges)
        self._bs_metric_cdfs[duration_ratio] = tuple(self.cdf_type(md, d, r) for md, d, r in md_dir_range)

    def get_cdfs(self, duration_ratio):
        if duration_ratio not in self._bs_metric_cdfs:
            raise RuntimeError("Target space not calibrated for duration ratio {!r}".format(duration_ratio))
        return self._bs_metric_cdfs[duration_ratio]

    def locate(self, sample, traffic_ratio=1, n_bs_samples=0, bs_interval_size=None, cls=(0.9,)):
        """Locate sample in the calibrated target space.

        Parameters
        ----------
        sample : Sample
        n_bs_samples : int, optional
            If `>0` use bootstraping to estimate confidence intervals.
        bs_interval_size : float or None, optional
            The time interval to generate bootstrap samples from `history`.
            Defaults to `None`, i.e. automatic interval-size selection.
            See `History.samples()`.
        cls : tuple of floats
            The confidence levels to calculate intervals for. Entries must
            be in `[0, 1]`.

        Returns
        -------
        location : float
            The location of `history` in the target space.
        intervals : list of float-triplets, optional
            If `n_bs_samples > 0`, the list of confidence levels and
            associated lower and upper interval limits as `(conf_level,
            lower_limit, upper_limit)` triplets.
        """
        if n_bs_samples:
            bs_samples = sample.samples(n_bs_samples, bs_interval_size, sample.duration / self.history.duration)
        else:
            bs_samples = []
        return self._locate_with_bs_samples(sample, bs_samples, cls, traffic_ratio)

    def _locate_with_bs_samples(self, sample, bs_samples, cls, traffic_ratio):
        duration_ratio = sample.duration / self.history.duration
        location = self._locate_sample(sample, duration_ratio, traffic_ratio)
        locations = np.asarray([self._locate_sample(s, duration_ratio, traffic_ratio) for s in bs_samples])
        if locations.size == 0:
            return location

        limits = []
        for cl in cls:
            alpha = 0.5 * (1.0 - cl)
            ll, ul = np.percentile(locations, [100 * alpha, 100 * (1-alpha)])
            limits.append((cl, ll, ul))
        return location, limits

    def _locate_sample(self, sample, duration_ratio, traffic_ratio):
        cdfs = self.get_cdfs(duration_ratio)

        metrics = self.metric_estimator(sample)
        evaluated = np.fromiter((cdf(m) for (cdf, m) in zip(cdfs, metrics)),
                                dtype=float)
        combined = self.combine_evaluated_metrics(evaluated)
        logger.debug("Located history  metrics: {}".format(metrics))
        logger.debug("Located history  evaluated: {}".format(evaluated))
        logger.debug("Located history  combined: {}".format(combined))

        return combined


class MWTargetSpace(TargetSpace):
    def calibrate(self, history, *_):
        self._references = self.metric_estimator(history)

    def _locate_sample(self, history, *_):
        metrics = self.metric_estimator(history)
        evaluated = []
        for ref, test in zip(self._references, metrics):
            evaluated.append(mannwhitneyu(ref, test).pvalue)
        return self.combine_evaluated_metrics(np.asarray(evaluated))


class MinMixin:
    def combine_evaluated_metrics(self, evaluated):
        return evaluated.min()

class OneMinusMaxMixin:
    def combine_evaluated_metrics(self, evaluated):
        return 1.0 - evaluated.max()

class GeometricMeanMixin:
    def combine_evaluated_metrics(self, evaluated):
        return np.power(np.prod(evaluated), 1.0 / len(evaluated))

class ArithmeticMeanCombineMixin:
    def combine_evaluated_metrics(self, evaluated):
        return evaluated.mean()


class PvalueReference:
    bs_sample_size = 500
    def __init__(self, history, metric_estimator):
        self.history = history
        self.metric_estimator = metric_estimator
        self.cdf_cache = {}

    def get_cdfs(self, duration_ratio):
        if duration_ratio not in self.cdf_cache:
            samples = self.history.samples(self.bs_sample_size, 1 / len(self.history), duration_ratio)
            metric_bs_data = tuple(np.empty(self.bs_sample_size) for _ in self.metric_estimator.directions)
            for i, s in enumerate(samples):
                for j, m in enumerate(self.metric_estimator(s)):
                    metric_bs_data[j][i] = m

            self.cdf_cache[duration_ratio] = tuple(
                MetricCDF(md, d, r) for md, d, r in zip(metric_bs_data,
                                                        self.metric_estimator.directions,
                                                        self.metric_estimator.ranges))
        return self.cdf_cache[duration_ratio]

    def p_values(self, sample):
        metrics = self.metric_estimator(sample)
        cdfs = self.get_cdfs(sample.duration / self.history.duration)
        return tuple(1-cdf(m) for cdf, m in zip(cdfs, metrics))

    def update(self, sample):
        self.history.merge(sample)
        self.cdf_cache.clear()
