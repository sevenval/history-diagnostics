from itertools import chain, islice
import numpy as np
from scipy.stats import mannwhitneyu

from .stats import MetricCDF
from .log import logger


class SampleGenerator:
    def __init__(self, sample, windows):
        self.sample = sample
        self.windows = windows

    def __call__(self, ratio=1.0):
        n_windows = len(self.windows)
        n_output = int(ratio * n_windows)

        while True:
            sampled_windows = np.random.choice(n_windows, n_output)
            sampled_windows.sort()
            sampled_idx = chain.from_iterable(self.windows[i] for i in sampled_windows)
            request_gen = (self.sample.requests[i] for i in sampled_idx)
            sampled_requests = np.fromiter(request_gen, dtype=self.sample.requests.dtype).view(np.recarray)

            yield Sample(sampled_requests, beginning=self.sample.beginning, end=self.sample.end)

class Sample:
    """Container for a metric sample."""
    def __init__(self, requests, request_dtype=None, beginning=None, end=None, is_sorted=False):
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

        self.__min_distances = {}
        self.sample_generators = {}

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
            self.sample_generators[n_windows] = self.make_bs_sample_generator(prec_window_size, n_windows)
        return self.sample_generators[n_windows](ratio)

    def make_bs_sample_generator(self, prec_window_size, n_windows):
        logger.info("window size: {} => {} windows".format(prec_window_size, n_windows))

        t = self.requests.time
        # List of request indices contained in each window.
        windows = []
        last_req_idx = 0
        num_requests = 0
        n_empty = 0
        for i in range(1, n_windows + 1):
            i_included = np.searchsorted(t[last_req_idx:], self.beginning + i * prec_window_size, "right")
            indices = list(range(last_req_idx, last_req_idx+i_included))
            windows.append(indices)
            num_requests += len(indices)
            last_req_idx += i_included
            if len(indices) == 0:
                n_empty += 1
        logger.info("Generated windows ({} of {} empty) to sample from.".format(n_empty, n_windows))

        assert num_requests == len(t)
        assert len(windows) == n_windows

        return SampleGenerator(self, windows)

        # while True:
            # sampled_windows = np.random.choice(n_windows, n_windows)
            # sampled_windows.sort()
            # sampled_idx = chain.from_iterable(windows[i] for i in sampled_windows)
            # request_gen = (self.requests[i] for i in sampled_idx)
            # sampled_requests = np.fromiter(request_gen, dtype=self.requests.dtype)

            # yield Sample(sampled_requests, self.requests.dtype, self.beginning, self.end)

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
        s1 = Sample(self.requests[:i], self.requests.dtype, self.beginning, where, True)
        s2 = Sample(self.requests[i:], self.requests.dtype, where, self.end, True)
        return s1, s2

    def merge(self, other):
        """Merge another sample in place.

        Parameters
        ----------
        other : Sample
        """
        self.requests = np.concatenate((self.requests, other.requests)).view(np.recarray)
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
        self.combine_max_idx = []

    def combine_evaluated_metrics(self, evaluated):
        raise NotImplementedError()

    def calibrate(self, ratio, n_bs_samples, bs_window_size, recalibrate=False):
        """Calibrate the target space via dynamic base lining.

        Parameters
        ----------
        ratio : float
            The duration ratio to calibrate for.
        n_bs_samples : int
            The number of bootstrap samples used for dynamic baselining.
        bs_window_size : float
            The time window to generate bootstrap samples from `self.history`.
        recalibrate : bool, optional
        """
        if ratio in self._bs_metric_cdfs:
            if not recalibrate:
                logger.warn("Target space for {!r} already calibrated".format(ratio))
                return

        logger.info("Calibrating for {!r}: generating samples".format(ratio))
        samples = self.history.samples(n_bs_samples, bs_window_size, ratio)
        logger.info("Calibrating for {!r}: estimating metrics".format(ratio))

        metric_bs_data = tuple(np.empty(n_bs_samples) for _ in range(len(self.metric_estimator.directions)))

        for i, s in enumerate(samples):
            metrics = self.metric_estimator(s)
            for j, m in enumerate(metrics):
                metric_bs_data[j][i] = m

        md_dir_range = zip(metric_bs_data, self.metric_estimator.directions, self.metric_estimator.ranges)
        self._bs_metric_cdfs[ratio] = tuple(self.cdf_type(md, d, r) for md, d, r in md_dir_range)

    def get_cdfs(self, ratio):
        if ratio not in self._bs_metric_cdfs:
            raise RuntimeError("Target space not calibrated for duration ratio {!r}".format(ratio))
        return self._bs_metric_cdfs[ratio]

    def locate(self, sample, n_bs_samples=0, bs_interval_size=None, cls=(0.9,)):
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
        self.combine_max_idx = []
        if n_bs_samples:
            bs_samples = sample.samples(n_bs_samples, bs_interval_size, sample.duration / self.history.duration)
        else:
            bs_samples = []
        return self._locate_with_bs_samples(sample, bs_samples)

    def _locate_with_bs_samples(self, sample, bs_samples, cls=(0.9,)):
        ratio = sample.duration / self.history.duration
        location = self._locate_sample(sample, ratio)
        locations = np.asarray([self._locate_sample(s) for s in bs_samples])
        if locations.size == 0:
            return location

        limits = []
        for cl in cls:
            alpha = 0.5 * (1.0 - cl)
            ll, ul = np.percentile(locations, [100 * alpha, 100 * (1-alpha)])
            limits.append((cl, ll, ul))
        return location, limits

    def _locate_sample(self, sample, ratio):
        cdfs = self.get_cdfs(ratio)

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
        self.combine_max_idx.append(evaluated.argmax())
        return 1.0 - evaluated.max()

class GeometricMeanMixin:
    def combine_evaluated_metrics(self, evaluated):
        return np.power(np.prod(evaluated), 1.0 / len(evaluated))


class ArithmeticMeanCombineMixin:
    def combine_evaluated_metrics(self, evaluated):
        return evaluated.mean()
