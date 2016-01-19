from itertools import chain
import numpy as np
from scipy.stats import mannwhitneyu

from .stats import MetricCDF
from .log import logger


class History:
    """Container for metric history."""
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
            self.request_dtype = self.requests.dtype
        elif hasattr(requests, "to_records"):
            self.requests = requests.to_records(index=False)
            self.request_dtype = self.requests.dtype

        else:
            if request_dtype is None:
                raise ValueError("request_dtype must be specified!")
            self.request_dtype = request_dtype
            self.requests = np.array(requests,
                                     dtype=self.request_dtype).view(np.recarray)
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

    def get_generator(self, interval_size, time_reshuffling=False):
        """Generator to create sample histories from this one grouping intervals in interval_size."""
        if time_reshuffling:
            raise NotImplementedError("This is not implemented, but should not be important for A/a/b.")
        n_intervals = int(self.duration / interval_size)
        print(n_intervals)
        prec_interval_size = self.duration / n_intervals
        logger.info("Asked and possible interval sizes differ by {}".format(prec_interval_size / interval_size - 1))

        if n_intervals not in self.sample_generators:
            self.sample_generators[n_intervals] = self.make_sample_generator(prec_interval_size, n_intervals)
        return self.sample_generators[n_intervals]

    def make_sample_generator(self, prec_interval_size, n_intervals):
        logger.info("Interval size: {} => {} intervals".format(prec_interval_size, n_intervals))

        t = self.requests.time
        # List of request indices contained in each interval.
        intervals = []
        last_req_idx = 0
        num_requests = 0
        n_empty = 0
        for i in range(1, n_intervals + 1):
            i_included = np.searchsorted(t[last_req_idx:], self.beginning + i * prec_interval_size, "right")
            indices = list(range(last_req_idx, last_req_idx+i_included))
            intervals.append(indices)
            num_requests += len(indices)
            last_req_idx += i_included
            if len(indices) == 0:
                n_empty += 1
        logger.info("Generated intervals ({} of {} empty) to sample from.".format(n_empty, n_intervals))

        assert num_requests == len(t)
        assert len(intervals) == n_intervals

        while True:
            sampled_intervals = np.random.choice(n_intervals, n_intervals)
            sampled_intervals.sort()
            sampled_idx = chain.from_iterable(intervals[i] for i in sampled_intervals)
            request_gen = (self.requests[i] for i in sampled_idx)
            sampled_requests = np.fromiter(request_gen, dtype=self.request_dtype)

            yield History(sampled_requests, self.request_dtype, self.beginning, self.end)

    def samples(self, n_samples, interval_size, time_reshuffling=False):
        samples = []
        gen = self.get_generator(interval_size, time_reshuffling)
        logger.info("Starting sample generation")
        for i, h_i in zip(range(n_samples), gen):
            samples.append(h_i)
        logger.info("Finished sample generation")
        return samples

    def split(self, where):
        """Split the history into two sub-histories."""
        if not self.beginning < where < self.end:
            raise ValueError("where outside of history")

        i = np.searchsorted(self.requests.time, where, "right")
        h1 = History(self.requests[:i], self.request_dtype, self.beginning, where, True)
        h2 = History(self.requests[i:], self.request_dtype, where, self.end, True)
        return h1, h2

    def append(self, other):
        if other.beginning < self.end:
            raise ValueError("Histories overlap!")
        self.requests = np.concatenate((self.requests, other.requests)).view(np.recarray)
        self.end = other.end
        self.__min_distances.clear()

    @classmethod
    def generate(cls, beginning, end, request_rate, **generators):
        """Generate a request history.

        Parameters
        ----------
        beginning, end : float
            See `__init__`.
        request_rate : float
        kwargs :
            Definitions of random generators.
        """
        n_requests = int((end - beginning) * request_rate)
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
    def __init__(self, metric_estimator, conf_levels=(0.95, 0.99), cdf_type=None):
        """
        Parameters
        ----------
        metric_estimator : callable
            Called to calculate metric values from a `History` object. Can
            have `ranges` and `directions` attributes which are used to
            contruct CDFs (see e.g. :class:`~targetspace.stats.MetricCDF`).
        conf_levels : pair of floats
            The confidence levels of the borderes between acceptable,
            tolerating and frustrating regions of the target space.
        cdf_cls : class, optional
            The CDF implementation. See :module:`targetspace/stats.py` for
            implementations. Defaults to
            :class:`~targetspace.stats.MetricCDF`.
        """
        self.metric_estimator = metric_estimator
        self.conf_levels = conf_levels
        if cdf_type is None:
            self.cdf_type = MetricCDF
        else:
            self.cdf_type = cdf_type

        #: The bootstrap CDFs for the metrics.
        self._bs_metric_cdfs = None
        self.combine_max_idx = []

    def combine_evaluated_metrics(self, evaluated):
        raise NotImplementedError()

    def evaluate_observed_metrics(self, metrics):
        return np.fromiter((cdf(m) for (cdf, m) in zip(self._bs_metric_cdfs, metrics)),
                           dtype=float)

    def calibrate(self, history, n_bs_samples, bs_interval_size=None):
        """Calibrate the target space via dynamic base lining.

        Parameters
        ----------
        history : History
        n_bs_samples : int
            The number of bootstrap samples used for dynamic baselining.
        bs_interval_size : float or None, optional
            The time interval to generate bootstrap samples from `history`.
            Defaults to `None`, i.e. `0.5 * history.min_interval(1)`,
            asserting that each sample time interval contains at most 1 data
            point. See `History.min_interval()`.
        """
        logger.info("Calibrating: generating samples")
        if bs_interval_size is None:
            bs_interval_size = history.min_interval(1) * 0.5
        samples = history.samples(n_bs_samples, bs_interval_size)
        self._calibrate_with_samples(samples)

    def locate(self, history, n_bs_samples=0, bs_interval_size=None, cls=(0.9,)):
        """Locate history in the calibrated target space.

        Parameters
        ----------
        history : History
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
            samples = history.samples(n_bs_samples, bs_interval_size)
        else:
            samples = []
        return self._locate_with_samples(history, samples)

    def _locate_with_samples(self, history, samples, cls=(0.9,)):
        location = self._locate_history(history)
        locations = np.asarray([self._locate_history(s) for s in samples])
        if locations.size == 0:
            return location

        limits = []
        for cl in cls:
            alpha = 0.5 * (1.0 - cl)
            ll, ul = np.percentile(locations, [100 * alpha, 100 * (1-alpha)])
            limits.append((cl, ll, ul))
        return location, limits

    def _locate_history(self, history):
        if self._bs_metric_cdfs is None:
            raise RuntimeError("Target space not calibrated!")
        metrics = self.metric_estimator(history)
        evaluated = self.evaluate_observed_metrics(metrics)
        combined = self.combine_evaluated_metrics(evaluated)
        logger.debug("Located history  metrics: {}".format(metrics))
        logger.debug("Located history  evaluated: {}".format(evaluated))
        logger.debug("Located history  combined: {}".format(combined))

        return combined

    def _calibrate_with_samples(self, samples):
        if self._bs_metric_cdfs is not None:
            raise RuntimeError("Target space already calibrated!")

        n_samples = len(samples)

        metric_bs_data = tuple(np.empty(n_samples) for _ in range(len(self.metric_estimator.directions)))

        for i, s in enumerate(samples):
            metrics = self.metric_estimator(s)
            for j, m in enumerate(metrics):
                metric_bs_data[j][i] = m

        md_dir_range = zip(metric_bs_data, self.metric_estimator.directions, self.metric_estimator.ranges)
        self._bs_metric_cdfs = tuple(self.cdf_type(md, d, r) for md, d, r in md_dir_range)


class MWTargetSpace(TargetSpace):
    def calibrate(self, history, *_):
        self._references = self.metric_estimator(history)

    def _locate_history(self, history, *_):
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
