import numpy as np
from scipy import stats

from .targetspace import metric_directions, metric_ranges, Sample, TargetSpace, OneMinusMaxMixin, MinMixin, MWTargetSpace

@metric_directions("upper", "lower", "lower", "upper")
@metric_ranges((0, None), (0, None), (0, 1), (0, 1))
def frontend_metrics(history):
    n_requests = len(history.requests)

    traffic = n_requests / history.duration
    perf = np.median(history.requests.performance)

    error_rate = history.requests.faulty.sum() / n_requests
    event_rate = history.requests.event.sum() / n_requests

    return traffic, perf, error_rate, event_rate

def mw_frontend_metrics(history):
    traffic = np.diff(history.requests.time)
    perf = history.requests.performance
    errors = np.diff(history.requests.time[history.requests.faulty])
    events = np.diff(history.requests.time[history.requests.event])

    return traffic, perf, errors, events


class FrontendTargetSpace(OneMinusMaxMixin, TargetSpace):
    def __init__(self, history):
        super().__init__(frontend_metrics, history)


class FrontendMWTargetSpace(MinMixin, MWTargetSpace):
    def __init__(self):
        super().__init__(mw_frontend_metrics)


def example():
    # Generate a history between t=0 and 1, 1000 requests, a
    # positive-constrained normal performace distribution and 
    history = Sample.generate(0, 1, 10000,
                              performance=(float, lambda times: np.abs(stats.norm.rvs(loc=4.5, scale=0.87, size=times.size))),
                              faulty=(bool, lambda times: stats.binom.rvs(1, 0.05, size=times.size)),
                              event=(bool, lambda times: stats.binom.rvs(1, 0.15, size=times.size)))
    h_T, h_C = history.split(0.8)
    ratio = h_C.duration / h_T.duration
    # print(ratio)

    ts = FrontendTargetSpace(h_T)
    # print("Calibrating")
    ts.calibrate(ratio, 100, 1e-4)
    # print(ts.metric_estimator(h_T))
    # print(ts.metric_estimator(h_C))
    # print(ts.evaluate_observed_metrics(ts.metric_estimator(h_C)))
    # print("Locating")
    # print(ts.locate(h_C, 100, 1e-4))
    print(ts.locate(h_C))
    # return ts, h_T, h_C
