import sys
import numpy as np

from history_diagnostics.stats import MetricCDF

n_bs = 200
n_test = 1000

def metric(s):
    return s.sum() / len(s)

def gen_sample(rate, sample_size):
    return np.random.binomial(1, rate, size=sample_size)

def bootstrap_gen(data, n_sample):
    for _ in range(n_sample):
        yield np.random.choice(data, len(data))


if __name__ == "__main__":
    toks = sys.argv[1].split(',')
    rate_ref = float(toks[0])
    rate_test = float(toks[1])
    sample_size = int(toks[2])
    opath = 'results/{:e}_{:e}_{:d}.txt'.format(rate_ref, rate_test, sample_size)
    print(opath)
    with open(opath, 'w') as of:
        for i in range(n_test):
            ref_sample = gen_sample(rate_ref, sample_size)
            cdf = MetricCDF(np.fromiter((metric(bs) for bs in bootstrap_gen(ref_sample, n_bs)),
                                        dtype=float, count=n_bs))
            test_metric = metric(gen_sample(rate_ref, sample_size))
            of.write("{:e}\n".format(cdf(test_metric)))

