Evaluation of p-value test based on bootstrapped CDF with an event-type sample
==============================================================================

We are evaluating an event-type sample. A sample with event rate
(=probability) can be drawn as::

  np.random.binomial(1, event_rate, size=sample_size)

or::

  (np.random.rand(sample_size) < event_rate)

Currently, the first option is used.

For any sample we use the observed rate `sum(event_sample) /
len(event_sample)` as metric (=estimator) for the event rate.

Coverage
--------

Procedure
~~~~~~~~~

* Generate sample with probability `p` and size `n`.
* Generate `M` bootstrap-samples with::

    np.random.choice(sample, len(sample))

  calculate the metrics for each sample, yielding:

  .. math::

    [m_1^*, ..., m_M^*]

  (superscripted * to indicate that the metric values are retrieved
  using bootstrapping) and feed the resulting sample of metrics to
  `MetricCDF`.
* Draw a test sample, calculate the metric and call the `MetricCDF`
  object to retrieve the probability to observe a metric equal or better
  than this test metric:

  .. math::

    P^*(\text{metric equal-or-better than metric(test sample)})
    = \sum_{m^*_i \geq m(s_\text{test})} 1

Results
~~~~~~~

* Ideally, the cumulativ histogram of
  :math:`P^*(\text{equal-or-better})` should follow the 1. diagonal.


* Observations: 

  .. figure:: illustration.jpg
    :width: 800px

    Sketch of the observations.

  At high :math:`P^*` the observed CDF is below the ideal behaviour (1.
  diagonal). This results in *under*-coverage. This is considered a
  serious problem in (real ;) science (as it will lead to claiming a
  discovery of something "new" when there is indeed nothing new at a
  rate higher than expected (i.e. the significance level)). However, in
  our context of monitoring/alerting, under-coverage leads to more
  false-alarms which is not so critical (definitely better than a true
  anomaly slipping through).

  No clear systematic behaviour on event rate or sample size observable.
  Increasing the bootstrap sample size seems to improve the
  under-coverage slightly (cf. 200 vs 500 below).

  .. figure:: p_better_cdfs_binomial.png
    :width: 400px

    Observations with 200 bootstrap samples.
      
  .. figure:: p_better_cdfs_binomial_n_bs=500.png
    :width: 400px

    Observations with 500 bootstrap samples.

