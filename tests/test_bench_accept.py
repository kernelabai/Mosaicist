import numpy as np
import pytest

from mosaicist.bench import MIN_NOISE, noise_floor, summarize, verdict
from mosaicist.converge import Beam, Candidate, accept


def test_summarize_and_noise_floor():
    rng = np.random.default_rng(0)
    batches = [100 + rng.standard_normal(60) * 0.3 for _ in range(4)]
    t = summarize(batches[0])
    assert t.ci_low <= t.median <= t.ci_high and t.n == 60
    noise = noise_floor(batches)
    assert MIN_NOISE <= noise < 0.02
    with pytest.raises(ValueError):
        noise_floor([batches[0]])


def test_verdict():
    assert verdict(97.0, 100.0, 0.01) == "faster"
    assert verdict(100.5, 100.0, 0.01) == "equal"
    assert verdict(103.0, 100.0, 0.01) == "slower"


def c(id, time, dist, ok=True):
    return Candidate(id, time, dist, ok)


def test_acceptance_rule():
    best = c("v1", 100.0, 0.5)
    assert not accept(c("x", 50.0, 0.1, ok=False), best, 0.01).accepted  # numerics gate first
    assert accept(c("x", 95.0, 0.9), best, 0.01).accepted  # faster wins even if farther
    assert accept(c("x", 100.3, 0.4), best, 0.01).accepted  # tie on time, closer in D
    assert not accept(c("x", 100.3, 0.6), best, 0.01).accepted
    d = accept(c("x", 110.0, 0.1), best, 0.01)
    assert not d.accepted and d.closer_but_slower


def test_beam_keeps_accepted_best_and_runners_up():
    beam = Beam(k=3)
    beam.offer(c("v0", 200.0, 0.9), 0.01)
    beam.offer(c("v1", 150.0, 0.7), 0.01)
    d = beam.offer(c("v2", 149.5, 0.8), 0.01)  # same speed within noise and farther: not best
    assert not d.accepted and beam.best.id == "v1"
    assert [m.id for m in beam.members] == ["v1", "v2", "v0"]
    beam.offer(c("v3", 180.0, 0.2), 0.01)  # closer but slower: logged
    assert [m.id for m in beam.rejected_closer_but_slower] == ["v3"]
    assert len(beam.members) == 3
    assert not beam.converged(t_ref=100.0, noise=0.01)
    beam.offer(c("v4", 100.2, 0.1), 0.01)
    assert beam.best.id == "v4" and beam.converged(t_ref=100.0, noise=0.01)
