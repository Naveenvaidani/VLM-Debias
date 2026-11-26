"""Toy model of weak injection and shortcut reliance (CPU, seconds).

Two image features reach a logistic readout through an injection gain g:
  core      c = y * mu_c + noise   (separates y in every group, small scale, low noise: slow to learn)
  shortcut  s = a * mu_s + noise   (a = y in 95% of training examples, large scale: fast to learn)
logit = g * (w_c * c + w_s * s) + b. A text-only bias b sees no image.
We train for a fixed number of steps and report worst-group accuracy on a group-balanced test set,
for SGD and for Adam, over gains g and seeds.

    python study/toy/injection_toy.py --out results_study/toy_injection.csv
"""
import argparse
import numpy as np
import pandas as pd


def data(n, p_major, rng, mu_c=0.5, sd_c=0.1, mu_s=2.0, sd_s=1.0):
    y = rng.choice([-1, 1], n)
    a = np.where(rng.random(n) < p_major, y, -y)
    x = np.stack([y * mu_c + sd_c * rng.standard_normal(n), a * mu_s + sd_s * rng.standard_normal(n)], 1)
    return x, y, a


def wga(w, b, g, x, y, a):
    pred = np.sign(g * x @ w + b)
    return min(np.mean(pred[(y == yy) & (a == aa)] == yy) for yy in (-1, 1) for aa in (-1, 1))


def train(g, opt, steps, lr, seed, n=2000, batch=64):
    rng = np.random.default_rng(seed)
    x, y, a = data(n, 0.95, rng)
    w, b = np.zeros(2), 0.0
    m, v = np.zeros(3), np.zeros(3)
    for t in range(1, steps + 1):
        i = rng.integers(0, n, batch)
        z = g * x[i] @ w + b
        r = -y[i] / (1 + np.exp(y[i] * z))                # d loss / d logit
        grad = np.r_[g * (r[:, None] * x[i]).mean(0), r.mean()]
        if opt == "adam":
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad ** 2
            step = lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        else:
            step = lr * grad
        w, b = w - step[:2], b - step[2]
    xt, yt, at = data(20000, 0.5, np.random.default_rng(10_000 + seed))
    return wga(w, b, g, xt, yt, at), w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    p.add_argument("--seeds", type=int, default=20)
    a = p.parse_args()
    rows = []
    for opt, lr in [("sgd", 0.5), ("adam", 0.01)]:
        for steps in [100, 300, 1000, 3000]:
            for g in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
                for s in range(a.seeds):
                    acc, w = train(g, opt, steps, lr, s)
                    rows.append(dict(opt=opt, steps=steps, gain=g, seed=s, wga=100 * acc,
                                     w_core=g * w[0], w_short=g * w[1]))
    d = pd.DataFrame(rows)
    if a.out:
        d.to_csv(a.out, index=False)
    t = d.groupby(["opt", "steps", "gain"]).agg(wga=("wga", "mean"), sd=("wga", "std"),
                                                 core=("w_core", "mean"), short=("w_short", "mean")).round(2)
    print(t.to_string())


if __name__ == "__main__":
    main()
