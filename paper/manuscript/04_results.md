# Results (manuscript section)

*Compact version for the journal; every number comes from `results_study/` (per-run CSV files) and the tables in
the README. Full tables are in `paper/RESULTS_DISCUSSION.md` and the supplementary material.*

---

## A learnable zero-start gate lowers worst-group accuracy

With Qwen3.5-4B and SigLIP-base features, three connectors reach similar worst-group accuracy, and the gated
cross-attention connector with a learnable gate falls well below them (Table \ref{tab:main}, Figure \ref{fig:main}).

| Task | MLP projector | Q-Former | Learnable gate | Fixed gate |
| :--- | :---: | :---: | :---: | :---: |
| Waterbirds | 74.5 ± 1.5 (3) | 71.7 ± 2.5 (3) | 63.6 ± 10.4 (15) | 74.2 ± 3.4 (15) |
| Watermark | 63.6 ± 4.4 (3) | 68.6 ± 4.1 (3) | 40.5 ± 18.0 (15) | 65.7 ± 10.3 (15) |

: Test worst-group accuracy (%), mean ± s.d. over seeds (number of seeds in brackets), Qwen3.5-4B. With 5 seeds the MLP projector and Q-Former give 74.4 and 72.8 on Waterbirds and 66.2 and 69.2 on Watermark. \label{tab:main}

\begin{figure*}[t]
\centering
\includegraphics[width=\linewidth]{figures/main_connectors.pdf}
\caption{Test worst-group accuracy of the four connectors (Qwen3.5-4B, SigLIP-base) on (a) Waterbirds and (b) Watermark. Bars are means with one standard deviation; dots are single seeds; hatching distinguishes connectors; the numbers under each bar give the mean and the number of seeds. Only the learnable, zero-start gate fails, and it is also the least stable}\label{fig:main}
\end{figure*}

Fixing the gate raises worst-group accuracy in all four LLM and task cells (Table \ref{tab:h1}, Figure \ref{fig:forest}). The
effects are large ($d_z$ = 1.10 and 1.34 on Qwen3.5-4B) and every cell survives Holm correction (largest adjusted
$p$ = 0.009). The result holds when the checkpoint is chosen without group labels (+17.7 and +25.5 points), with
unpaired Welch and Mann–Whitney tests (all $p \le 0.002$), and on the confirmation seeds alone (Waterbirds +10.5,
$p = 9.5 \times 10^{-4}$; Watermark +26.7, $p = 2.5 \times 10^{-4}$). The fixed gate also raises average accuracy
(84.1 to 88.0 on Waterbirds, 76.9 to 86.1 on Watermark), so the gain is not a trade against average accuracy.

| LLM | Task | Seeds | Learnable | Fixed | Gain [95% CI] | H1 $p$ | Collapsed (learnable / fixed) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3.5-4B | Waterbirds | 15 | 63.6 | 74.2 | +10.6 [5.3, 16.0] | $4 \times 10^{-4}$ | 2 / 0 |
| Qwen3.5-4B | Watermark | 15 | 40.5 | 65.7 | +25.2 [14.8, 35.6] | $7 \times 10^{-5}$ | 10 / 0 |
| Phi-4-mini | Waterbirds | 5 | 60.4 | 70.1 | +9.7 [3.8, 15.7] | 0.005 | 0 / 0 |
| Phi-4-mini | Watermark | 10 | 29.4 | 54.9 | +25.5 [8.2, 42.7] | 0.004 | 8 / 6 |

: Hypothesis H1, paired by seed. One-sided paired t-test at $\alpha = 0.025$; all four cells survive Holm
correction. Collapsed runs have worst-group accuracy below 50%. On Phi-4-mini Watermark, seeds 5 to 9 were added
after the 5-seed test missed $\alpha$ ($p = 0.044$); tested alone they give $p = 0.023$. \label{tab:h1}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/forest_gain.pdf}
\caption{Paired gain from fixing the gate (fixed minus learnable gate, worst-group accuracy points) with 95\% confidence intervals; squares Qwen3.5-4B, circles Phi-4-mini, open triangles Qwen3.5-9B (exploratory, Table \ref{tab:9b})}\label{fig:forest}
\end{figure}

The learnable gate also makes robustness depend on the seed (H2). On Qwen3.5-4B its variance is higher on both tasks
(variance-ratio $p = 9 \times 10^{-5}$ and 0.02) and it collapses in 2 of 15 and 10 of 15 runs, against none for
the fixed gate (Figure \ref{fig:seeds}). On Phi-4-mini the variance difference is not significant ($p$ = 0.20 and
0.23), and on Watermark the fixed gate still collapses in 6 of 10 runs.

\begin{figure*}[t]
\centering
\includegraphics[width=\linewidth]{figures/h1_seeds_qwen.pdf}
\caption{Each line joins one paired run of the learnable gate to the fixed gate (Qwen3.5-4B) on (a) Waterbirds and (b) Watermark. Solid lines rise and dashed lines fall; the dotted line marks the 50\% collapse threshold}\label{fig:seeds}
\end{figure*}

## The effect holds with a larger language model

We repeated the comparison with Qwen3.5-9B, keeping the code, features, data and training budget unchanged
(Table \ref{tab:9b}). These runs are exploratory. On Watermark, the zero-start gate fell to 12.8% worst-group
accuracy and collapsed in all five seeds, while the fixed gate reached 63.2% and collapsed in none. The gain, +50.3
points ($d_z$ = 2.03), is twice the 4B gain. Average accuracy rose from 63.7% to 86.1%. On Waterbirds, five
paired seeds give +9.9 points ($p$ = 0.040, $d_z$ = 1.05; Wilcoxon $p$ = 0.031), about the 4B gain; the confidence
interval includes zero. One of five paired zero-start runs collapsed, against none of the fixed-gate runs, and a
further, unpaired zero-start run of seed 3 reached only 19.3%. Seeds 3 and 4 were aligned in a later session than
seeds 0 to 2, so the alignment is shared only within each pair. The behaviour of the gate is the same as at 4B. The zero-start gate ended below $|\tanh\alpha| = 0.005$ in every run, and its gradient
changed sign on 39% to 51% of steps. A gate started at 0.1 drifted to about 0.073 and matched the fixed gate (69.7%,
+56.9 over the zero start). On this task the capped injection also kept the gain (63.9%), unlike at 4B, where it
lost 21.5 points on Watermark.

| Task | Seeds | Learnable | Fixed | Gain [95% CI] | $p$ | $d_z$ | Collapsed |
| :---------------------- | :---: | :-------: | :-----: | :--------------------: | :----------: | :----: | :-----------: |
| Waterbirds | 5 | 60.9 | 70.7 | +9.9 [−1.8, 21.6] | 0.040 | 1.05 | 1 of 5 / 0 of 5 |
| Watermark | 5 | 12.8 | 63.2 | +50.3 [19.6, 81.1] | 0.005 | 2.03 | 5 of 5 / 0 of 5 |
| Watermark, start 0.1 | 5 | 12.8 | 69.7 | +56.9 [42.3, 71.4] | $2 \times 10^{-4}$ | 4.85 | 5 of 5 / 0 of 5 |
| Watermark, capped | 5 | 12.8 | 63.9 | +51.0 [45.7, 56.4] | $6 \times 10^{-6}$ | 11.94 | 5 of 5 / 1 of 5 |

: Qwen3.5-9B (exploratory), test worst-group accuracy (%), paired by seed. "Fixed" is the fixed gate, or the
variant named in the row. The $p$-values are from one-sided paired t-tests and are uncorrected. On Waterbirds an
additional, unpaired zero-start run of seed 3 (19.3%) is not included in the paired test. Collapsed: paired runs
below 50%, learnable / fixed. \label{tab:9b}

## The zero start, not learnability, is the cause

Controls change one property of the learnable gate at a time (Table \ref{tab:controls}; all exploratory).

| Control | Seeds | Variant | Reference | What it shows |
| :------------------------- | :---: | :---------- | :---------------------- | :---------------------------------------- |
| Learnable gate started open ($\tanh\alpha_0 = 0.995$) | 5 | 71.8 / 66.8 | fixed 72.9 / 72.1 | starting open removes most of the gap |
| Fixed gate with injection capped at the text norm | 5 | 69.9 / 50.6 | zero start 61.9 / 49.7; fixed 72.9 / 72.1 | keeps most of the gain on Waterbirds ($-3.0$ against fixed, $p$ = 0.090) but loses all of it on Watermark ($-21.5$, $p$ = 0.003; +0.9 over the zero start); 2 of 5 Watermark seeds collapse |
| Learnable gate started at $\tanh\alpha_0 = 0.01$ | 5 | 69.1 / 22.5 | zero start 61.9 / 49.7; fixed 72.9 / 72.1 | gate falls back to 0.001 to 0.008; Watermark collapses in 5 of 5 seeds, $-27.2$ below the zero start ($p$ = 0.009) |
| Learnable gate started at $\tanh\alpha_0 = 0.1$ | 5 | 73.1 / 65.3 | zero start 61.9 / 49.7; fixed 72.9 / 72.1 | +11.2 and +15.6 over the zero start ($p$ = 0.11, 0.11); +0.2 and $-6.8$ against fixed ($p$ = 0.91, 0.49) |
| Learnable gate started at $\tanh\alpha_0 = 0.01$, default $W_o$ | 5 | 68.9 / 52.0 | zero start 61.9 / 49.7; fixed 72.9 / 72.1 | gate falls back to 0.002 to 0.009; not significantly different from the zero start (+7.0, +2.4; $p$ = 0.39, 0.89); 2 of 5 Watermark seeds collapse |
| Learnable gate started at $\tanh\alpha_0 = 0.1$, default $W_o$ | 5 | 73.9 / 57.6 | zero start 61.9 / 49.7; fixed 72.9 / 72.1 | gate stays at 0.046 to 0.074; +12.0 and +7.9 over the zero start ($p$ = 0.12, 0.39); +0.9 and $-14.6$ against fixed ($p$ = 0.48, 0.017) |
| Learnable gate started half open ($\tanh\alpha_0 = 0.5$) | 5 | 72.6 / 62.7 | zero start 61.9 / 49.7; fixed 72.9 / 72.1 | +10.7 and +13.0 over the zero start ($p$ = 0.19, 0.15); $-0.3$ and $-9.5$ against fixed ($p$ = 0.91, 0.066) |
| Gate excluded from gradient clipping | 3 | 67.2 / 52.9 | zero start, same seeds 62.1 / 50.3 | no significant change (+5.1, +2.5); gate stays below 0.003 |
| No weight decay on the gate | 2 | 52.2 / 28.0 | zero start, same seeds 56.8 / 47.6 | no rescue ($-4.6$, $-19.6$) |
| Connector learning rate $5 \times 10^{-5}$ / $2 \times 10^{-4}$ / $10^{-3}$ (Waterbirds) | 2 | 35.9 / 56.8 / 65.8 | fixed 65.6 / 73.5 / 66.5 | the gap shrinks as the gate moves further (final 0.002 / 0.004 / 0.018) |
| fp16 weights instead of 4-bit | 3 | 63.2 / 44.1 | fixed 73.2 / 65.1 | same direction (+10.0, +21.1; not significant with 3 seeds) |
| Alignment 300 steps at $3 \times 10^{-4}$ (Waterbirds) | 5 | 71.6 | fixed 76.0 | +4.4 ($p$ = 0.065); no collapsed runs |
| Alignment 3,000 steps at $3 \times 10^{-4}$ | 5 | 71.3 / 54.6 | fixed 73.9 / 69.1 | gate stays below 0.003 in alignment; gap +2.7 and +14.4 (neither significant); 2 of 5 Watermark seeds collapse |
| Alignment 6,000 steps at $3 \times 10^{-4}$ | 5 | 65.5 / 46.2 | fixed 74.0 / 70.8 | +8.5 ($p$ = 0.062) and +24.7 ($p$ = 0.014); 0 and 3 of 5 learnable seeds collapse, none fixed |
| Eight training epochs (Waterbirds) | 4 | 67.9 | fixed 79.0 | +11.0 ($p$ = 0.023); one of five fixed-gate runs collapsed |

: Controls on Qwen3.5-4B, test worst-group accuracy (%), Waterbirds / Watermark unless stated. \label{tab:controls}

A learnable gate that starts open ($\tanh\alpha_0 = 0.995$) comes within 1.1 and 5.3 points of the fixed gate, so
the cost does not come from the gate being learnable. Smaller starts help on Waterbirds but only partly on Watermark.
On Waterbirds, gates started at 0.1 or 0.5 match the fixed gate (within one point) and lie 10.7 to 12.0 points above
the zero start, although with five seeds none of these gains over the zero start is significant. On Watermark they
recover between a third and two thirds of the gap: they lie 7.9 to 15.6 points above the zero start and 6.8 to 14.6
below the fixed gate; with the default $W_o$ the 0.1 start remains significantly below the fixed gate ($p$ = 0.017).
The half-open gate moves little (final $\tanh\alpha$ 0.44). The 0.1 start shrinks during training, to 0.058 to 0.064,
but stays about a hundred times above the zero-start gate; its gradient also changes sign often (27% to 51% of steps
per epoch push the gate up, with a slight net push down). A start of 0.01 is not enough. That gate falls back to the
zero-start range (final $|\tanh\alpha|$ 0.001 to 0.008), and with $W_o = 0$ Watermark collapses below 50% in all five
seeds (22.5, against 49.7 for the zero start on the same seeds, $p$ = 0.009). Because this control also starts $W_o$
at zero, we repeated the 0.01 and 0.1 starts with the default $W_o$ of the zero-start arm, so that only the gate's
start value differs. With the default $W_o$ the 0.01 start is no different from the zero start (68.9 / 52.0 against
61.9 / 49.7), so the extra loss with $W_o = 0$ came from the $W_o$ initialisation, and the 0.1 start still stays open
(0.046 to 0.074). The threshold therefore lies between 0.01 and 0.1. A start at or near zero is the largest single
cost, but on the watermark task a small non-zero start does not fully close the gap to a fixed gate.

*Longer alignment.* A first run with ten times longer alignment at the default learning rate destabilised the
learnable-gate alignment (caption loss rose from 2.0 to 7.9), so we repeated it at a lower learning rate
($3 \times 10^{-4}$) with 3,000 and 6,000 steps, five paired seeds each. Both arms now aligned stably. The zero-start
gate still never exceeded $|\tanh\alpha| = 0.003$ during alignment, and its caption loss ended higher than the fixed
gate's (2.23 to 2.34 against 1.84 to 2.05). The robustness gap depends on the task. On Watermark it remained: +14.4
points after 3,000 steps ($p$ = 0.11, 2 of 5 learnable runs collapsed, none fixed) and +24.7 after 6,000 steps
($p$ = 0.014, $d_z$ = 1.50, 3 of 5 against none), about the size of the gap with the default alignment (+22.5 on
the same seeds). On Waterbirds the gap was smaller at every length: +2.7 after 3,000 steps ($p$ = 0.15) and +8.5 after 6,000
($p$ = 0.062), with no collapsed runs. To separate the learning rate from the length, we also aligned for the
default 300 steps at $3 \times 10^{-4}$ on Waterbirds. The gap was +4.4 ($p$ = 0.065), already within the range of
the longer runs and below the +11.0 of the default setting on the same seeds. On Waterbirds, then, the lower
alignment learning rate rather than the longer alignment shrinks the gap, and ten or twenty times more steps do not
shrink it further. In no setting did longer alignment open the gate or reverse the direction of the gap.

## The gate's gradient is large but cancels itself

After alignment the zero-start gate sits at $|\tanh\alpha| \approx 0.0005$ and never exceeds 0.01. Its gradient is
not small: the mean absolute pre-clipping gradient per epoch ranges from 3 to 1,528 across 13 learnable-gate runs,
and in some runs it is almost all of the global gradient norm. With Adam, each step moves $\alpha$ by at most about
the learning rate $\eta = 2 \times 10^{-4}$, so over the roughly 900 to 1,200 task steps a gradient of constant sign
would move $\alpha$ by up to about 0.2. The observed change is below 0.003. The gradient must therefore change sign
from step to step and cancel. The signed gradient, logged in seven further zero-start runs, confirms this: the mean
absolute gradient reaches 37 to 1,282 per epoch, yet 37% to 60% of steps push the gate up, the sign changes on 45%
to 49% of consecutive steps, and the gate ends at 0.001 to 0.005. The gate started at 0.01 behaves the same way
(30% to 58% of steps push it up). This differs from the dead-gradient account of [farazi2026gate], which requires an
injected signal that does not depend on the text; ours does.

## The pathway is weak, not closed

The learnable gate passes 15 to 35 times less visual signal than the fixed gate (injection ratio 3.5 against 112,
6 runs each), because $W_o$ grows to partly offset the small gate. Removing the visual pathway at inference, or
zeroing the image features, makes the learnable-gate model give the same answer to every image in every run:
average accuracy falls from 83.0 to the share of one class on Waterbirds ($-46.9$ points, 4 of 4 seeds, $p$ = 0.035)
and from 81.8 to 50.0 on Watermark (3 of 3 seeds, $p$ = 0.003). The fixed-gate model does the same. The learnable
gate therefore still uses the image; its pathway is weak, not closed. Giving each test example the image of a
randomly chosen other example (4 seeds on Watermark, 3 on Waterbirds) confirms that the answer depends on the
particular image. Average accuracy falls to chance on Watermark for both gates (learnable 68.6 to 51.1, fixed 86.2 to
52.0) and drops on Waterbirds (85.8 to 62.3 and 90.1 to 64.9). On Watermark the learnable gate's worst-group accuracy
rises with the wrong image (24.6 to 44.2), as expected if its errors come from the watermark in its own image. In these runs the fixed gate again led on
worst-group accuracy (+16.2 on Waterbirds, 4 seeds; +14.4 on Watermark, 3 seeds). Within learnable-gate runs the injection ratio does not predict
worst-group accuracy (Spearman $\rho$ = 0.33, $p$ = 0.27). Both gate variants attend to the watermark corner more
than to the rest of the image (16.7% and 18.2% of attention against 8.2% if uniform), yet only the learnable-gate
model relies on it.

## Connector type matters little once the gate is fixed

MLP projector and Q-Former do not differ significantly (5 seeds; $p$ = 0.23 on Waterbirds and 0.40 on Watermark),
and neither differs significantly from the fixed gate, although equivalence within three points is not established.
Matching the number of image tokens (36 instead of 49 MLP tokens) changes nothing (74.7 against 74.9). The fixed-gate
connector needs about half the step time and memory of the MLP projector (1.7 against 4.0 s per step; 6.2 against
12.1 GB on one T4).

## Boundaries of the effect

*Injection depth.* With four gated blocks inside the decoder, the learnable and fixed gates do not differ (Waterbirds
70.8 against 70.4, 5 seeds; Watermark 68.3 against 70.6, 4 seeds), although the per-layer learnable gates also stay
small (mean $|\tanh\alpha|$ 0.027).

*Encoder size.* With SigLIP-large every connector improves on Waterbirds by 6 to 11 points, and the gap narrows to
+8.0 ($p$ = 0.09, 4 paired seeds). On Watermark it remains: 53.2 against 75.6 (+22.4, $d_z$ = 0.85, $p$ = 0.13,
5 seeds) (Figure \ref{fig:encoder}).

*Third task.* On CelebA the direction is the same but not significant (43.5 against 51.1, +7.7, $p$ = 0.48, 5 paired
seeds).

*Shortcut strength.* At a 90% watermark correlation all arms learn the task (learnable 75.7, fixed 77.5); at 99% all
arms fail (6.7 and 20.1). The gap is largest at the 95% used in the main experiment.

*Group labels.* Group distributionally robust optimisation (Group DRO) [sagawa2020groupdro], which uses group labels in training, closes the gap on Waterbirds (+2.3, $p$ = 0.70,
4 seeds) but not on Watermark (+17.5, $p$ = 0.059).

\begin{figure*}[t]
\centering
\includegraphics[width=\linewidth]{figures/encoder_scale.pdf}
\caption{Waterbirds worst-group accuracy with SigLIP-base and SigLIP-large features. Bars are means with one standard deviation, dots are single seeds, and the numbers under each bar give the mean and the number of seeds. Every connector gains from the larger encoder; the learnable gate still lags}\label{fig:encoder}
\end{figure*}

## Free-form answers under shifted priors

On VQA-CP v2 every model loses 15 to 21 points when the answer priors shift, so the subset exposes the language
prior (Table \ref{tab:vqacp}). A text-only model with the same LLM, LoRA and budget sets the reference. The learnable gate does no
better than this model in distribution (47.6 against 48.2, $p$ = 0.23), and removing its visual pathway changes
nothing, so it does not use the image at test time (injection ratio 0.05). Its 2.1-point lead under shifted priors
($p$ = 0.041) remains with the image removed (29.2), so it comes from training with the pathway present rather
than from the test image. The fixed gate is worse than having no image at all: $-5.7$ points in distribution
against the text-only model ($p$ = 0.001). Its injected signal is about 70 times the text norm, and removing it
raises shifted-prior accuracy by 4.6 points.

Capping the injected signal at the text norm removes this cost. The capped connector scores 49.6 in distribution and
30.6 under shifted priors, 7.0 points above the uncapped fixed gate in distribution ($p$ = 0.020) and 3.5 points
above the text-only model under shifted priors ($p$ = 0.028). It is also the only model whose accuracy depends on
the test image: removing the pathway or zeroing the image features costs 2.8
to 3.5 points in distribution and 1.0 to 2.0 under shifted priors ($p$ = 0.041 to 0.063, 3 seeds). Strong injection
therefore disturbs free-form generation, and an injection held at the text norm avoids that while still using the
image. On the binary shortcut tasks, however, the cap gives back part of the fixed gate's gain on Waterbirds and all of it on Watermark (Table \ref{tab:controls}):
Waterbirds 69.9 against 72.9 for the uncapped fixed gate ($p$ = 0.090) and Watermark 50.6 against 72.1 ($p$ = 0.003,
$d_z$ = $-3.0$), the level of the zero-start gate (49.7), with two of five Watermark seeds collapsed. The capped model still uses the image on these tasks: removing
the pathway makes it give one answer to every image. Injection strength therefore trades shortcut robustness on a
class readout against the quality of free-form answers.

| Model | Test, same priors | Image removed | Test, shifted priors | Image removed | Prior answer rate | Injection ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Text only | 48.2 ± 0.4 | n/a | 27.1 ± 0.1 | n/a | 39.9% / 35.0% | 0 |
| Learnable gate | 47.6 ± 0.2 | 47.6 ± 0.2 | 29.1 ± 0.8 | 29.2 ± 0.7 | 39.9% / 33.2% | 0.05 |
| Fixed gate | 42.6 ± 0.7 | 40.4 ± 0.9 | 27.6 ± 0.8 | 32.2 ± 0.4 | 33.8% / 31.1% | 70.1 |
| Capped injection | 49.6 ± 1.7 | 46.7 ± 0.4 | 30.6 ± 0.9 | 29.7 ± 1.2 | 39.0% / 32.7% | 1.00 |

: VQA-CP v2 subset, VQA accuracy (%), mean ± standard deviation over 3 seeds. Prior answer rate: share of answers
equal to the most frequent training answer for the question type (same priors / shifted priors). Not comparable with
published VQA-CP results. \label{tab:vqacp}

## Standard fixes on both gates

Retraining only the connector on half of the group-balanced validation set, with the LLM and LoRA frozen (a
connector-level form of deep feature reweighting, DFR [kirichenko2023dfr]), lifts the fixed-gate connector on
Watermark from 72.5 to 89.6 ± 0.5 (5 seeds). This beats the same fix on the MLP projector (81.2; +8.4,
$p < 0.001$) and a linear probe trained the same way on the frozen features (85.1) (Figure \ref{fig:fixes}). On
Waterbirds one epoch of retraining gives almost nothing (+0.2 over five seeds, 73.9 to 74.1; seeds 0 to 2 gained
2.6 to 4.2 points and seeds 3 and 4 lost), and three epochs give +8.9 over three seeds (73.9 to 82.7, $p$ = 0.077).
The Watermark result above uses one epoch.
Qwen3.5-4B with its own vision tower, zero-shot and with a threshold calibrated on validation, reaches only 35.5 on
Waterbirds.

To test whether standard fixes remove the gate effect, we applied the same connector-level DFR and a
just-train-twice (JTT) retraining [liu2021jtt], which repeats the stage-2 model's training errors 20 times, to both
gates (Table \ref{tab:fixes}; three paired seeds, exploratory). DFR rescues the zero-start gate on Watermark (51.3 to
87.9, $p$ = 0.013), where it comes within 2.5 points of the fixed gate with DFR. On Waterbirds it does not rescue it
(65.0 to 68.5, $p$ = 0.22), and the fixed gate with DFR stays 14.2 points ahead ($p$ = 0.023). JTT does not help the
zero-start gate on either task, and on Waterbirds the fixed gate with JTT is 20.9 points ahead of the zero-start gate
with JTT ($p$ = 0.006). Group labels used only at the connector thus undo the gate's cost when the shortcut is a
local visual mark, but not when it is the image background, and error upweighting without group labels does not
undo it.

| Task | Learnable | + DFR | + JTT | Fixed | + DFR | + JTT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Waterbirds | 65.0 ± 9.5 | 68.5 ± 5.9 | 54.6 ± 6.2 | 73.9 ± 4.8 | 82.7 ± 4.4 | 75.5 ± 4.3 |
| Watermark | 51.3 ± 13.0 | 87.9 ± 2.4 | 52.9 ± 24.9 | 68.8 ± 1.8 | 90.4 ± 0.9 | 56.6 ± 8.1 |

: Connector-level DFR (balanced half of validation, other half selects) and JTT (training errors repeated 20 times),
both retraining only the connector for three epochs from the stage-2 model, Qwen3.5-4B, test worst-group accuracy
(%), mean ± s.d. over seeds 0 to 2. \label{tab:fixes}

\begin{figure*}[t]
\centering
\includegraphics[width=\linewidth]{figures/fixes.pdf}
\caption{Connector-only deep feature reweighting (conn-DFR), group-balanced sampling and a linear probe on the frozen features. Bars are means with one standard deviation, dots are single seeds, and the numbers under each bar give the mean and the number of seeds; (a) Waterbirds, (b) Watermark. Starred bars use group labels during training (group-balanced sampling). Conn-DFR on the fixed gate reaches 89.6\% worst-group accuracy on Watermark and gives no gain on Waterbirds}\label{fig:fixes}
\end{figure*}

## Gates in released models

The released weights of models that use the same zero-start tanh gates show the same regime. Nine of the 20
cross-attention gates of IDEFICS-80B have $|\tanh\alpha| < 0.05$, and so do all gates of Llama 3.2 11B Vision (8 of 8)
and 90B Vision Instruct (20 of 20), whose values are at most 0.012. A small gate does not show that these models
ignore the image; we did not test them for shortcut reliance.
