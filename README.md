# Myoelectric Signal Processing

Filtering, onset detection, and a time and frequency domain feature library for myoelectric signals.

[![CI](https://github.com/Eelis03/myoelectric-signal-processing/actions/workflows/ci.yml/badge.svg)](https://github.com/Eelis03/myoelectric-signal-processing/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

This library implements the signal processing that sits between a surface electromyography
amplifier and a prosthesis controller: filter design and application, onset detection behind
a common protocol, a time and frequency domain feature library, and amplitude estimation for
proportional control. It is aimed at anyone building or evaluating a myoelectric control
chain who needs the delay, the false positive rate, and the timing bias of each stage stated
rather than assumed.

**Every result reported here is computed on synthetic signals.** No dataset is downloaded
and none is committed. The signals come from the documented generator in
`src/myoelectric/pipeline/generation.py`: a motor unit pool with size ordered recruitment
and rate coding after Fuglevand, Winter and Patla (1993), Hermite Rodriguez action
potentials after Lo Conte, Merletti and Sandri (1994), additive wideband noise at a
specified signal to noise ratio, power line interference with harmonics, and low frequency
movement artefact. What this establishes and what it does not is set out in
[docs/design-notes.md](docs/design-notes.md). In short, synthetic signals establish that the
mathematics is correct and that each method behaves as its source describes, and they do not
establish how any of it performs on a human arm.

For a real evaluation, three public datasets are suitable and none is redistributed here:
Ninapro (Atzori et al., 2014, <https://ninapro.hevs.ch>), putEMG (Kaczmarek et al., 2019,
<https://biolab.put.poznan.pl/putemg-dataset/>), and the PhysioNet examples of
electromyograms (Goldberger et al., 2000, <https://physionet.org/content/emgdb/1.0.0/>).
`src/myoelectric/pipeline/loaders.py` provides the loader interface, in the two formats a
dataset export normally takes, so a real recording can be dropped in without changing any
algorithm code.

## Problem

A myoelectric prosthesis has to decide, from a noisy voltage measured at the skin, when the
user intended to move and how hard. Four things stand between the electrode and that
decision.

1. The recording carries contaminations that overlap the signal in ways a single filter
   cannot address. Movement artefact sits below the signal band, mains interference sits
   inside it as narrow lines, and instrumentation noise covers all of it.
2. The instant the muscle became active has to be located. Detectors for this differ in how
   often they find a real onset, how often they declare one that is not there, and by how
   much they are systematically late or early. A detection rate reported without the
   matching false positive rate carries no information, since a detector that fires
   constantly reaches a detection rate of one.
3. Amplitude and spectral features have to be computed with definitions exact enough to be
   reproduced, and with an estimator whose resolution is stated.
4. Every stage costs delay, and the delay budget of a usable prosthesis is finite. Farrell
   and Weir (2007) measured that budget and reported an upper bound near 100 ms to 125 ms
   from muscle contraction to device response, covering everything: filtering, onset
   decision, classification, and the actuator.

The last point is why this library keeps causal and non causal processing separate
throughout. Zero phase filtering, which runs a filter forwards and then backwards, has
exactly zero group delay and is the right choice for offline analysis. It is also non causal:
producing the output at one sample requires samples that have not been acquired. A latency or
a timing bias measured under zero phase filtering understates the value a real controller
would experience.

## Approach

Filtering. Three designs, each built from second order sections, each with a stated rationale
and a group delay the library computes rather than quotes. A fourth order Butterworth band
pass from 20 Hz to 450 Hz covers the surface signal band, with the lower corner following the
movement artefact measurements of De Luca et al. (2010). A cascade of second order notches
removes the mains fundamental and its harmonics, which lie inside the pass band and cannot be
reached by moving the corners. A separate fourth order Butterworth high pass at 20 Hz is
provided for pipelines that need artefact removal without a band limit. Both application
modes are exposed. `apply_causal` is what a controller can run; `apply_zero_phase` is what
offline analysis should use, and `group_delay_samples` returns exactly zero for it, because
the two passes have equal and opposite phase responses.

Onset detection. Three methods behind one `OnsetDetector` protocol so they are directly
comparable: a threshold on a causally smoothed rectified envelope, after Di Fabio (1987); the
sliding window mean of Hodges and Bui (1996); and a Bonato style statistical detector after
Bonato, D'Alessio and Knaflitz (1998), which whitens the signal with an autoregressive model
fitted to the resting baseline and tests pairs of whitened samples against a chi squared
distribution with two degrees of freedom. The last of these sets its threshold from the
distribution, so its per test false alarm probability is a design parameter rather than an
outcome. Micera, Sabatini and Dario (1998) give the equivalent generalised likelihood ratio
formulation.

Features. The time domain set of Hudgins, Parker and Scott (1993) together with the
definitions collected by Phinyomark, Phukpattaranont and Limsakul (2012) and the
autoregressive description of Graupe and Cline (1975). The frequency domain set is computed
from a Welch averaged periodogram (Welch, 1967) with a stated segment length, and the median
frequency is located by interpolating the cumulative power rather than snapping to a bin.

Amplitude estimation. Four causal estimators with their design delay stated and their
imposed delay measured from a step response.

The alternatives that were considered and not chosen are recorded in
[docs/design-notes.md](docs/design-notes.md).

## Installation

Requires Python 3.12 or later.

```bash
git clone https://github.com/Eelis03/myoelectric-signal-processing.git
cd myoelectric-signal-processing
uv sync
```

Using pip instead of uv:

```bash
python -m venv .venv
.venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

```python
import numpy as np

from myoelectric.algorithm.filters import apply_causal, design_bandpass
from myoelectric.algorithm.onset import HodgesBuiDetector
from myoelectric.model.contraction import ContractionProfile
from myoelectric.model.noise import NoiseSpec
from myoelectric.model.sampling import SamplingSpec
from myoelectric.pipeline.generation import GenerationSpec, generate

sampling = SamplingSpec(sample_rate_hz=2000.0, duration_s=2.0)
trace = generate(
    GenerationSpec(
        sampling=sampling,
        profile=ContractionProfile.single(onset_s=0.8, offset_s=1.9, plateau_excitation=0.6),
        noise=NoiseSpec(snr_db=10.0),
    ),
    np.random.default_rng(20260731),
)

bandpass = design_bandpass(sampling.sample_rate_hz)
print(f"group delay at 100 Hz: {bandpass.group_delay_ms(np.array([100.0]))[0]:.2f} ms")

detected = HodgesBuiDetector().detect(apply_causal(bandpass, trace.signal), 2000.0)
truth = trace.onset_indices[0]
print(f"true onset {truth}, detected {detected.first_onset_index}")
print(f"timing error {1e3 * (detected.first_onset_index - truth) / 2000.0:.1f} ms")
```

Runnable examples live in `examples/`:

```bash
uv run python examples/generate_signal.py     # generate and describe one record
uv run python examples/filter_design.py       # responses, delays, causal against zero phase
uv run python examples/onset_benchmark.py     # detector comparison, the tables below
uv run python examples/feature_report.py      # the feature library on one window
uv run python examples/fatigue_demo.py        # median frequency over a sustained contraction
uv run python examples/amplitude_latency.py   # smoothing against latency
```

Each script accepts `--quick` for reduced settings, `--outdir` for the figure directory
(default `outputs/`, which is not tracked), and `--no-figures`.

## Results

Every number below is the output of the command named above it, at a 2000 Hz sample rate, on
synthetic signals, with the generator seeded at 20260731.

### Filter responses

`uv run python examples/filter_design.py`. Band pass, 20 Hz to 450 Hz, fourth order
Butterworth. Group delay is computed from the design by summing the group delays of its
second order sections.

| Frequency (Hz) | Causal gain (dB) | Causal group delay (ms) | Zero phase gain (dB) | Zero phase group delay (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 5 | -49.40 | 20.68 | -98.80 | 0.00 |
| 10 | -25.09 | 23.28 | -50.17 | 0.00 |
| 20 | -3.01 | 31.68 | -6.02 | 0.00 |
| 50 | -0.00 | 4.46 | -0.00 | 0.00 |
| 100 | -0.00 | 1.69 | -0.00 | 0.00 |
| 250 | -0.00 | 1.18 | -0.01 | 0.00 |
| 450 | -3.01 | 2.01 | -6.02 | 0.00 |
| 600 | -17.46 | 1.06 | -34.93 | 0.00 |
| 800 | -45.74 | 0.62 | -91.48 | 0.00 |

The group delay is 1.18 ms to 4.46 ms across the part of the band that carries most of the
power, and rises to 31.68 ms at the lower corner, where the envelope of a rising contraction
has much of its energy. Zero phase filtering doubles the attenuation in decibels and removes
the delay entirely.

Power line notch, 50 Hz with two harmonics, quality factor 30, so each section is 1.67 Hz
wide at the fundamental.

| Frequency (Hz) | Gain (dB) | Group delay (ms) |
| ---: | ---: | ---: |
| 45 | -0.11 | 5.34 |
| 48 | -0.67 | 28.59 |
| 50 | -250.97 | nan |
| 52 | -0.72 | 28.28 |
| 55 | -0.13 | 5.36 |
| 100 | -274.04 | nan |
| 150 | -276.07 | nan |

The gain at a notch centre is a transmission zero, so the printed figure is limited only by
floating point, and the group delay there is undefined rather than large, which is why those
entries read `nan`.

Measured against injected contamination on a generated record, over the settled part of the
record after the first 0.5 s, the band pass and notch chain gives:

| Component | Root mean square before | After | Change (dB) |
| --- | ---: | ---: | ---: |
| Power line | 0.2430 | 0.0040 | -35.7 |
| Movement artefact | 0.2492 | 0.0027 | -39.4 |
| Wideband noise | 0.1795 | 0.1160 | -3.8 |
| Clean signal | 0.9873 | 0.9433 | -0.4 |

Delay measured rather than quoted: an amplitude modulated burst at 100 Hz is displaced by
3.38 samples under causal filtering, against a design group delay of 3.37 samples, and by
-0.00 samples under zero phase filtering.

### Onset detection against signal to noise ratio

`uv run python examples/onset_benchmark.py`. Sixty active trials and sixty resting trials at
each ratio, 2 s each, causal band pass before detection, match tolerance 150 ms. Bias is
positive when the detection is late. Detector settings are the ones each source recommends.

| Detector | SNR (dB) | Detection rate | False positive rate | FP per s | Bias mean (ms) | Bias SD (ms) | Bias p25 (ms) | Bias median (ms) | Bias p75 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| envelope-threshold k=3 | -5 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 | nan | nan | nan | nan | nan |
| envelope-threshold k=3 | 0 | 0.600 +/- 0.063 | 0.000 +/- 0.000 | 0.000 | 100.9 | 28.3 | 83.0 | 102.8 | 125.2 |
| envelope-threshold k=3 | 5 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 | 64.7 | 23.7 | 47.2 | 61.8 | 77.6 |
| envelope-threshold k=3 | 10 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 | 47.8 | 18.2 | 32.9 | 47.5 | 58.2 |
| envelope-threshold k=3 | 15 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 | 41.8 | 19.9 | 29.3 | 36.5 | 44.9 |
| envelope-threshold k=3 | 20 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 | 39.3 | 12.7 | 31.0 | 37.2 | 44.2 |
| hodges-bui k=3 | -5 | 0.333 +/- 0.061 | 0.000 +/- 0.000 | 0.000 | 81.4 | 46.4 | 37.4 | 98.8 | 118.6 |
| hodges-bui k=3 | 0 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 | 52.4 | 30.5 | 32.2 | 53.8 | 70.2 |
| hodges-bui k=3 | 5 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 | 21.4 | 18.5 | 8.5 | 14.0 | 34.6 |
| hodges-bui k=3 | 10 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 | 12.9 | 15.4 | 1.0 | 9.8 | 24.1 |
| hodges-bui k=3 | 15 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 | 6.7 | 14.1 | -2.6 | 3.2 | 11.5 |
| hodges-bui k=3 | 20 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 | 8.4 | 10.8 | 0.5 | 6.5 | 16.0 |
| bonato-glr p=0.001 | -5 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 | nan | nan | nan | nan | nan |
| bonato-glr p=0.001 | 0 | 0.233 +/- 0.055 | 0.000 +/- 0.000 | 0.000 | 86.9 | 40.5 | 59.2 | 80.2 | 122.4 |
| bonato-glr p=0.001 | 5 | 0.717 +/- 0.058 | 0.000 +/- 0.000 | 0.000 | 79.8 | 29.8 | 60.0 | 74.0 | 98.5 |
| bonato-glr p=0.001 | 10 | 0.983 +/- 0.017 | 0.000 +/- 0.000 | 0.000 | 55.6 | 30.9 | 30.2 | 54.0 | 70.5 |
| bonato-glr p=0.001 | 15 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 | 32.3 | 21.2 | 17.5 | 27.8 | 41.8 |
| bonato-glr p=0.001 | 20 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 | 27.5 | 15.1 | 16.9 | 27.8 | 35.2 |

Reading of the table. At these settings all three detectors produce no false positives at
all: with sixty resting trials, a rate recorded as zero has an upper 95 per cent bound of
0.050 by the rule of three. Because every threshold is estimated from the resting baseline of
the record being tested, the false positive rate is a property of the threshold and the
decision rule and does not vary with the amplitude of the contraction, which is why that
column is flat. The detectors separate on the other two measures. Hodges and Bui reaches a
detection rate of one by 0 dB and a mean bias of 6.7 ms to 12.9 ms above 10 dB. The envelope
threshold reaches one by 5 dB but stays 39 ms to 48 ms late, because it cannot respond faster
than the group delay of its own 8 Hz smoothing filter, which is 28 ms at zero frequency. The
Bonato style detector needs 15 dB to reach one at this operating point and is 27 ms to 32 ms
late there.

The bias distribution matters. At 5 dB the Bonato detector has a mean bias of 79.8 ms with a
standard deviation of 29.8 ms and an interquartile range from 60.0 ms to 98.5 ms, so the mean
is not a summary of a tight distribution. Reporting the mean alone would hide that.

### Threshold sensitivity at 5 dB

The same experiment with each detector at three sensitivities. This is the operating
characteristic that a single detection rate hides.

| Detector | Detection rate | False positive rate | FP per s | Bias mean (ms) | Bias median (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| envelope-threshold k=1 | 0.950 +/- 0.028 | 0.350 +/- 0.062 | 0.217 | 26.6 | 27.5 |
| envelope-threshold k=2 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 | 46.9 | 38.8 |
| envelope-threshold k=3 | 0.983 +/- 0.017 | 0.000 +/- 0.000 | 0.000 | 62.9 | 52.5 |
| hodges-bui k=1 | 0.717 +/- 0.058 | 0.950 +/- 0.028 | 1.358 | -13.4 | -1.0 |
| hodges-bui k=2 | 0.983 +/- 0.017 | 0.183 +/- 0.050 | 0.100 | 11.0 | 6.5 |
| hodges-bui k=3 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 | 19.7 | 11.0 |
| bonato-glr p=0.1 | 0.633 +/- 0.062 | 1.000 +/- 0.000 | 1.842 | -14.1 | 10.8 |
| bonato-glr p=0.01 | 0.967 +/- 0.023 | 0.133 +/- 0.044 | 0.067 | 53.5 | 46.5 |
| bonato-glr p=0.001 | 0.717 +/- 0.058 | 0.000 +/- 0.000 | 0.000 | 67.4 | 71.0 |

At the most sensitive settings the detection rate falls rather than rises, because a detector
triggered by noise places its first onset before the contraction and that detection then
falls outside the match window. The negative mean bias of `hodges-bui k=1` and
`bonato-glr p=0.1` is the same effect seen from the other side. This is why the two rates
have to be read together.

Decision delay, which is separate from timing bias: 50.0 ms for the envelope threshold,
25.0 ms for Hodges and Bui, and 5.0 ms for the Bonato style detector at 2000 Hz. Timing bias
says where a detector places the onset; decision delay says how long after that instant it
can say so. A controller pays the sum.

### Muscle fatigue

`uv run python examples/fatigue_demo.py`. A 60 s sustained contraction at 0.6 normalised
excitation, analysed in thirty 2 s epochs, band pass and notch applied with zero phase
filtering because the analysis is offline and no timing decision depends on it. The action
potential time constant is scaled from 1.00 to 1.35 across the contraction, representing
slowed muscle fibre conduction velocity.

| Feature | Epochs | Start (Hz) | End (Hz) | Slope (Hz/s) | Normalised slope (%/s) | t | df | One sided p | R squared |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| median frequency | 30 | 95.8 | 69.9 | -0.351 | -0.383 | -11.23 | 28 | 3.47e-12 | 0.818 |
| mean frequency | 30 | 102.5 | 76.2 | -0.418 | -0.419 | -23.10 | 28 | 4.49e-20 | 0.950 |

The statistic is the Student t statistic of the ordinary least squares slope of the feature
on epoch start time, on 28 degrees of freedom, with the one sided probability under the null
hypothesis of no trend. The test is one sided because the physiological prediction is
directional: fatigue lowers median frequency and does not raise it.

The size of the fall agrees with the model that produced it. Scaling the action potential
time constant by 1.35 should compress the spectrum to 1/1.35, that is 0.741, of its initial
frequency, and the fitted line falls to 0.778 of its initial value over the 58 s span. The
median frequency shift under sustained contraction is one of the most reliably reproduced
effects in surface electromyography, which is why it is used here to validate the spectral
feature implementation rather than presented as a finding.

### Amplitude estimation for proportional control

`uv run python examples/amplitude_latency.py`. One step contraction with zero rise time, at
19.97 dB achieved signal to noise ratio, causal band pass and notch before estimation.
Latency is the time to reach half of the change from rest to plateau. Plateau ripple is the
standard deviation of the estimate over the settled plateau divided by its mean.

| Estimator | Nominal delay (ms) | Measured latency (ms) | Rise time 10-90 (ms) | Plateau ripple (%) |
| --- | ---: | ---: | ---: | ---: |
| moving-average-50ms | 24.8 | 25.7 | 29.1 | 23.0 |
| moving-average-100ms | 49.8 | 47.8 | 84.6 | 15.1 |
| moving-average-200ms | 99.8 | 107.9 | 173.4 | 7.8 |
| moving-rms-100ms | 49.8 | 24.9 | 73.0 | 16.3 |
| lowpass-2Hz-order2 | 112.5 | 118.1 | 184.6 | 7.0 |
| lowpass-4Hz-order2 | 56.3 | 55.4 | 90.5 | 13.9 |
| lowpass-8Hz-order2 | 28.1 | 31.2 | 24.4 | 21.8 |
| exponential-25ms | 24.8 | 20.9 | 14.6 | 21.9 |
| exponential-50ms | 49.8 | 28.1 | 96.2 | 14.6 |

Measured latency tracks the nominal group delay for the moving average and the low pass
estimators, which checks that the delay figures quoted in the design are the delays actually
imposed. Two entries differ from their nominal delay for reasons that have closed forms. The
moving root mean square reaches half amplitude in a quarter of its window rather than half,
because the mean square ramps linearly across the window and the square root compresses the
first part of that ramp. The exponential estimator reaches half amplitude at 0.69 of its
group delay, the natural logarithm of two, because its step response is a single exponential
rather than a ramp. Ripple falls from 23.0 per cent to 7.0 per cent across the family, and
every point of that reduction is paid for in delay.

### Feature library on one window

`uv run python examples/feature_report.py`. A 0.25 s window taken from the plateau of a
generated contraction at 20 dB, after a causal band pass and notch. The amplitude threshold
is three resting standard deviations, 0.3916, and the slope threshold is the variance of the
resting first difference, 0.002988, which has the units of amplitude squared.

| Time domain feature | Value | Frequency domain feature | Value |
| --- | ---: | --- | ---: |
| Mean absolute value | 0.7009 | Median frequency (Hz) | 103.51 |
| Mean absolute value slope | -0.1856, 0.04576, 0.06071 | Mean frequency (Hz) | 105.06 |
| Zero crossings | 21 | Spectral moment 0 | 0.7333 |
| Slope sign changes | 33 | Spectral moment 1 | 77.04 |
| Waveform length | 116.13 | Spectral moment 2 | 9212 |
| Root mean square | 0.8984 | Root mean square frequency (Hz) | 112.08 |
| Variance | 0.8087 | Resolution (Hz) | 8 |
| Integrated electromyogram | 350.44 | | |
| Willison amplitude | 88 | | |
| Autoregressive coefficients | 1.9656, -1.1670, 0.0500, 0.0653 | | |

Under a gain of 2.0, mean absolute value, root mean square, waveform length and integrated
electromyogram scale by exactly 2.0000, variance by exactly 4.0000, and zero crossings, slope
sign changes, Willison amplitude, median frequency and mean frequency are all unchanged,
provided each threshold is scaled by the factor its own units require.

## Architecture

Five layers. The dependency direction runs one way: `analysis` reads traces from `pipeline`,
`pipeline` calls `algorithm` and constructs `model` values, `algorithm` consumes `model`
values, and `model` depends on nothing.

| Module | Responsibility |
| --- | --- |
| `src/myoelectric/model/sampling.py` | Sample rate and record length, with conversions between seconds and samples |
| `src/myoelectric/model/motor_unit.py` | Motor unit pool: recruitment thresholds, rate coding, Hermite Rodriguez action potentials |
| `src/myoelectric/model/contraction.py` | Contraction and gesture definitions as a trapezoidal neural excitation profile |
| `src/myoelectric/model/noise.py` | Specifications for wideband noise, power line interference, and movement artefact |
| `src/myoelectric/algorithm/filters.py` | Band pass, notch and high pass design, causal and zero phase application, group and phase delay |
| `src/myoelectric/algorithm/autoregressive.py` | Biased autocorrelation, Levinson Durbin recursion, whitening filter |
| `src/myoelectric/algorithm/features_time.py` | Time domain feature library with every definition written out |
| `src/myoelectric/algorithm/features_freq.py` | Welch spectrum, median and mean frequency, spectral moments |
| `src/myoelectric/algorithm/onset.py` | Three onset detectors behind the `OnsetDetector` protocol |
| `src/myoelectric/algorithm/envelope.py` | Four causal amplitude estimators and the step response measurement |
| `src/myoelectric/pipeline/generation.py` | Synthetic record generation with ground truth onsets |
| `src/myoelectric/pipeline/detection_sweep.py` | Detector evaluation over signal to noise ratio, active and resting trials |
| `src/myoelectric/pipeline/fatigue.py` | Sustained contraction protocol with per epoch spectral features |
| `src/myoelectric/pipeline/latency.py` | Step contraction study for the amplitude estimators |
| `src/myoelectric/pipeline/loaders.py` | `EmgRecording` and the loader protocol for substituting a real dataset |
| `src/myoelectric/analysis/detector_metrics.py` | Detection rate, false positive rate, and the timing bias distribution |
| `src/myoelectric/analysis/fatigue_stats.py` | Linear trend of a spectral feature with its t statistic and probability |
| `src/myoelectric/analysis/reporting.py` | Markdown tables for every trace |
| `src/myoelectric/analysis/figures.py` | Figures, built through the matplotlib object interface with no global state |
| `examples/` | Thin wiring scripts, no logic |

## Testing

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

The suite has three tiers: property and invariant tests covering the mathematics,
regression tests pinning recorded behaviour, and integration tests running each
example script under a reduced iteration count.

The property tier measures rather than assumes. A filter gain is measured from a filtered
sine and compared against the gain the design predicts. The distinction between causal and
zero phase filtering is tested by perturbing one sample and checking which outputs move, so
the property is demonstrated rather than described. Every time domain feature is checked on a
signal whose answer is known by inspection or in closed form, including zero crossings on a
square wave, waveform length on a triangle wave, and autoregressive coefficients on a
sinusoid, whose order two Yule Walker solution is exactly `(2 cos w, -1)`. Median frequency is
checked against a line spectrum whose analytic value is known. Every detector is checked both
for finding a clean onset and for declaring nothing on a resting record, and its timing bias
is measured over repeated trials rather than assumed to be zero.

Tolerances are derived from the measurement, never from the error that happened to be
observed. Timing tolerances are expressed in samples, spectral tolerances in frequency bins,
and rate tolerances come from the binomial standard error over the number of trials. Steady
state gain tolerances are computed from the tail of the filter's own impulse response, plus a
floating point accumulation bound, because a tolerance derived only from the settling
argument falls below the arithmetic noise of the computation it constrains and then fails on
a different machine by a few units in the last place. The regression tier pins feature values
on fixed inputs, closed form filter responses, detector verdicts, counts and aggregate rates,
all of which reproduce elsewhere.

## References

### Methods

- Fuglevand, A. J., Winter, D. A., and Patla, A. E. Models of recruitment and rate coding
  organization in motor-unit pools. Journal of Neurophysiology, 70(6):2470-2488, 1993.
  DOI: [10.1152/jn.1993.70.6.2470](https://doi.org/10.1152/jn.1993.70.6.2470)
- Lo Conte, L. R., Merletti, R., and Sandri, G. V. Hermite expansions of compact support
  waveforms: applications to myoelectric signals. IEEE Transactions on Biomedical
  Engineering, 41(12):1147-1159, 1994.
  DOI: [10.1109/10.335863](https://doi.org/10.1109/10.335863)
- De Luca, C. J. The use of surface electromyography in biomechanics. Journal of Applied
  Biomechanics, 13(2):135-163, 1997.
  DOI: [10.1123/jab.13.2.135](https://doi.org/10.1123/jab.13.2.135)
- De Luca, C. J., Gilmore, L. D., Kuznetsov, M., and Roy, S. H. Filtering the surface EMG
  signal: movement artifact and baseline noise contamination. Journal of Biomechanics,
  43(8):1573-1579, 2010.
  DOI: [10.1016/j.jbiomech.2010.01.027](https://doi.org/10.1016/j.jbiomech.2010.01.027)
- Hermens, H. J., Freriks, B., Disselhorst-Klug, C., and Rau, G. Development of
  recommendations for SEMG sensors and sensor placement procedures. Journal of
  Electromyography and Kinesiology, 10(5):361-374, 2000.
  DOI: [10.1016/S1050-6411(00)00027-4](https://doi.org/10.1016/S1050-6411%2800%2900027-4)
- Gustafsson, F. Determining the initial states in forward-backward filtering. IEEE
  Transactions on Signal Processing, 44(4):988-992, 1996.
  DOI: [10.1109/78.492552](https://doi.org/10.1109/78.492552)
- Di Fabio, R. P. Reliability of computerized surface electromyography for determining the
  onset of muscle activity. Physical Therapy, 67(1):43-48, 1987.
  DOI: [10.1093/ptj/67.1.43](https://doi.org/10.1093/ptj/67.1.43)
- Hodges, P. W., and Bui, B. H. A comparison of computer-based methods for the determination
  of onset of muscle contraction using electromyography. Electroencephalography and Clinical
  Neurophysiology / Electromyography and Motor Control, 101(6):511-519, 1996.
  DOI: [10.1016/S0013-4694(96)95190-5](https://doi.org/10.1016/S0013-4694%2896%2995190-5)
- Bonato, P., D'Alessio, T., and Knaflitz, M. A statistical method for the measurement of
  muscle activation intervals from surface myoelectric signal during gait. IEEE Transactions
  on Biomedical Engineering, 45(3):287-299, 1998.
  DOI: [10.1109/10.661154](https://doi.org/10.1109/10.661154)
- Micera, S., Sabatini, A. M., and Dario, P. An algorithm for detecting the onset of muscle
  contraction by EMG signal processing. Medical Engineering and Physics, 20(3):211-223, 1998.
  DOI: [10.1016/S1350-4533(98)00017-4](https://doi.org/10.1016/S1350-4533%2898%2900017-4)
- Hudgins, B., Parker, P., and Scott, R. N. A new strategy for multifunction myoelectric
  control. IEEE Transactions on Biomedical Engineering, 40(1):82-94, 1993.
  DOI: [10.1109/10.204774](https://doi.org/10.1109/10.204774)
- Phinyomark, A., Phukpattaranont, P., and Limsakul, C. Feature reduction and selection for
  EMG signal classification. Expert Systems with Applications, 39(8):7420-7431, 2012.
  DOI: [10.1016/j.eswa.2012.01.102](https://doi.org/10.1016/j.eswa.2012.01.102)
- Graupe, D., and Cline, W. K. Functional separation of EMG signals via ARMA identification
  methods for prosthesis control purposes. IEEE Transactions on Systems, Man, and
  Cybernetics, SMC-5(2):252-259, 1975.
  DOI: [10.1109/TSMC.1975.5408681](https://doi.org/10.1109/TSMC.1975.5408681)
- Durbin, J. The fitting of time-series models. Revue de l'Institut International de
  Statistique, 28(3):233-244, 1960.
  DOI: [10.2307/1401322](https://doi.org/10.2307/1401322)
- Welch, P. D. The use of fast Fourier transform for the estimation of power spectra: a
  method based on time averaging over short, modified periodograms. IEEE Transactions on
  Audio and Electroacoustics, 15(2):70-73, 1967.
  DOI: [10.1109/TAU.1967.1161901](https://doi.org/10.1109/TAU.1967.1161901)
- Stulen, F. B., and De Luca, C. J. Frequency parameters of the myoelectric signal as a
  measure of muscle conduction velocity. IEEE Transactions on Biomedical Engineering,
  BME-28(7):515-523, 1981.
  DOI: [10.1109/TBME.1981.324738](https://doi.org/10.1109/TBME.1981.324738)
- Merletti, R., Knaflitz, M., and De Luca, C. J. Myoelectric manifestations of fatigue in
  voluntary and electrically elicited contractions. Journal of Applied Physiology,
  69(5):1810-1820, 1990.
  DOI: [10.1152/jappl.1990.69.5.1810](https://doi.org/10.1152/jappl.1990.69.5.1810)
- Merletti, R., and Parker, P. A. Electromyography: Physiology, Engineering, and Noninvasive
  Applications. Wiley-IEEE Press, 2004.
  DOI: [10.1002/0471678384](https://doi.org/10.1002/0471678384)
- Farrell, T. R., and Weir, R. F. The optimal controller delay for myoelectric prostheses.
  IEEE Transactions on Neural Systems and Rehabilitation Engineering, 15(1):111-118, 2007.
  DOI: [10.1109/TNSRE.2007.891391](https://doi.org/10.1109/TNSRE.2007.891391)

### Datasets cited for real evaluation, none redistributed here

- Atzori, M., Gijsberts, A., Castellini, C., Caputo, B., Hager, A.-G. M., Elsig, S.,
  Giatsidis, G., Bassetto, F., and Muller, H. Electromyography data for non-invasive
  naturally-controlled robotic hand prostheses. Scientific Data, 1:140053, 2014.
  DOI: [10.1038/sdata.2014.53](https://doi.org/10.1038/sdata.2014.53).
  Available from <https://ninapro.hevs.ch>
- Kaczmarek, P., Mankowski, T., and Tomczynski, J. putEMG: a surface electromyography hand
  gesture recognition dataset. Sensors, 19(16):3548, 2019.
  DOI: [10.3390/s19163548](https://doi.org/10.3390/s19163548).
  Available from <https://biolab.put.poznan.pl/putemg-dataset/>
- Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark,
  R. G., Mietus, J. E., Moody, G. B., Peng, C.-K., and Stanley, H. E. PhysioBank,
  PhysioToolkit, and PhysioNet: components of a new research resource for complex
  physiologic signals. Circulation, 101(23):e215-e220, 2000.
  DOI: [10.1161/01.CIR.101.23.e215](https://doi.org/10.1161/01.CIR.101.23.e215).
  Examples of electromyograms available from <https://physionet.org/content/emgdb/1.0.0/>

### Dependencies

| Package | Purpose | Licence |
| --- | --- | --- |
| [numpy](https://numpy.org/) | Arrays, fast Fourier transforms, seeded random generation with the PCG64 bit generator | BSD-3-Clause |
| [scipy](https://scipy.org/) | Butterworth and notch design, second order section filtering, Welch periodogram, chi squared and Student t distributions, linear regression | BSD-3-Clause |
| [matplotlib](https://matplotlib.org/) | Figures, used through the object interface with the Agg canvas | Matplotlib licence, a BSD style licence based on the Python Software Foundation licence |
| [pytest](https://pytest.org/) | Test runner, development only | MIT |
| [ruff](https://docs.astral.sh/ruff/) | Linting and import ordering, development only | MIT |
| [mypy](https://mypy-lang.org/) | Static type checking in strict mode, development only | MIT |

Software citations: Harris, C. R., et al. Array programming with NumPy. Nature,
585:357-362, 2020.
DOI: [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
Virtanen, P., et al. SciPy 1.0: fundamental algorithms for scientific computing in Python.
Nature Methods, 17:261-272, 2020.
DOI: [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
Hunter, J. D. Matplotlib: a 2D graphics environment. Computing in Science and Engineering,
9(3):90-95, 2007. DOI: [10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55).

## License

Released under the MIT license. See [LICENSE](LICENSE).
