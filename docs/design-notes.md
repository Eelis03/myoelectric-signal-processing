# Design notes for Myoelectric Signal Processing

## What the results are computed on, and what that establishes

Every number this project reports is computed on synthetic signals produced by
`src/myoelectric/pipeline/generation.py`. No dataset is downloaded and none is committed.
This section comes first because it determines how everything below should be read.

The generator is a motor unit pool model, not a recording. Recruitment thresholds are
distributed exponentially across the pool, discharge rate rises linearly with excitation
above threshold and saturates at a unit specific peak, and action potential amplitude grows
with recruitment threshold, all following Fuglevand, Winter and Patla (1993). Each unit
discharges as an inhomogeneous renewal process with an interspike interval coefficient of
variation of 0.2. Action potentials are second order Hermite Rodriguez functions, the
standard compact support basis for this purpose (Lo Conte, Merletti and Sandri, 1994), with
time constants chosen so that individual potentials peak between 71 Hz and 114 Hz. The
resulting composite spectrum has a median frequency near 100 Hz, which is the range surface
recordings occupy.

What synthetic signals establish:

- That the mathematics is correct. A filter can be shown to have the gain and the group delay
  its design predicts, a feature can be checked against a closed form, and a spectral
  estimator can be checked against a line spectrum whose analytic median frequency is known.
  None of that needs real data and all of it would be harder to verify with real data,
  because a real recording has no ground truth to check against.
- That the ground truth is exact. The first motor unit discharge is known to the sample, so a
  detector's timing bias is measurable without an annotator. In a real recording the onset is
  whatever an expert marked, and the disagreement between experts is comparable in size to
  the differences between the detectors being compared.
- That each method behaves as its source describes, under the conditions the source assumed.
  The Bonato style detector assumes the resting signal is Gaussian after whitening, and the
  generator produces exactly that, so the detector's false alarm rate can be checked against
  the chi squared quantile that set its threshold. On real data that assumption is
  approximate.

What synthetic signals do not establish:

- How any of this performs on a human arm. Real surface recordings carry electrode impedance
  drift, crosstalk from neighbouring muscles, non stationary noise, amplifier saturation,
  movement artefact correlated with the contraction rather than independent of it, and
  subject to subject variation in every parameter the generator fixes.
- The absolute detection rates and timing biases in the Results section. Those numbers
  characterise the detectors against this generator. Their ordering is more likely to survive
  a change of data than their values, and even the ordering should be measured again before
  it is relied on.
- Anything about a specific gesture set, muscle, or electrode placement. The generator has
  one channel and one muscle.

The route to a real evaluation is deliberately short. Everything downstream of
`EmgRecording` in `src/myoelectric/pipeline/loaders.py` takes an array and a sample rate.
`NpzRecordingLoader` and `CsvRecordingLoader` cover the two formats a dataset export normally
takes, and `recording_from_trace` wraps a generated record in the same object, so the
synthetic path and the real path are identical from that point on. Three things must be
checked when substituting real data, and they are listed in the module docstring: the sample
rate, because every filter has to be redesigned at the recording rate rather than reused; the
mains frequency, which is 50 Hz or 60 Hz depending on where the recording was made and which
`analysis/powerline_check.py` measures from the record when the dataset does not say; and the
meaning of any annotated onset, because a cue time or an expert annotation is not the instant
the first motor unit discharged, and a timing bias measured against one is not comparable
with the biases reported here.

## Method selection

### Filter designs

Butterworth, band pass 20 Hz to 450 Hz, fourth order. The lower corner follows De Luca et al.
(2010), who measured the trade off between movement artefact rejection and signal loss
directly and recommend 20 Hz for surface recordings. The upper corner sits below the Nyquist
frequency of a 1000 Hz record, which is the lowest rate in common use, while retaining the
fast edges of individual potentials. Butterworth rather than Chebyshev because the pass band
is maximally flat: amplitude features computed after a filter with pass band ripple carry
that ripple, and the ripple is frequency dependent, so it appears as an apparent change in
amplitude when the spectrum shifts, which is exactly what the fatigue analysis measures.

Second order sections rather than transfer function coefficients throughout. A direct form
transfer function of order eight or more loses accuracy when its roots cluster near the unit
circle, and a quality factor 30 notch places its roots as close to the unit circle as this
project gets.

Notches for the mains rather than a wider stop band. Mains interference is narrowband and
lies inside the electromyogram pass band. Removing it by moving the band pass corners would
remove a large part of the signal with it, because the surface spectrum peaks between 50 Hz
and 150 Hz. A quality factor of 30 gives a 1.67 Hz wide notch at 50 Hz, and the measured
response table in the README shows that a component 5 Hz away is attenuated by 0.13 dB.

Group delay computed rather than quoted. The library computes the group delay of a design by
summing the group delays of its second order sections, which is exact because delays add in
cascade, and which avoids expanding the cascade into a single transfer function. At a
transmission zero the group delay is undefined, because the phase of a response of zero
magnitude carries no information, and the library returns a value that is not a number rather
than the arbitrarily large figure that differentiating a vanishing response produces.

Causal and zero phase kept separate. The distinction is not stylistic. Zero phase filtering
reads samples that have not been acquired, which a real time controller cannot do, and it
also smears an onset backwards in time, so a detector evaluated after a zero phase filter
appears to have less bias than the same detector would have in a controller. The detector
sweep therefore uses causal filtering by default, and the fatigue analysis uses zero phase
filtering, since the fatigue analysis is offline and no timing decision depends on it.

### Deciding the mains frequency rather than assuming it

The notch above has to be designed at a frequency, and on synthetic data that frequency is
whatever the generator was given. On a real recording it is 50 Hz or 60 Hz, and a dataset that
states its sample rate and its annotation protocol does not always state where it was recorded.
Designing at the wrong one is a silent failure of a particular kind: three transmission zeros
are placed in the middle of the signal band, the interference passes through untouched, and
nothing downstream reports either, because a notch cannot tell whether it removed anything.

`analysis/powerline_check.py` measures the answer instead. Each candidate is scored by the
power at its fundamental over the median power of the bins beside it, at a 1 Hz resolution, and
a frequency is named only when it stands 6 dB above both that floor and the other candidate.
Three choices inside that are worth stating. The floor is local rather than an average over the
record, because the surface spectrum peaks between 50 Hz and 150 Hz, which is exactly where the
lines are. The floor of every candidate excludes the main lobe of every candidate, so that a
60 Hz line cannot raise the floor the 50 Hz candidate is judged against. And the score is the
excess at the fundamental alone: the harmonics are measured and reported, because they say how
many sections a notch needs, but mains interference falls away with each harmonic while the
signal it competes against rises towards its peak, so a buried harmonic is not evidence against
a fundamental that is not buried.

The threshold is what makes the check able to answer that there is no line, which a rule that
simply took the larger of the two candidates could not do. Where it sits relative to the
fluctuation of the estimate is measured rather than asserted: the test suite runs sixty records
that carry no line and requires that none is declared.

### Onset detectors

Three methods were implemented so that they can be compared under one experiment rather than
across three papers with different data.

The envelope threshold, after Di Fabio (1987), is the method everything else is measured
against. It is included because it is what most implementations actually do, and because its
weakness is instructive: the low pass filter that makes the envelope stable enough to
threshold is also what delays it, and its 39 ms to 48 ms bias at high signal to noise ratio
is close to the 28 ms group delay of its own 8 Hz smoothing filter.

Hodges and Bui (1996) compared computer based onset methods against visual determination and
found that a sliding window mean of the rectified, 50 Hz low pass filtered signal, with a
threshold of one to three baseline standard deviations and a window near 25 ms, tracked the
visual determination most closely. The sliding window makes the test insensitive to isolated
large samples without needing the heavy smoothing the envelope method requires, which is why
it has both the lowest bias and no false positives at three standard deviations in the
results.

Bonato, D'Alessio and Knaflitz (1998) derived a test statistic as an approximation of the
generalised likelihood ratio for a change in the variance of a Gaussian process: whiten with
an autoregressive model fitted to the resting baseline, then compare the sum of squares of
pairs of whitened samples against a chi squared distribution with two degrees of freedom.
Micera, Sabatini and Dario (1998) describe the equivalent formulation. The reason to include
it is that its threshold comes from a distribution rather than from tuning, so its per test
false alarm probability is a design parameter. That property is checked directly in the
tests: on a resting record the fraction of pair statistics above the threshold matches the
probability that set it, to within the binomial sampling error of the measurement.

All three take their baseline statistics from a leading resting segment of the record being
tested. That is what makes their false positive rate independent of the contraction
amplitude, and therefore why the false positive rate column in the Results section does not
vary with signal to noise ratio.

All three share one decision rule, in `_run_starts`: a run of the statistic above the
threshold qualifies as an onset when it is long enough, when it starts after a refractory
period, and when the preceding stretch below the threshold was long enough. The last
condition is the second half of the activation interval rule of Bonato et al.: an interval
that has begun must end before another can begin. Without it, a detector declares a new onset
every time its statistic dips during an ongoing contraction, which inflates the count of
declarations on an active record without saying anything about the detector.

### Spectral estimation

Welch (1967) averaged periodogram with a Hann window and 50 per cent overlap. Averaging
reduces the variance of the estimate at the cost of frequency resolution, and the resolution
is carried alongside every spectrum in `PowerSpectrum.resolution_hz` and reproduced in the
method string, so a reported median frequency always comes with the resolution it was
computed at. Every spectral tolerance in the test suite is expressed as a multiple of that
resolution.

Median frequency by interpolating the cumulative power between the two bins that bracket half
the total, rather than returning the first bin at or above half. Returning the bin quantises
the estimate to the bin width and introduces a systematic upward bias of half a bin, which
for the 4 Hz resolution used in the fatigue analysis is 2 Hz, comparable with the fall being
measured over the first several epochs.

### Feature thresholds and their units

Zero crossings and Willison amplitude compare an amplitude against a threshold, so their
thresholds have the units of the signal. Slope sign changes compare a product of two first
differences, so its threshold has the units of the signal squared. Setting the two to the
same numeric value is a units error, and it is a common one. It also produces a specific
failure that is easy to miss: on an oversampled record consecutive samples differ by much
less than the signal amplitude, so a slope threshold set to the square of a sensible
amplitude threshold suppresses every turning point and returns zero. `time_domain_features`
therefore takes the two separately, and the recommended setting for the slope threshold is
the variance of the first difference of a resting segment, which is the scale on which noise
alone produces turning points and which scales as the square of the gain, so the scale
invariance property is preserved.

### Reducing a frequency dependent group delay to one number

A delay budget adds numbers up, and a filter does not have one delay. The group delay of the
band pass runs from 1.18 ms at 250 Hz to 31.68 ms at its 20 Hz corner, and the group delay of
a quality factor 30 notch approaches 90 ms one Hertz from its centre. Any single figure charged
to a budget is therefore the output of a rule, and the rule has to be stated with the figure.

Two rules are implemented and `analysis/delay_budget.py` records which one produced a stage.

The default is the group delay averaged over the band and weighted by the squared magnitude
response of the design, under the stated assumption that the input carries equal power at every
frequency in the band. This is the delay that the envelope of a broadband signal experiences,
and the envelope is the quantity a proportional controller acts on. It is the correct default
for the specific reason that it treats a notch properly: at a transmission zero the filter is
not delaying the signal, it is removing it, and a weight proportional to surviving power gives
that frequency the weight it has earned.

The alternative is the largest group delay anywhere in the pass band, where the pass band is
the part of the requested band within a stated number of decibels of the design's maximum gain
in it. This is a strict bound. It is also a very conservative one whenever a notch is present,
because group delay peaks exactly where a magnitude response falls fastest. On the band pass
and notch chain used here the two rules give 4.16 ms and 94.14 ms, and the entire difference
belongs to a component within about one Hertz of the mains line.

Neither rule was chosen alone. Reporting only the weighted mean would hide that a bound exists;
reporting only the bound would declare a chain infeasible on the strength of a frequency the
chain exists to destroy. Both are computed by the same function under a `rule` argument, and
both appear in the README.

### Fatigue as a validation rather than a finding

The downward shift of median frequency during a sustained contraction is one of the most
reliably reproduced effects in surface electromyography (De Luca, 1997; Merletti, Knaflitz
and De Luca, 1990). It is included here for that reason: an implementation of median
frequency that fails to reproduce it is wrong. The generator produces the effect through its
stated mechanism, by scaling the Hermite Rodriguez time constant, and because the magnitude
spectrum of that function peaks at `1 / (pi lambda)`, a scale of 1.35 predicts a compression
to 0.741 of the initial frequency. The fitted line falls to 0.778, so the implementation
reproduces both the direction and, within the scatter of the estimate, the size.

## Rejected alternatives

### A finite impulse response band pass instead of Butterworth

An equiripple finite impulse response filter has exactly linear phase, so its group delay is
constant across frequency, which would remove the frequency dependence visible in the band
pass response table. It was not chosen because reaching a 20 Hz transition at a 2000 Hz sample
rate needs several hundred taps, and the resulting group delay of half the filter length
would be tens of milliseconds at every frequency rather than the 1.18 ms to 4.46 ms measured
across the mid band. Trading a variable few milliseconds for a constant several tens of
milliseconds is the wrong trade in a system with a 125 ms total budget.

### An adaptive notch or spectral interpolation for the mains

An adaptive notch tracks a drifting mains frequency, and spectral interpolation removes the
line without a notch at all by replacing the affected bins. Both are better than a fixed notch
when the mains frequency wanders. Neither was chosen because both introduce state that has to
converge, and a value from an unconverged adaptation is not reproducible across machines,
which would make it unpinnable in the regression tier. A fixed notch is a closed form design
with a stated width, and that width covers the drift a mains supply actually exhibits.

### Notching both mains frequencies instead of deciding between them

A record could be passed through notches at 50 Hz and at 60 Hz and their harmonics without
deciding anything, which needs no measurement at all and cannot name the wrong frequency
because it names both. It was not chosen because the three extra transmission zeros are not
free. Each removes a 1.67 Hz slice of a band whose power is concentrated between 50 Hz and
150 Hz, each contributes its own group delay, and the worst case rule of the delay budget
charges a chain the largest group delay anywhere in its pass band, which is where a notch is
steepest. The measurement costs one Welch estimate, once, before any filter is designed, and
leaves the notch with the three sections it needs rather than six.

### Adaptive whitening in the Bonato detector

The reference method fits the autoregressive model once, on the resting baseline. Refitting
it continuously would track a signal whose statistics change during the contraction. It was
not chosen because the detector's threshold is derived from the null distribution of the
whitened resting signal, and continuously refitting the model changes that null distribution
in a way that no longer has the closed form the threshold depends on. The value of this
detector is that its false alarm probability is a design parameter; an adaptive version would
lose that.

### A double threshold onset and offset segmenter

Bonato et al. pair each onset with an offset to segment activation intervals. Only the onset
half is implemented here, with the silence requirement described above standing in for the
offset rule. A full segmenter would allow activation duration to be reported and would make
the count of declarations on an active record meaningful. It was not implemented because the
question this project asks is where the onset is, and adding offsets would add a second set
of parameters and a second set of metrics without changing the onset comparison.

### Higher order or non parametric spectral estimation

An autoregressive spectral estimate gives a smoother spectrum from a shorter record than
Welch does, which would allow shorter fatigue epochs. It was not chosen for the spectral
features because its resolution is not a stated bin width but a consequence of the model
order, so a tolerance expressed in bins would have no meaning, and because model order
selection introduces a choice that would have to be justified separately at every epoch. The
autoregressive machinery is present and is used where a parametric model is the right tool,
in the whitening filter and in the autoregressive feature.

### Pinning per trial detector outputs in the regression tier

An earlier design pinned per trial detector outputs across the whole sweep rather than
aggregate rates. That was rejected on reproducibility grounds. A per trial verdict at a signal
to noise ratio where the detection rate is neither zero nor one sits close to a threshold by
definition, and a difference of one unit in the last place can flip it, so a pinned list of
verdicts fails on a different machine without anything having changed. The regression tier
pins aggregate rates with a tolerance derived from the binomial standard error, feature values
on fixed deterministic inputs, closed form filter responses, and the verdicts on one seeded
record where the crossings are not marginal.

## Limitations closed

### The delay budget is now enforced rather than quoted

This section previously listed, as limitation 8, that the library reported the delay each stage
imposes and compared those figures against the range Farrell and Weir (2007) report, while
nothing in the code prevented a caller from assembling a chain whose total delay exceeded that
budget. That is now closed.

What was added. `src/myoelectric/analysis/delay_budget.py` turns each stage into a `DelayStage`
carrying a delay and the basis on which it was computed, sums them into a `DelayBudget`, and
compares the total against a limit whose default is the 125 ms upper bound of the Farrell and
Weir range. `enforce` raises `DelayBudgetExceededError` naming the largest contributor. Three
constructors cover the stages this library implements: `filter_stage` from a `FilterDesign` and
a band, `detector_stage` from an `OnsetDetector`, and `envelope_stage` from an
`EnvelopeEstimator`. `fixed_stage` covers everything else, so a caller can enter a classifier
and an actuator rather than leaving the budget quietly short.

What it cost. Three things, none of them free.

The first is that a filter has no single group delay, so charging one to a budget needs a rule.
That is the subject of the section above, and it forced two rules rather than one, both
implemented, both reported. A budget assembled without noticing which rule produced its filter
stage is still a number without a meaning; the rule is recorded on the stage so that it cannot
be read without its conditions.

The second is that zero phase filtering had to be refused outright rather than charged zero.
Its group delay genuinely is zero, so an implementation that simply summed group delays would
accept it and produce a total that no controller can achieve. `filter_stage` raises on a
`zero_phase` mode instead. The cost of that decision is that the same function cannot be used
to describe the offline analysis path, which is the correct outcome and still a restriction.

The third is that the enforced total covers only what the caller enters into it. The check is
a sum, not a discovery procedure, and it cannot know about a stage nobody mentioned.

What it changed. On the chain the README documents, conditioning, onset decision and amplitude
estimation consume 78.91 ms of the 125 ms bound, leaving 46.09 ms for the classifier and the
actuator. Substituting the steadiest amplitude estimator in the library, on the reasonable
grounds that 7.0 per cent plateau ripple beats 14.6 per cent, takes the total to 141.70 ms and
the check fails. That substitution is exactly the kind a reader of the amplitude table would
make, and before this module nothing in the library would have objected to it.

What remains. The budget is a design time check on a chain that has been described to it. It
does not measure the delay of a running implementation, it does not account for the cost of the
feature window a classifier would need, which is 250 ms for the 0.25 s window used in the
feature report and larger than the whole budget on its own, and it has no view of the
computational time any stage takes on real hardware. The seven limitations below are unchanged.

## Known limitations

Limitation 8 of this list, that the delay budget was quoted rather than enforced, has been
removed and is documented above.

The first entry is structural rather than an oversight. Every number this project reports is
computed on synthetic signals, and closing that would require downloading and evaluating on a
public dataset. This project takes no network access at runtime and commits no data, so the
synthetic data limitation cannot be closed from inside it; what it can do, and does, is keep
the substitution path short and documented, which is the subject of the first section of these
notes.

1. **Single channel.** The generator produces one channel. Crosstalk between neighbouring
   muscles, which is one of the main practical difficulties in myoelectric control, cannot be
   represented, and no multi channel method can be evaluated. Adding channels would require a
   volume conductor model relating each motor unit to each electrode.

2. **Stationary noise.** The wideband noise is Gaussian and stationary, and the movement
   artefact is independent of the contraction. In a real recording the noise level changes
   with electrode impedance, and the artefact is largest exactly when the limb moves, that is
   during the contraction the detector is trying to find. Every false positive rate reported
   here is therefore an optimistic figure.

3. **The ground truth is the first discharge.** That is the correct definition for a
   generator, and it is not the definition any real dataset uses. A cue based dataset marks
   the instant the subject was told to move, which precedes the first discharge by the
   reaction time. An expert annotated dataset marks where an expert saw the signal depart
   from baseline, which follows the first discharge. Timing biases measured against those
   references are not comparable with the ones reported here, and the difference is larger
   than the differences between the detectors.

4. **The action potential shape is a single analytic function.** Real motor unit action
   potentials vary in shape as well as in amplitude and duration, and their shape at the skin
   depends on fibre depth and orientation. The Hermite Rodriguez function reproduces the
   duration and the spectral peak but not that variety, so any method whose performance
   depends on waveform shape rather than on amplitude or spectrum would be flattered here.

5. **Fatigue is simulated by one mechanism.** Slowing conduction velocity is the dominant
   mechanism behind the median frequency shift but not the only one: recruitment of additional
   units, increased discharge synchronisation, and changes in the intracellular action
   potential all contribute in a real contraction. The generator represents only the first.
   The demonstration therefore validates that the spectral features track a known spectral
   compression; it does not model fatigue physiology.

6. **The detectors are evaluated on one contraction shape.** A 20 ms excitation ramp
   represents a rapid contraction, which is the condition onset detectors are usually
   characterised under. A slow ramp inflates every detector's timing bias by the time the
   muscle takes to reach a detectable amplitude, which is a property of the contraction rather
   than of the detector, and the ratio between the detectors changes with it.

7. **No electrode or amplifier model.** Sampling is ideal, quantisation is not represented,
   there is no anti alias filter, and there is no amplifier saturation or direct current
   offset. A record from a real front end can fail in ways nothing here would predict.
