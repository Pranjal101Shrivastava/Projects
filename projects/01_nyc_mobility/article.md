# The dataset had no answer, so we asked a different question

*What 4,534,327 real New York taxi dispatches can and cannot tell you*

There is a widely-copied data science project that predicts NYC taxi trip duration. It
appears in portfolios, tutorials and course submissions. I wanted to build it too.

Then I opened the data.

```
Date/Time,Lat,Lon,Base
"4/1/2014 0:11:00",40.769,-73.9549,"B02512"
```

Four columns. A timestamp, a location, a dispatch base. **No duration. No fare.**

This is the real thing — the Taxi & Limousine Commission's response to a Freedom of
Information Law request, 4,534,327 genuine dispatched pickups. And it simply
does not contain the number that project claims to predict.

So how do all those projects predict it?

They generate it. A random number generator produces a plausible-looking duration, a model
learns to recover it, and the reported R² measures how well gradient boosting can reverse
engineer `numpy.random`. That is a fact about NumPy, not about New York.

## Asking what the data can answer

The data does support a real question: **how many pickups will this zone see next hour?**
That is countable directly from the records, and it is what a dispatcher actually needs — a
vehicle in the wrong place forfeits the fare, a vehicle in the right place earns it.

That reframing turned out to be the most interesting decision in the project.

## Yesterday is a worse guide than last week

Before building anything, I established what "doing nothing" achieves. Two baselines:
repeat the same hour yesterday, or repeat the same hour last week.

| Baseline | MAE |
|---|---:|
| Same hour yesterday | 35.36 |
| Same hour last week | 25.70 |

Last week wins, by a wide margin. Thursday at 6pm resembles *last* Thursday at 6pm far more
than it resembles Wednesday at 6pm. Weekday-versus-weekend is a bigger effect than
day-to-day drift.

The trained model confirmed it from the other direction: `lag_168h` accounts for
35.6% of its decision-making.

Final result: MAE 15.65, a 39.1% improvement
over the baseline anyone gets for free.

## The part that did not work

I attached conformal prediction intervals — a technique that gives coverage guarantees
without assuming a distribution. You calibrate on held-out residuals, take a quantile, and
the interval is supposed to cover at the nominal rate.

Here is what actually happened:

| Interval | Should cover | Actually covered |
|---|---:|---:|
| 80% | 80% | **71.2%** |
| 90% | 90% | 87.2% |
| 95% | 95% | 93.1% |

The 80% interval covers 71%. That is not a small miss.

The cause is in the data, and it is instructive. Conformal prediction assumes calibration
residuals and test residuals are *exchangeable* — drawn from the same distribution. Over
these six months, daily volume grew **82.9%**. Uber was scaling. My calibration residuals
came from a quieter period than my test residuals, so every error in the test window was
larger than the calibration set predicted. Wide bands had slack to absorb it. The tight one
did not.

I could have reported only the 90% and 95% intervals. Both are calibrated, and the table
would have looked clean.

But "this method has an assumption, my data violates it, and here is the measurable
consequence" is a more useful thing to know than "conformal prediction works". Techniques
have preconditions. Checking them is the job.

## And the 82,581 duplicates I kept

The data contains 82,581 exactly duplicated rows. Same minute, same
coordinates, same base.

Standard practice: drop them.

I kept every one. Timestamps are minute-resolution and coordinates are rounded to about 11
metres. Two genuinely different pickups, dispatched from the same base, in the same minute,
on the same block in Midtown are *identical in this schema* — and Midtown at 6pm dispatches
many vehicles per minute.

The decisive evidence is that duplicate share **rises with demand**. If they were data
faults you would expect them scattered uniformly. Instead they concentrate exactly where
many simultaneous dispatches are expected. They are collisions under coarse resolution, not
errors.

Dropping them would have understated demand precisely in the peak hours the model exists to
predict.

---

**Live demo:** [https://pranjal101shrivastava.github.io/Projects/#/p/nyc-mobility](https://pranjal101shrivastava.github.io/Projects/#/p/nyc-mobility)
**Code:** [github.com/Pranjal101Shrivastava/Projects](https://github.com/Pranjal101Shrivastava/Projects)

*Every figure above is read from a JSON artifact produced by the pipeline, not typed into the
article.*
