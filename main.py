"""Forecast GTA VI trophy earn rates on Exophase.

For every trophy in a case, this program fits a recency-weighted
least-squares line of logit(rate) against ln(1 + hours since the game's
release) over the case's history of reads, reads that line at the
target instant for the median, and widens the five quantiles with a
spread that combines a random-walk term growing with the horizon, the
fit's residual scatter, and a floor for Exophase's refresh jumps (the
page's counts move in discrete refreshes, not continuously).

Rates are percents (0-100). Logits use p = rate / 100 clamped to
[0.0001, 0.9999]. The forecast file is {"trophies": [{"id": n,
"q": [q05, q25, q50, q75, q95]}, ...]} with quantiles in percent,
non-decreasing, rounded to 4 decimals.

Standard library only; no network; deterministic (no randomness, so
two runs write identical bytes). Reads $MAKEFUN_INPUT_DIR (else
/input) and writes $MAKEFUN_OUTPUT_DIR (else /output), one .out file
per .in case file found, in name order.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# Random-walk volatility of a logit rate, per square-root day.
WALK = 0.5
# Floor on the logit spread: Exophase refreshes jump the rates.
SPREAD_FLOOR = 0.25
# Recency half-life for weighting reads, in hours.
HALF_LIFE_HOURS = 12.0
MAX_SLOPE = 3.0
MAX_SPREAD = 2.5


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp(value, low, high):
    return low if value < low else high if value > high else value


def logit(rate_percent):
    p = clamp(rate_percent / 100.0, 0.0001, 0.9999)
    return math.log(p / (1.0 - p))


def inv_logit_percent(y):
    if y >= 0:
        p = 1.0 / (1.0 + math.exp(-y))
    else:
        e = math.exp(y)
        p = e / (1.0 + e)
    return 100.0 * p


def weighted_line(points):
    """Weighted least squares y ~ a + b*x.

    points: list of (x, y, w). Returns (intercept, slope, resid_std,
    slope_se): the slope's standard error, so a long extrapolation from
    a noisy fit widens the forecast spread. Falls back to a flat line
    at the mean y when the fit is degenerate.
    """
    sw = sum(w for _, _, w in points)
    if sw <= 0 or not points:
        return 0.0, 0.0, 0.0, 0.0
    mx = sum(w * x for x, _, w in points) / sw
    my = sum(w * y for _, y, w in points) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in points)
    if sxx <= 1e-12:
        return my, 0.0, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in points)
    slope = sxy / sxx
    intercept = my - slope * mx
    resid_var = sum(w * (y - (intercept + slope * x)) ** 2
                    for x, y, w in points) / sw
    resid = math.sqrt(resid_var)
    dof = max(len(points) - 2, 1)
    slope_se = math.sqrt(resid_var / sxx * len(points) / dof) \
        if len(points) > 2 else 0.0
    return intercept, slope, resid, slope_se


def forecast_one(rates_over_time, target_dt, horizon_days):
    """rates_over_time: list of (datetime, rate_percent or None).

    Returns the five quantiles for the target instant.
    """
    usable = [(t, r) for t, r in rates_over_time if r is not None]
    if not usable:
        return [50.0, 50.0, 50.0, 50.0, 50.0]
    last_t = usable[-1][0]
    points = []
    for t, r in usable:
        hours = (t - RELEASE).total_seconds() / 3600.0
        x = math.log1p(max(hours, 0.0))
        age = (last_t - t).total_seconds() / 3600.0
        w = 0.5 ** (max(age, 0.0) / HALF_LIFE_HOURS)
        points.append((x, logit(r), w))
    intercept, slope, resid, slope_se = weighted_line(points)
    slope = clamp(slope, -MAX_SLOPE, MAX_SLOPE)
    target_hours = (target_dt - RELEASE).total_seconds() / 3600.0
    x_target = math.log1p(max(target_hours, 0.0))
    mean_x = (sum(w * x for x, _, w in points)
              / sum(w for _, _, w in points))
    y_med = intercept + slope * x_target
    y_med = clamp(y_med, logit(0.05), logit(99.5))
    spread = math.sqrt((WALK * math.sqrt(max(horizon_days, 0.0))) ** 2
                       + resid ** 2 + SPREAD_FLOOR ** 2
                       + (slope_se * abs(x_target - mean_x)) ** 2)
    spread = clamp(spread, 0.2, MAX_SPREAD)
    out = []
    prev = None
    for z in Z:
        q = clamp(inv_logit_percent(y_med + z * spread), 0.01, 99.99)
        if prev is not None and q < prev:
            q = prev
        out.append(round(q, 4))
        prev = q
    return out


def forecast_case(case):
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    history = case.get("history") or []
    rows = []
    for trophy in case["trophies"]:
        tid = trophy["id"]
        idx = tid - 1
        series = []
        for read in history:
            rates = read.get("rates") or []
            rate = rates[idx] if idx < len(rates) else None
            series.append((parse_time(read["at"]), rate))
        rows.append({"id": tid,
                     "q": forecast_one(series, target, horizon_days)})
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    try:
        names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    except OSError as exc:
        print("error: cannot list input directory: %s" % exc, file=sys.stderr)
        return 1
    if not names:
        print("error: no .in case file in %s" % in_dir, file=sys.stderr)
        return 1
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as handle:
            case = json.load(handle)
        forecast = forecast_case(case)
        out_name = name[:-3] + ".out"
        out_path = os.path.join(out_dir, out_name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > 65536:
            print("error: forecast for %s over 65,536 bytes" % name,
                  file=sys.stderr)
            return 1
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(text)
        print("wrote %s (%d trophies)" % (out_path, len(forecast["trophies"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
