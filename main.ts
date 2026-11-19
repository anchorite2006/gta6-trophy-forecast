// GTA VI trophy earn-rate forecaster (Node 22, standard library only).
//
// Forecasts, for every trophy in a case, five quantiles of its Exophase earn
// rate (percent of tracked players holding it) at the case's target instant.
// The model works in COUNT space, not rate space, because Exophase rates are
// ratios of two moving quantities: earners E(t) and players tracked N(t).
// A rate can fall while earners rise (dilution: new tracked players start
// with zero trophies) and rise while the player base stalls, so the two
// counts are forecast separately and combined only at the end:
//
//   rate(T) = 100 * E(T) / N(T)
//
// Median: the average, in logit space, of two views of the same trophy.
//   (A) A per-trophy logit trend: logit(rate) fitted by least squares
//       against x = ln(1 + hours since 2026-11-19T00:00Z), over the last
//       24 hours of reads only (the early launch-night rise is steep and
//       saturating; a whole-history line extrapolates it far past the
//       trophy's ceiling). Each trophy's slope is shrunk toward its tier's
//       mean slope, and the extrapolated increment is damped x0.85, because
//       earn-rate curves are concave in log time: players age into a
//       trophy, then the rate bends toward its ceiling. The trend is
//       anchored at the last observed rate, not at the fit intercept, so a
//       refresh jump in level cannot drag the forecast with it.
//   (B) A count trend: log(E + 0.5) fitted against the same x over the
//       same window, anchored at the last earner count, increment damped
//       x0.85, then divided by the forecast N(T). Earners can never exceed
//       players tracked, so E(T) is capped at 0.99 * N(T): an uncapped
//       log-count extrapolation is the single easiest way to forecast a
//       rate above 100% and wreck the median.
//   N(T) itself comes from a least-squares fit of log N against x over
//   the same window, anchored at the last N, increment damped x0.85
//   (launch growth decelerates), clamped to never fall below the last
//   observed N and never exceed 100x it.
//   When a fit has too few distinct points it degrades gracefully to
//   persistence (view A) or pure dilution (view B: earners frozen, N grows).
//
// Spread: a per-trophy logit sigma combining a random-walk term that grows
// with the square root of the horizon in days and the trophy's own logit-fit
// residual scale, floored at 0.35 and capped at 1.5 logits. Refreshes move
// every rate at once, so the floor is deliberately wide: a band that only
// covers smooth drift is worthless the first time Exophase refreshes.
//
// Edge cases:
//   - A trophy at 100% (every tracked player holds it) has survived
//     dilution so far; its median stays 99.99 with a narrow downside band.
//   - A trophy at 0% gets a half-earner pseudo-count median and a wide
//     upside band (it can only move up).
//   - Null rates in a read are skipped for that trophy only.
//   - Deterministic: no randomness, no clock reads; only the case file
//     decides the output, so two runs are byte-identical.
//
// I/O contract (from the competition rules): read every *.in file in
// $MAKEFUN_INPUT_DIR (else /input), in name order; write one <id>.out per
// case to $MAKEFUN_OUTPUT_DIR (else /output) as
// {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]}. Only mkdir + write;
// the output directory itself is never removed.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Quantile levels, in the order each output row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard normal quantiles at LEVELS. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Launch origin for the log-time axis (hours since this instant). */
const ORIGIN_MS = Date.parse('2026-11-19T00:00:00Z');
/** Random-walk spread, logit units per square-root day. */
const WALK = 0.35;
/** Sigma floor / cap, in logits. */
const SIGMA_MIN = 0.35;
const SIGMA_MAX = 1.5;

interface Trophy {
  id: number;
  tier: string;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface CaseFile {
  levels: number[];
  horizon_minutes: number;
  target: string;
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number, lo: number, hi: number): number =>
  Math.min(hi, Math.max(lo, x));
const logit = (pct: number): number => {
  const p = clamp(pct, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
};
const logistic = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
/** Log-time coordinate of a read: ln(1 + hours since the launch origin). */
const xOf = (ms: number): number =>
  Math.log(Math.max(0.05, 1 + (ms - ORIGIN_MS) / 3600000));

/** Least-squares line through (xs, ys); returns [intercept, slope]. */
function ols(xs: number[], ys: number[]): [number, number] {
  const n = xs.length;
  if (n === 0) return [0, 0];
  if (n === 1) return [ys[0], 0];
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx < 1e-12) return [my, 0];
  const slope = sxy / sxx;
  return [my - slope * mx, slope];
}

/** Sample standard deviation (0 when fewer than 2 values). */
function stdev(vals: number[]): number {
  const n = vals.length;
  if (n < 2) return 0;
  const m = vals.reduce((a, b) => a + b, 0) / n;
  const v = vals.reduce((a, b) => a + (b - m) * (b - m), 0) / (n - 1);
  return Math.sqrt(v);
}

function forecastCase(c: CaseFile): { id: number; q: number[] }[] {
  const targetMs = Date.parse(c.target);
  const xT = xOf(targetMs);
  const days = c.horizon_minutes / 1440;
  const nT = c.trophies.length;
  const hist = c.history ?? [];
  const last = hist[hist.length - 1];
  // Fitting window: the last 24h of reads (falls back to all history when
  // the case is younger than a day, e.g. the 90-minute practice cases).
  const cutMs = Date.parse(c.at);
  const windowStart = cutMs - 24 * 3600000;
  const win = hist.filter((r) => Date.parse(r.at) >= windowStart);
  const fitHist = win.length >= 3 ? win : hist;
  const xLast = last ? xOf(Date.parse(last.at)) : xT;
  const dx = Math.max(0, xT - xLast);

  // --- Players tracked N(T): log-linear fit over the window, anchored at
  // the last N, increment damped x0.85, clamped. ---
  let nForecast = last?.players_tracked ?? 0;
  {
    const xs: number[] = [];
    const ys: number[] = [];
    for (const r of fitHist) {
      if (typeof r.players_tracked === 'number' && r.players_tracked > 0) {
        xs.push(xOf(Date.parse(r.at)));
        ys.push(Math.log(r.players_tracked));
      }
    }
    if (xs.length >= 3 && last) {
      const [, b] = ols(xs, ys);
      // Extrapolate from the last read with a damped slope: launch growth
      // decelerates, and an undamped log-linear fit overshoots badly.
      const logN = Math.log(last.players_tracked) + 0.85 * b * dx;
      nForecast = Math.exp(logN);
      nForecast = clamp(nForecast, last.players_tracked, last.players_tracked * 100);
    } else if (last) {
      nForecast = last.players_tracked;
    }
    if (!(nForecast > 0) || !Number.isFinite(nForecast)) {
      nForecast = last?.players_tracked ?? 1000;
    }
  }

  // --- Per-trophy logit slopes over the window, then tier means. ---
  interface TrophyFit {
    slope: number;
    residSd: number;
    nPts: number;
    lastRate: number;
    lastEarners: number;
  }
  const fits: TrophyFit[] = [];
  for (let i = 0; i < nT; i++) {
    const xs: number[] = [];
    const ys: number[] = [];
    let lastRate = 1;
    let lastEarners = 0;
    for (const r of fitHist) {
      const rate = r.rates?.[i];
      if (typeof rate === 'number') {
        lastRate = rate;
        lastEarners = Math.round((rate / 100) * r.players_tracked);
      }
      if (typeof rate === 'number' && rate > 0 && rate < 100) {
        xs.push(xOf(Date.parse(r.at)));
        ys.push(logit(rate));
      }
    }
    // lastRate/lastEarners come from the whole history's last read.
    if (last) {
      const lr = last.rates?.[i];
      if (typeof lr === 'number') {
        lastRate = lr;
        lastEarners = Math.round((lr / 100) * last.players_tracked);
      }
    }
    const [a, b] = ols(xs, ys);
    const resid = xs.map((x, j) => ys[j] - (a + b * x));
    fits.push({ slope: b, residSd: stdev(resid), nPts: xs.length, lastRate, lastEarners });
  }
  const tierSlope = new Map<string, number>();
  for (const tier of new Set(c.trophies.map((t) => t.tier))) {
    const ss: number[] = [];
    c.trophies.forEach((t, i) => {
      if (t.tier === tier && fits[i].nPts >= 2) ss.push(fits[i].slope);
    });
    tierSlope.set(tier, ss.length > 0 ? ss.reduce((a, b) => a + b, 0) / ss.length : 0);
  }

  // --- Per-trophy forecast. ---
  return c.trophies.map((t, i) => {
    const f = fits[i];
    const rLast = f.lastRate;

    // Trophy held by everyone: median stays at the top, narrow band.
    if (rLast >= 99.99) {
      return { id: t.id, q: [99.0, 99.9, 99.99, 99.99, 99.99] };
    }

    // View A: shrunk logit trend, anchored at the last rate, increment
    // damped x0.85 for saturation.
    const w = f.nPts / (f.nPts + 4);
    const tierMean = tierSlope.get(t.tier) ?? 0;
    const slopeA = w * f.slope + (1 - w) * tierMean;
    const logitA = f.nPts >= 2 ? logit(rLast) + 0.85 * slopeA * dx : logit(rLast);

    // View B: count trend, anchored at the last earner count, increment
    // damped x0.85, earners capped at 99% of forecast players tracked.
    let rateB: number;
    {
      const xs: number[] = [];
      const ys: number[] = [];
      for (const r of fitHist) {
        const rate = r.rates?.[i];
        if (typeof rate === 'number' && rate >= 0 && rate <= 100) {
          const e = Math.round((rate / 100) * r.players_tracked);
          xs.push(xOf(Date.parse(r.at)));
          ys.push(Math.log(e + 0.5));
        }
      }
      if (xs.length >= 3) {
        const [, bB] = ols(xs, ys);
        const eT = Math.exp(Math.log(f.lastEarners + 0.5) + 0.85 * bB * dx) - 0.5;
        rateB = (100 * clamp(eT, 0.5, 0.99 * nForecast)) / nForecast;
      } else {
        // Pure dilution: earners frozen at the last count, N grows.
        rateB = (100 * Math.max(0.5, f.lastEarners)) / nForecast;
      }
    }

    // Median: mean of the two views in logit space.
    let medianLogit: number;
    let sigma: number;
    if (rLast <= 0.01) {
      // Never earned yet: half-earner pseudo-count median, wide upside.
      medianLogit = logit((100 * 0.5) / nForecast);
      sigma = 1.2;
    } else {
      medianLogit = 0.5 * (logitA + logit(rateB));
      const walk = WALK * Math.sqrt(days);
      const resid = clamp(f.residSd, 0.15, 0.6);
      sigma = clamp(Math.sqrt(walk * walk + resid * resid), SIGMA_MIN, SIGMA_MAX);
    }
    const q = Z.map((z) => round4(clamp(logistic(medianLogit + z * sigma), 0.001, 99.999)));
    // Enforce non-decreasing (rounding can tie, never invert, but be safe).
    for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (!Array.isArray(c.trophies) || !Array.isArray(c.history)) {
    throw new Error(`${file}: missing trophies/history`);
  }
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
