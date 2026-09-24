// Site engine audit. Run from the repo root:  node tests/test_site.js
// Blast radius: reads index.html; pure in-memory computation. No network, no files written.
const fs = require('fs'), path = require('path');
const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const engine = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new Function(engine)(); const L = globalThis.AlphaLab;
let fails = 0; const ok = (name, cond, msg) => { console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}: ${msg}`); if (!cond) fails++; };
// 1. look-ahead: scramble after CUT (inside the search window); positions on or before CUT must not move
{ const D = L.simulate(600, 9, 1), T = D.T, N = D.N, CUT = 220, E = { T, N, _fwd: D._fwd }, r = L.RNG(123);
  for (const k of ['close', 'open', 'high', 'low', 'volume', 'vwap', 'returns']) { E[k] = Float64Array.from(D[k]); for (let t = CUT + 1; t < T; t++) for (let i = 0; i < N; i++) E[k][t * N + i] *= 0.5 + r(); }
  let leaks = 0, live = 0, n = 0;
  for (let s = 0; s < 300; s++) { const f = new L.Forge(D, { pop: 1, seed: 1000 + s }), g = new L.Forge(E, { pop: 1, seed: 1000 + s }); f.init(); g.init();
    const a = f.pop[0], b = g.pop[0]; if (!a || !b || a.expr !== b.expr) continue; n++;
    const Wa = f.weightsFor({ ...a, flip: false }), Wb = g.weightsFor({ ...b, flip: false });
    let m = false; for (let k = 0; k < (CUT + 1) * N; k++) if (Math.abs(Wa[k] - Wb[k]) > 1e-12) { m = true; break; } leaks += m;
    for (let k = (CUT + 2) * N; k < Wa.length; k++) if (Math.abs(Wa[k] - Wb[k]) > 1e-12) { live++; break; } }
  ok('no look-ahead', leaks === 0 && live > 100, `${n} formulas, ${leaks} leaks (${live} reacted after the cutoff, so the test is live)`); }
// 2. re-simulation: a longer history must not change any earlier day (catches a simulator whose past depends on its future)
{ const A = L.simulate(600, 9, 1), B = L.simulate(700, 9, 1), C = L.simulate(600, 10, 1), same = (x, y, n) => { for (let k = 0; k < n; k++) if (!(x[k] === y[k] || (x[k] !== x[k] && y[k] !== y[k]))) return false; return true; };
  const moved = ['close', 'open', 'high', 'low', 'volume', 'vwap', 'returns', '_fwd'].filter(f => !same(A[f], B[f], (f === '_fwd' ? 598 : 599) * A.N));
  ok('simulator has no future', !moved.length && !same(A.close, C.close, 599 * A.N), moved.length ? `fields changed on earlier days: ${moved}` : 'extending the history by 100 days changes nothing about the first 599 days (control: a different seed does)'); }
// 3. rolling operators vs an exact two-pass reference, including a spike, an Infinity and flat stretches
{ const fin = Number.isFinite, O = L.OPS;
  const ref = (op, x, y, T, N, d) => { const o = new Float64Array(T * N).fill(NaN);
    for (let i = 0; i < N; i++) for (let t = d - 1; t < T; t++) { const w = [], z = []; for (let k = t - d + 1; k <= t; k++) { w.push(x[k * N + i]); if (y) z.push(y[k * N + i]); }
      if (w.some(v => !fin(v)) || z.some(v => !fin(v))) continue;
      const m = w.reduce((a, b) => a + b) / d, v = w.reduce((a, b) => a + (b - m) ** 2, 0) / d, sd = Math.sqrt(v) <= 2e-7 * Math.abs(m) ? 0 : Math.sqrt(v);
      if (op === 'ts_mean') o[t * N + i] = m; else if (op === 'ts_std') o[t * N + i] = sd; else if (op === 'ts_zscore') o[t * N + i] = sd > 0 ? (x[t * N + i] - m) / sd : NaN;
      else { const my = z.reduce((a, b) => a + b) / d, vy = z.reduce((a, b) => a + (b - my) ** 2, 0) / d, c = w.reduce((a, b, k) => a + (b - m) * (z[k] - my), 0) / d;
        o[t * N + i] = sd > 0 && Math.sqrt(vy) > 2e-7 * Math.abs(my) ? c / Math.sqrt(v * vy) : NaN; } } return o; };
  const D = L.simulate(700, 4, 1), T = D.T, N = D.N, spiky = Float64Array.from(D.returns); spiky[300 * N + 2] = 1e9; spiky[301 * N + 5] = Infinity; spiky[450 * N + 7] = -1e12;
  const flat = O.ts_max[2](O.ts_max[2](D.close, T, N, 20), T, N, 40); let bad = 0, total = 0, worst = '';
  for (const [nm, x] of [['returns', D.returns], ['close', D.close], ['spiky', spiky], ['flat', flat]]) for (const op of ['ts_mean', 'ts_std', 'ts_zscore', 'ts_corr']) for (const d of [3, 20, 60]) {
    const got = op === 'ts_corr' ? O[op][2](x, D.vwap, T, N, d) : O[op][2](x, T, N, d), exp = ref(op, x, op === 'ts_corr' ? D.vwap : null, T, N, d); total++;
    let off = 0; for (let k = 0; k < got.length; k++) { const a = got[k], b = exp[k]; if (!fin(a) && !fin(b)) continue; if (!(fin(a) && fin(b)) || Math.abs(a - b) > 1e-6 * Math.max(1e-9, Math.abs(b)) + 1e-9) off++; }
    if (off) { bad++; worst = `${op}(${nm},${d}): ${off} values off`; } }
  ok('rolling operators exact', bad === 0, bad ? `${bad}/${total} combinations disagree, e.g. ${worst}` : `${total} operator/input/window combinations match an exact two-pass reference, spikes and flat stretches included`); }
// 4. weights: dollar-neutral every day; rounding noise gets no position; ties rank equally
{ const D = L.simulate(400, 5, 1), sig = Float64Array.from(D.close, (v, k) => Math.log(v) + (k % 7)), W = L.weights(sig, D); let worst = 0;
  for (let t = 0; t < D.T; t++) { let net = 0, g = 0; for (let i = 0; i < D.N; i++) { net += W[t * D.N + i]; g += Math.abs(W[t * D.N + i]); } if (g) worst = Math.max(worst, Math.abs(net) / g); }
  const noise = Float64Array.from({ length: D.T * D.N }, (_, k) => 0.3 + 1e-17 * (k % D.N)), Wn = L.weights(noise, D);
  const r = L.OPS.cs_rank[2](Float64Array.from([1, 1, 1, 2]), 1, 4);
  ok('dollar-neutral, noise-free weights', worst < 1e-12 && Wn.every(v => v === 0) && r[0] === r[1] && r[1] === r[2] && r[3] === 0.5,
    `max net exposure ${(worst * 100).toExponential(1)}% of gross; rounding-noise signal gets no position; tied inputs share a rank`); }
// 5. eigenvalues of a known matrix: tridiagonal (2, -1) has eigenvalues 2 - 2cos(k*pi/(n+1))
{ const n = 60, A = new Float64Array(n * n); for (let i = 0; i < n; i++) { A[i * n + i] = 2; if (i) A[i * n + i - 1] = A[(i - 1) * n + i] = -1; }
  const got = L.symEig(A, n).sort((a, b) => a - b), exp = Array.from({ length: n }, (_, k) => 2 - 2 * Math.cos((k + 1) * Math.PI / (n + 1))).sort((a, b) => a - b);
  const err = Math.max(...got.map((v, k) => Math.abs(v - exp[k]))); ok('eigen-solver', err < 1e-10, `max error ${err.toExponential(1)} on a 60x60 matrix with known eigenvalues`); }
// 6. tribunal calibration: planted-edge markets ship, zero-edge markets don't
{ let shipReal = 0, shipTrap = 0; const run = (seed, edge) => { const F = new L.Forge(L.simulate(2520, seed, edge), { seed: seed + 3 }); F.init(); for (let g = 0; g < F.cfg.gens; g++) F.step(); return F.tribunal(F.hallOfFame()).verdicts.some(v => v.pass); };
  for (let s = 0; s < 3; s++) { shipReal += run(500 + s * 17, 1); shipTrap += run(503 + s * 17, 0); }
  ok('tribunal calibration', shipReal >= 2 && shipTrap === 0, `planted edges shipped ${shipReal}/3, zero-edge markets shipped ${shipTrap}/3`); }
process.exit(fails ? 1 : 0);
