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
// 2. tribunal calibration: planted-edge markets ship, zero-edge markets don't
{ let shipReal = 0, shipTrap = 0; const run = (seed, edge) => { const F = new L.Forge(L.simulate(2520, seed, edge), { seed: seed + 3 }); F.init(); for (let g = 0; g < F.cfg.gens; g++) F.step(); return F.tribunal(F.hallOfFame()).verdicts.some(v => v.pass); };
  for (let s = 0; s < 3; s++) { shipReal += run(500 + s * 17, 1); shipTrap += run(503 + s * 17, 0); }
  ok('tribunal calibration', shipReal >= 2 && shipTrap === 0, `planted edges shipped ${shipReal}/3, zero-edge markets shipped ${shipTrap}/3`); }
process.exit(fails ? 1 : 0);
