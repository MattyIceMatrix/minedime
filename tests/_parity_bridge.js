// Helper for tests/test_parity.py: run the website's tribunal statistics on inputs handed
// over as JSON on stdin and write the results as JSON on stdout. Nothing else.
// Blast radius: reads index.html and stdin; writes stdout only. No network, no files written.
'use strict';
const fs = require('fs'), path = require('path');
const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
new Function(html.match(/<script>([\s\S]*?)<\/script>/)[1])();
const L = globalThis.AlphaLab;
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const out = { dsr: [], pbo: [], eff: [], sharpe: [], eta: [] };
for (const c of input.dsr || []) {
  const d = L.deflatedSharpe(Float64Array.from(c.pnl), c.n);
  out.dsr.push({ dsr: d.dsr, bar: d.bar, eta: d.eta });
}
for (const m of input.pbo || []) out.pbo.push(L.pbo(m.map(col => Float64Array.from(col))));
for (const m of input.eff || []) out.eff.push(L.effectiveTrials(m.map(col => Float64Array.from(col)), L.RNG(0)));
for (const c of input.sharpe || []) out.sharpe.push(L.sharpe(Float64Array.from(c)));
process.stdout.write(JSON.stringify(out));
