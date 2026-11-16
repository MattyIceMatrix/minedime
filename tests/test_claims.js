// Claim guards for the published page and the README: the countable statements must stay true.
// Run from the repo root:  node tests/test_claims.js
// Blast radius: reads index.html, README.md, AUDIT.md, SECURITY.md. Writes nothing. No network.
'use strict';
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const ROOT = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const readme = fs.readFileSync(path.join(ROOT, 'README.md'), 'utf8');

let fails = 0;
const ok = (name, cond, msg) => { console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}: ${msg}`); if (!cond) fails++; };

// 1. "13 tabs" - count them in the shipped page, and check every one has a panel to open
{
  const src = html.match(/\nconst TABS = (\[[\s\S]*?\]);\n/)[1];
  const TABS = eval(src);
  const keys = eval(html.match(/\nconst KEYS = (\[[\s\S]*?\]);\n/)[1]);
  const missing = TABS.filter(t => !html.includes(`id="p-${t[0]}"`)).map(t => t[0]);
  const claimed = Number((readme.match(/\*\*(\d+) tabs:\*\*/) || [])[1]);
  const named = (readme.match(/\*\*\d+ tabs:\*\*([\s\S]*?)\n-/)[1].match(/,/g) || []).length + 1;
  ok('13 tabs', TABS.length === 13 && claimed === 13 && keys.length === 13 && !missing.length && named === 13,
    `page defines ${TABS.length} tabs with ${keys.length} shortcut keys, README claims ${claimed} and names ${named}` +
    (missing.length ? `; tabs with no panel: ${missing}` : '; every tab has a panel'));
}

// 2. Content-Security-Policy: the pinned hashes must be the hashes of the scripts actually shipped.
//    An edit to either script silently breaks the live page unless this is checked.
{
  const csp = html.match(/<meta http-equiv="Content-Security-Policy" content="([^"]*)"/)[1];
  const pinned = (csp.match(/'sha256-[^']+'/g) || []).map(s => s.slice(8, -1));
  const scripts = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
  const actual = scripts.map(s => crypto.createHash('sha256').update(s, 'utf8').digest('base64'));
  const unmatched = actual.filter(h => !pinned.includes(h));
  const stale = pinned.filter(h => !actual.includes(h));
  ok('CSP pins the shipped scripts', scripts.length === pinned.length && !unmatched.length && !stale.length,
    `${scripts.length} inline scripts, ${pinned.length} pinned hashes, ` +
    (unmatched.length || stale.length ? `MISMATCH (${unmatched.length} unpinned, ${stale.length} stale)` : 'all match'));
  const needs = ["default-src 'none'", "connect-src 'none'", "base-uri 'none'", "form-action 'none'"];
  ok('CSP locks the page down', needs.every(d => csp.includes(d)),
    `directives present: ${needs.filter(d => csp.includes(d)).length}/${needs.length}`);
}

// 3. "no outside connections" / "Fonts are embedded in the page" - no remote URL anywhere in the page
{
  const urls = [...html.matchAll(/https?:\/\/[^\s"'<>)]+/g)].map(m => m[0]);
  const fonts = (html.match(/@font-face/g) || []).length;
  const dataFonts = (html.match(/src:url\(data:font\//g) || []).length;
  ok('no third-party requests', urls.length === 0 && fonts > 0 && dataFonts === fonts,
    urls.length ? `page references ${urls.length} remote URL(s): ${urls.slice(0, 3)}` :
      `0 remote URLs; all ${fonts} @font-face rules use embedded data: URIs`);
}

// 4. No credential of any kind in a public single-file app. The AI analyst must go through the
//    in-Claude sampling API, which carries no key, and must degrade to a note elsewhere.
{
  const bad = [/sk-[A-Za-z0-9]{20,}/, /AKIA[0-9A-Z]{16}/, /ghp_[A-Za-z0-9]{36}/,
  /api[_-]?key\s*[:=]\s*['"][^'"]{8,}/i, /Authorization\s*:\s*['"]Bearer/i,
  /BEGIN [A-Z ]*PRIVATE KEY/].filter(re => re.test(html));
  const usesSampling = /window\.claude\s*&&\s*window\.claude\.use/.test(html);
  const degrades = /Open this page inside Claude/.test(html);
  ok('no embedded credentials', !bad.length && usesSampling && degrades,
    bad.length ? `${bad.length} credential pattern(s) matched` :
      'no credential patterns; analyst uses window.claude.use("sample") and shows a note when absent');
}

// 5. README numbers that the code decides
{
  const oos = Number(html.match(/oos:\s*([\d.]+)/)[1]);
  const val = Number(html.match(/val:\s*([\d.]+)/)[1]);
  const dsrMin = Number(html.match(/dsrMin:\s*([\d.]+)/)[1]);
  const alpha = Number(html.match(/alpha:\s*([\d.]+)/)[1]);
  const claims = [
    ['sealed vault is the last 30%', oos === 0.3, /last 30% of history stays\s+sealed/.test(readme)],
    ['family-wise alpha is 5%', alpha === 0.05, /family-wise 5%\s+\(Bonferroni\)/.test(readme)],
    ['deflated-Sharpe threshold 0.95', dsrMin === 0.95, true],
  ];
  const broken = claims.filter(([, code, doc]) => !(code && doc)).map(([n]) => n);
  ok('README matches the engine config', !broken.length,
    broken.length ? `mismatched: ${broken}` : `oos ${oos}, validation ${val}, DSR threshold ${dsrMin}, alpha ${alpha}`);
}

// 6. The repository's own re-run instructions must point at files that exist
{
  const named = [...readme.matchAll(/`(?:python3|node) ([\w/.]+)`/g)].map(m => m[1]);
  const gone = named.filter(f => !fs.existsSync(path.join(ROOT, f)));
  const links = [...readme.matchAll(/\]\(([^)#:]+?)\)/g)].map(m => m[1]).filter(f => !f.startsWith('http'));
  const deadLinks = links.filter(f => !fs.existsSync(path.join(ROOT, f)));
  ok('README commands and links resolve', !gone.length && !deadLinks.length,
    gone.length || deadLinks.length ? `missing: ${[...gone, ...deadLinks]}` :
      `${named.length} commands and ${links.length} relative links all resolve`);
}

process.exit(fails ? 1 : 0);
