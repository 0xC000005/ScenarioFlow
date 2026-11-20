// ScenarioFlow explainer, ported for the website from the explainer video's animation source.
// Drawing maths, data and wording are verbatim; only the plumbing differs:
// render(t, opts) returns the SVG markup instead of writing #stage, data globals bind lazily, captions can be
// recorded (SFStory.captions) or omitted, ids can be prefixed, and the closing line can be set for the web.
(function () {
'use strict';
// ScenarioFlow — 4:5 explainer, v16 (v15 + show-don't-tell beats: attention bar, nearest-memory lines, tilting correlation line, typed close)
// was v15 (v14 + network told in six plain beats: dots are days, lines are real attention)
// was v14 (v13 + network as a red wedge into a red circle: attention cones, radial 128-number memory)
// was v13 (v12 + refined network, crossfades with one red wipe, uncropped fans, closing echo)
// was v12 (v8 + typed view and a shifting distribution on the title page; flow matching shown on one S&P–VIX plane, no attention row or rollout)
// was v8 (v7 with the fan switched to the words-conditioned run; scatter labelled market-state-only). Strict red / black / white, constructivist geometry, real model output.
// render(t) is pure: the same t always draws the same frame (the renderer steps t).
let D = window.DATA;
const W = 1080, H = 1350, DUR = 89.0;
const K = { black: '#000000', white: '#FFFFFF', red: '#E81132', grey: '#8C8C8C', mute: '#5C5C5C', rule: '#DADADA' };
const F = 'font-family="Inter"';
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const io = x => { x = clamp(x); return x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
const out = x => 1 - Math.pow(1 - clamp(x), 3);
const lin = x => clamp(x);
const P = (t, a, b, e = io) => e((t - a) / (b - a));
const lerp = (a, b, k) => a + (b - a) * k;
const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
const pts = a => a.map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' ');

const S = { s1: 0, s2: 7.0, sD: 15.5, s3: 15.5, s4: 25.0, s5: 33.0, s6: 47.0, s7: 67.0, s8: 82.5 };  // sD unused
const SF = S.s7;   // flow-matching scene start (the ported v11 scene reads S.flow / S.mem)

// 45° hatches keep fills in pure ink instead of pink/grey washes
const DEFS = `<defs>
  <pattern id="hR" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="3.2" height="9" fill="${K.red}"/></pattern>
  <pattern id="hG" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="2.2" height="9" fill="${K.grey}"/></pattern>
  <pattern id="hW" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="2.2" height="9" fill="#9A9A9A"/></pattern>
</defs>`;

function wipe(t, at) {
  const k = (t - (at - .35)) / .7; if (k <= 0 || k >= 1) return '';
  const x = lerp(-900, W + 300, io(k));
  return `<polygon points="${x},0 ${x + 520},0 ${x + 100},${H} ${x - 420},${H}" fill="${K.red}"/>`;
}
function caption(t, a, b, text, col) {
  if (REC) { REC.push({ t0: a, t1: b, text, col }); return ''; }
  if (!CAPS) return '';
  const o = Math.min(P(t, a, a + .4), 1 - P(t, b - .3, b)); if (o <= 0) return '';
  return `<text x="72" y="1282" ${F} font-size="32" font-weight="600" fill="${col}" opacity="${o}">${esc(text)}</text>`;
}

// ---------- 1 · hook: the future blooms out of today, cut by a hard diagonal ----------
function scene1(t) {
  let s = `<rect width="${W}" height="${H}" fill="${K.black}"/>`;
  // headline is at full contrast from the very first frame
  s += `<g ${F} font-weight="800" letter-spacing="-3">
    <text x="64" y="190" font-size="96" fill="${K.white}">Describe the</text>
    <text x="64" y="292" font-size="96" fill="${K.white}">market in words.</text>
    <text x="64" y="${412 + 14 * (1 - P(t, .8, 1.5))}" font-size="96" fill="${K.red}" opacity="${P(t, .8, 1.5)}">See its future</text>
    <text x="64" y="${514 + 14 * (1 - P(t, .8, 1.5))}" font-size="96" fill="${K.red}" opacity="${P(t, .8, 1.5)}">distribution.</text></g>`;
  // the view, typed (the live demo's real prompt)
  const story = D.story, n = Math.round(story.length * P(t, .4, 2.8, lin));
  const words = story.slice(0, n).split(' '), lines = []; let cur = '';
  for (const w of words) { if ((cur + ' ' + w).trim().length > 52) { lines.push(cur.trim()); cur = w; } else cur += ' ' + w; } if (cur.trim()) lines.push(cur.trim());
  if (!lines.length) lines.push('');
  lines.forEach((ln, i) => { s += `<text x="72" y="${600 + i * 40}" ${F} font-size="30" font-style="italic" fill="#D0D0D0">${esc(ln)}${i === lines.length - 1 && t < 3.0 && Math.floor(t * 2.4) % 2 === 0 ? '|' : ''}</text>`; });
  // gold's 30-day range: market state only (grey), then with the words (red)
  const m = D.markets[7], [lo, hi] = scaleFor(m), Hn = m.hist.length, gx = 72, gw = 880, gy = 812, gh = 320;
  const X = d => gx + gw * (d + Hn - 1) / (Hn - 1 + 30), Y = v => gy + gh * (1 - (v - lo) / (hi - lo)), k = P(t, 3.1, 4.8), ca = P(t, .2, .7);
  s += `<g opacity="${ca}"><text x="72" y="${gy - 22}" ${F} font-size="28" font-weight="700" fill="${K.white}">Gold, next 30 days from 20 Jan 2016</text>
    <line x1="${X(0)}" y1="${gy}" x2="${X(0)}" y2="${gy + gh}" stroke="#333" stroke-width="3"/>
    <defs><clipPath id="t1c"><rect x="${gx}" y="${gy - 8}" width="${gw}" height="${gh + 16}"/></clipPath></defs><g clip-path="url(#t1c)">`;
  s += morphFan(m, k, X, Y, 1, { path: 1.4, med: 4.5, grey: 'hW', greyLine: '#B0B0B0' });
  if (k > 0) ['p90', 'p10'].forEach(key => { s += `<polyline fill="none" stroke="#B0B0B0" stroke-width="2.5" opacity="${P(t, 3.1, 3.6)}" points="${pts(m.base[key].map((v, d) => [X(d), Y(v)]))}"/>`; });
  s += `</g><polyline fill="none" stroke="${K.white}" stroke-width="5" stroke-linejoin="round" points="${pts(m.hist.map((v, i) => [X(i - Hn + 1), Y(v)]))}"/></g>`;
  s += `<g ${F} font-size="26" opacity="${ca}"><rect x="72" y="${gy + gh + 30}" width="30" height="16" fill="url(#hW)" stroke="#B0B0B0" stroke-width="2"/><text x="112" y="${gy + gh + 46}" fill="#BDBDBD">market state only</text>
    <rect x="390" y="${gy + gh + 30}" width="30" height="16" fill="url(#hR)" opacity="${P(t, 3.1, 3.5)}"/><text x="430" y="${gy + gh + 46}" fill="${K.white}" opacity="${P(t, 3.1, 3.5)}">+ your words</text></g>`;
  s += `<text x="72" y="1282" ${F} font-size="32" font-weight="800" fill="${K.white}">ScenarioFlow</text>`;
  return s;
}

// ---------- 2 · the gap: a red link from words to equations, cut by a black slab ----------
function scene2(t) {
  const u = t - S.s2;
  let s = `<rect width="${W}" height="${H}" fill="${K.white}"/>`;
  s += `<g ${F}>
    <text x="72" y="150" font-size="44" font-weight="600" fill="${K.black}" opacity="${P(u, .1, .6)}">Risk is discussed in</text>
    <text x="64" y="270" font-size="128" font-weight="800" letter-spacing="-4" fill="${K.black}" opacity="${P(u, .2, .7)}">words,</text>
    <text x="1008" y="760" text-anchor="end" font-size="44" font-weight="600" fill="${K.black}" opacity="${P(u, 1.4, 1.9)}">but simulated in</text>
    <text x="1016" y="880" text-anchor="end" font-size="128" font-weight="800" letter-spacing="-4" fill="${K.red}" opacity="${P(u, 1.5, 2.0)}">equations.</text></g>`;
  // the words themselves
  const q = '“Dollar funding is tight; credit is cracking.”', n = Math.round(q.length * P(u, .6, 1.9, lin));
  const qs = q.slice(0, n), cut = q.indexOf(';') + 1, l1 = qs.slice(0, cut), l2 = qs.slice(cut).trimStart();
  s += `<text x="72" y="370" ${F} font-size="44" font-style="italic" font-weight="500" fill="${K.black}">${esc(l1)}</text>`;
  if (l2) s += `<text x="72" y="428" ${F} font-size="44" font-style="italic" font-weight="500" fill="${K.black}">${esc(l2)}</text>`;
  // the equation
  const e = P(u, 2.0, 2.5);
  s += `<text x="1008" y="1000" text-anchor="end" ${F} font-size="60" fill="${K.black}" opacity="${e}"><tspan font-style="italic">dS</tspan> = <tspan font-style="italic">μS</tspan> d<tspan font-style="italic">t</tspan> + <tspan font-style="italic">σS</tspan> d<tspan font-style="italic">W</tspan></text>`;
  s += `<text x="1008" y="1048" text-anchor="end" ${F} font-size="28" fill="${K.mute}" opacity="${e}">one equation per asset · its form fixed by hand</text>`;
  // red link from the end of the words to the equation …
  const x0 = 140, y0 = 480, x1 = 860, y1 = 690, r = P(u, 2.6, 3.3), sl = P(u, 3.3, 3.65, out), fall = P(u, 3.6, 4.6);
  const xm = lerp(x0, x1, .5), ym = lerp(y0, y1, .5), gap = 46;
  const dx = (x1 - x0), dy = (y1 - y0), L = Math.hypot(dx, dy), ux = dx / L, uy = dy / L;
  if (r > 0) {
    const xe = lerp(x0, x1, r), ye = lerp(y0, y1, r), cutx = xm - ux * gap / 2, cuty = ym - uy * gap / 2;
    const firstEnd = r < .5 || sl === 0 ? [xe, ye] : [cutx, cuty];
    s += `<line x1="${x0}" y1="${y0}" x2="${firstEnd[0]}" y2="${firstEnd[1]}" stroke="${K.red}" stroke-width="22"/>`;
    if (r > .5 && sl > 0) s += `<g opacity="${1 - fall}" transform="translate(${60 * fall},${300 * fall * fall}) rotate(${20 * fall} ${xm} ${ym})"><line x1="${xm + ux * gap / 2}" y1="${ym + uy * gap / 2}" x2="${xe}" y2="${ye}" stroke="${K.red}" stroke-width="22"/></g>`;
  }
  // … and a black slab drives through it
  if (sl > 0) {
    const off = lerp(700, 0, sl);
    s += `<polygon points="${xm - 10 + off},${ym - 130 - off} ${xm + 62 + off},${ym - 130 - off} ${xm - 4 + off},${ym + 120 - off} ${xm - 76 + off},${ym + 120 - off}" fill="${K.black}"/>`;
  }
  // the status quo, in one line each
  [['Prescribe the dynamics', 'only what you assumed'], ['Replay history', 'only what already happened']].forEach((rw, i) => {
    const a = P(u, 4.4 + i * .3, 4.9 + i * .3), y = 1160 + i * 62;
    s += `<g opacity="${a}"><rect x="72" y="${y - 36}" width="16" height="46" fill="${K.red}"/>
      <text x="110" y="${y}" ${F} font-size="34" font-weight="700" fill="${K.black}">${rw[0]}<tspan font-weight="400" fill="${K.mute}">  ${rw[1]}</tspan></text></g>`;
  });
  return s;
}

// ---------- fan helpers ----------
function scaleFor(m, padK = .03) {
  const all = [...m.hist, ...m.base.p10, ...m.base.p90, ...m.cond.p10, ...m.cond.p90, ...m.base.paths.flat(), ...m.cond.paths.flat()];
  let lo = Math.min(...all), hi = Math.max(...all); const p = (hi - lo) * padK; return [lo - p, hi + p];
}
// k: 0 = market state only (grey hatch), 1 = market state + words (red hatch); grow: share of the 30 days revealed
function morphFan(m, k, X, Y, grow, o) {
  const days = Math.max(1, Math.round(30 * grow));
  const q = key => m.base[key].map((v, d) => lerp(v, m.cond[key][d], k));
  const up = q('p90').slice(0, days + 1).map((v, d) => [X(d), Y(v)]), dn = q('p10').slice(0, days + 1).map((v, d) => [X(d), Y(v)]).reverse();
  const band = `${pts(up)} ${pts(dn)}`;
  let s = `<polygon points="${band}" fill="url(#${o.grey || 'hG'})" opacity="${1 - k}"/>`;
  s += `<polygon points="${band}" fill="url(#hR)" opacity="${k}"/>`;
  const col = k < .5 ? (o.greyLine || K.grey) : K.red;
  m.base.paths.forEach((bp, i) => {
    const cp = m.cond.paths[i] || bp;
    s += `<polyline fill="none" stroke="${col}" stroke-width="${o.path}" opacity=".7" points="${pts(bp.slice(0, days + 1).map((v, d) => [X(d), Y(lerp(v, cp[d], k))]))}"/>`;
  });
  s += `<polyline fill="none" stroke="${col}" stroke-width="${o.med}" points="${pts(q('p50').slice(0, days + 1).map((v, d) => [X(d), Y(v)]))}"/>`;
  return s;
}


// ---------- 2b · how ScenarioFlow differs from the two status-quo tools ----------
function sceneDiff(t) {
  const u = t - S.sD;
  let s = `<rect width="${W}" height="${H}" fill="${K.black}"/>`;
  s += `<text x="64" y="196" ${F} font-size="92" font-weight="800" letter-spacing="-3" fill="${K.white}" opacity="${P(u, .1, .6)}">ScenarioFlow</text>`;
  s += `<text x="64" y="296" ${F} font-size="92" font-weight="800" letter-spacing="-3" fill="${K.red}" opacity="${P(u, .3, .8)}">learns it instead.</text>`;
  const cols = [['Dynamics learned,', 'not assumed'], ['New paths,', 'not replays'], ['Takes', 'words']], CX = [622, 812, 952];
  s += `<g ${F} font-size="21" text-anchor="middle" fill="#BDBDBD" opacity="${P(u, .8, 1.2)}">` + cols.map((c, k) => `<text x="${CX[k]}" y="430">${c[0]}</text><text x="${CX[k]}" y="458">${c[1]}</text>`).join('') + `</g>`;
  const rows = [['Prescribe the dynamics', 'SDEs, GARCH', [0, 1, 0]], ['Replay history', 'historical simulation', [0, 0, 0]], ['ScenarioFlow', 'one learned model, 39 markets', [1, 1, 1]]];
  rows.forEach((r, i) => {
    const y = 560 + i * 150, a = P(u, 1.2 + i * .9, 1.7 + i * .9), sf = i === 2;
    s += `<g opacity="${a}"><rect x="72" y="${y - 64}" width="936" height="2" fill="#333"/>`;
    if (sf) s += `<rect x="72" y="${y - 62}" width="${936 * P(u, 3.0, 3.6)}" height="132" fill="${K.red}"/>`;
    s += `<text x="72" y="${y}" ${F} font-size="38" font-weight="800" fill="${K.white}">${r[0]}</text>
      <text x="72" y="${y + 40}" ${F} font-size="24" fill="${sf ? K.white : '#8C8C8C'}">${r[1]}</text>`;
    r[2].forEach((ok, k) => {
      const m = P(u, 1.5 + i * .9 + k * .15, 1.8 + i * .9 + k * .15), x = CX[k], yc = y - 8;
      s += ok ? `<path d="M${x - 18},${yc} L${x - 5},${yc + 14} L${x + 20},${yc - 16}" fill="none" stroke="${K.white}" stroke-width="7" stroke-linecap="square" opacity="${m}"/>`
              : `<g stroke="#5C5C5C" stroke-width="5" opacity="${m}"><line x1="${x - 14}" y1="${yc - 14}" x2="${x + 14}" y2="${yc + 14}"/><line x1="${x + 14}" y1="${yc - 14}" x2="${x - 14}" y2="${yc + 14}"/></g>`;
    });
    s += `</g>`;
  });
  s += `<text x="72" y="1100" ${F} font-size="40" font-weight="800" fill="${K.white}" opacity="${P(u, 4.2, 4.7)}">Learned from data. Conditioned on your words.</text>`;
  s += caption(t, S.sD + .8, S.sD + 4.0, 'Today’s tools assume the dynamics or replay the past.', '#BDBDBD');
  s += caption(t, S.sD + 4.1, S.s3 - .1, 'ScenarioFlow learns how 39 markets move together.', K.red);
  return s;
}

// ---------- 3 · hero: one market reshapes under the words ----------
function scene3(t) {
  const u = t - S.s3;
  let s = `<rect width="${W}" height="${H}" fill="${K.white}"/>`;
  const story = D.story, n = story.length;   // already typed on the title page
  const words = story.slice(0, n).split(' '), lines = []; let cur = '';
  for (const w of words) { if ((cur + ' ' + w).trim().length > 42) { lines.push(cur.trim()); cur = w; } else cur += ' ' + w; } if (cur.trim()) lines.push(cur.trim());
  s += `<rect x="0" y="0" width="${W}" height="352" fill="${K.black}"/>`;
  s += `<text x="72" y="92" ${F} font-size="30" font-weight="700" fill="${K.red}">Your words</text>`;
  lines.forEach((ln, i) => { s += `<text x="72" y="${156 + i * 48}" ${F} font-size="38" font-style="italic" fill="${K.white}">${esc(ln)}${''}</text>`; });
  // latent-space matches
  const ta = P(u, .6, .9);
  s += `<text x="72" y="416" ${F} font-size="28" fill="${K.mute}" opacity="${ta}">Past markets matched to your words (cosine similarity)</text>`;
  [['11 Feb 2015', '0.81'], ['3 Nov 2014', '0.81'], ['19 Apr 2013', '0.79']].forEach((p, i) => {
    const a = P(u, .7 + i * .3, 1.1 + i * .3, out), x = 72 + i * 316;
    s += `<g opacity="${a}"><rect x="${x}" y="436" width="300" height="62" fill="${K.black}"/><rect x="${x}" y="436" width="10" height="62" fill="${K.red}"/>
      <text x="${x + 26}" y="477" ${F} font-size="29" font-weight="700" fill="${K.white}">${p[0]}<tspan font-weight="400" fill="#BDBDBD" dx="10">${p[1]}</tspan></text></g>`;
  });
  // hero chart (compressed so the number can own the bottom band)
  const m = D.markets[6], [lo, hi] = scaleFor(m), Hn = m.hist.length;
  const gx = 72, gw = 856, gy = 590, gh = 420;  // right 80px hold the day-30 width brackets
  const X = d => gx + gw * (d + Hn - 1) / (Hn - 1 + 30), Y = v => gy + gh * (1 - (v - lo) / (hi - lo));
  const ca = P(u, 0, .4);
  s += `<g opacity="${ca}"><text x="72" y="566" ${F} font-size="36" font-weight="800" fill="${K.black}">Crude oil, next 30 days from 20 Jan 2016</text>`;
  s += `<line x1="${X(0)}" y1="${gy}" x2="${X(0)}" y2="${gy + gh}" stroke="${K.rule}" stroke-width="3"/>`;
  s += `<defs><clipPath id="hc"><rect x="${gx}" y="${gy - 10}" width="${gw}" height="${gh + 20}"/></clipPath></defs><g clip-path="url(#hc)">`;
  const grow = P(u, .2, 1.4, lin), k = P(u, 2.4, 4.4);
  if (u > .2) s += morphFan(m, k, X, Y, grow, { path: 2.2, med: 7 });
  // ghost: the words-free band's edges stay behind once it turns red
  const gh_o = P(u, 2.4, 3.0);
  if (gh_o > 0) ['p90', 'p10'].forEach(key => {
    s += `<polyline fill="none" stroke="${K.grey}" stroke-width="3.5" opacity="${gh_o}" points="${pts(m.base[key].map((v, d) => [X(d), Y(v)]))}"/>`;
  });
  s += `</g><polyline fill="none" stroke="${K.black}" stroke-width="7" stroke-linejoin="round" points="${pts(m.hist.map((v, i) => [X(i - Hn + 1), Y(v)]))}"/></g>`;
  // day-30 width brackets: grey = market state only, red = + words; red length tracks the ×-counter
  const bracket = (x, y0, y1, col, o) => `<g stroke="${col}" stroke-width="5" opacity="${o}"><line x1="${x}" y1="${y0}" x2="${x}" y2="${y1}"/><line x1="${x - 10}" y1="${y0}" x2="${x + 10}" y2="${y0}"/><line x1="${x - 10}" y1="${y1}" x2="${x + 10}" y2="${y1}"/></g>`;
  s += bracket(X(30) + 30, Y(m.base.p90[30]), Y(m.base.p10[30]), K.grey, P(u, 1.2, 1.6) * ca);
  if (u > 2.4) {
    const kc = P(u, 2.6, 4.6);
    s += bracket(X(30) + 64, Y(lerp(m.base.p90[30], m.cond.p90[30], kc)), Y(lerp(m.base.p10[30], m.cond.p10[30], kc)), K.red, P(u, 2.4, 2.8));
  }
  // legend, 28px floor
  const l1 = P(u, .2, .6), l2 = P(u, 2.4, 2.8);
  s += `<g ${F} font-size="28"><rect x="72" y="1036" width="34" height="18" fill="url(#hG)" stroke="${K.grey}" stroke-width="2.5" opacity="${l1}"/><text x="118" y="1054" fill="${K.mute}" opacity="${l1}">market state only</text>
    <rect x="420" y="1036" width="34" height="18" fill="url(#hR)" opacity="${l2}"/><text x="466" y="1054" fill="${K.black}" opacity="${l2}">+ your words</text></g>`;
  // the number owns the bottom band; it starts counting as the morph begins
  if (u > 2.4) {
    const r = 1 + (m.ratio - 1) * P(u, 2.6, 4.6), a = P(u, 2.4, 2.9);
    s += `<rect x="0" y="1080" width="${W}" height="270" fill="${K.red}" opacity="${a}"/>`;
    s += `<text x="56" y="1300" ${F} font-size="250" font-weight="800" letter-spacing="-10" fill="${K.white}" opacity="${a}">×${r.toFixed(1)}</text>`;
    s += `<g text-anchor="end" ${F} font-size="28" font-weight="600" fill="${K.white}" opacity="${a}"><text x="1008" y="1136">wider: crude’s conditional</text><text x="1008" y="1172">distribution with words</text><text x="1008" y="1208">than without</text><text x="1008" y="1250" font-size="22" font-weight="400" opacity=".85">10–90% range on day 30</text></g>`;
  }
  return s;
}

// ---------- 4 · every market, at once (black) ----------
function scene4(t) {
  const u = t - S.s4;
  let s = `<rect width="${W}" height="${H}" fill="${K.black}"/>`;
  s += `<text x="64" y="160" ${F} font-size="96" font-weight="800" letter-spacing="-3" fill="${K.white}" opacity="${P(u, 0, .5)}">Every market,</text>`;
  s += `<text x="64" y="262" ${F} font-size="96" font-weight="800" letter-spacing="-3" fill="${K.red}" opacity="${P(u, .2, .7)}">at once.</text>`;
  const cw = 444, ch = 196, k = P(u, 1.5, 3.6);
    D.markets.forEach((m, i) => {
    const px = 72 + (i % 2) * (cw + 48), py = 320 + Math.floor(i / 2) * (ch + 22);
    const a = P(u, .3 + i * .06, .8 + i * .06), [lo, hi] = scaleFor(m), Hn = m.hist.length;
    const gy = py + 50, gh = ch - 56;
    const X = d => px + cw * (d + Hn - 1) / (Hn - 1 + 30), Y = v => gy + gh * (1 - (v - lo) / (hi - lo));
    s += `<g opacity="${a}"><text x="${px}" y="${py + 32}" ${F} font-size="30" font-weight="700" fill="${K.white}">${m.name}</text>`;
    s += `<defs><clipPath id="c${i}"><rect x="${px}" y="${gy - 6}" width="${cw}" height="${gh + 12}"/></clipPath></defs><g clip-path="url(#c${i})">`;
    s += morphFan(m, k, X, Y, P(u, .4 + i * .06, 1.3 + i * .06, lin), { path: 1.2, med: 3.6, grey: 'hW', greyLine: '#B0B0B0' });
    s += `</g><polyline fill="none" stroke="${K.white}" stroke-width="3.4" points="${pts(m.hist.map((v, j) => [X(j - Hn + 1), Y(v)]))}"/>`;
    const r = 1 + (m.ratio - 1) * k;
    s += `<text x="${px + cw}" y="${py + 34}" text-anchor="end" ${F} font-size="36" font-weight="800" fill="${m.ratio > 1 ? K.red : K.white}" opacity="${P(u, 1.5, 1.9)}">×${r.toFixed(1)}<tspan dx="10" font-size="24" font-weight="600" fill-opacity="${P(u, 3.4, 3.9)}">${m.ratio > 1 ? 'wider' : 'narrower'}</tspan></text></g>`;
  });
  s += caption(t, S.s4 + .9, S.s5 - .2, 'Multimodal: market history and words, one generator.', K.white);
  return s;
}


// ================= HOW IT WORKS (v5: every element is real model output or real data) =================
let V = window.V4, TR = window.TR;
function kicker(u, txt, col) { return `<text x="72" y="104" ${F} font-size="30" font-weight="600" fill="${col}" opacity="${P(u, 0, .4)}">${txt}</text>`; }
const fmtN = n => n.toLocaleString('en-US');

// ---------- 1 · inside the network: dots are days, lines are the model's real attention (told one idea at a time) ----------
const NROWS = [450, 610, 770, 930], NX = d => 96 + d * (888 / 29);
function memBars(x0, base, amp, o, fly, tx, ty) {   // the real 128-number memory as thin bars (shared with the memory scene)
  const mm = TR.memory, mmx = Math.max(...mm.map(Math.abs));
  let s = '';
  mm.forEach((z, i) => {
    const hgt = amp * z / mmx, x = lerp(x0 + i * 7.3, tx, fly), y = lerp(hgt > 0 ? base - hgt : base, ty, fly);
    s += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${lerp(4.6, 2, fly).toFixed(2)}" height="${lerp(Math.max(1.5, Math.abs(hgt)), 2, fly).toFixed(1)}" fill="${z > 0 ? K.red : K.white}" opacity="${(o * .9).toFixed(3)}"/>`;
  });
  return s;
}
function sceneNet(t) {
  const u = t - S.s5, halo = 'paint-order="stroke" stroke="#000" stroke-width="7" stroke-linejoin="round"';
  let s = `<rect width="${W}" height="${H}" fill="${K.black}"/>`;
  s += kicker(u, 'How it works · 1 of 3', '#9A9A9A');
  s += `<text x="64" y="186" ${F} font-size="76" font-weight="800" letter-spacing="-3" fill="${K.white}" opacity="${P(u, .1, .6)}">Inside the network.</text>`;
  // ---- beat 1: a year of real S&P closes; the 30 days it reads light up and become 30 dots ----
  const yr = window.HIST['S&P 500'], N = yr.length, lo = Math.min(...yr), hi = Math.max(...yr);
  const sx = i => 72 + 936 * i / (N - 1), sy = v => 330 - 80 * (v - lo) / (hi - lo), dr = P(u, .3, 1.6, lin);
  s += `<text x="72" y="236" ${F} font-size="22" fill="#8C8C8C" opacity="${P(u, .3, .7)}">S&amp;P 500, Jan 2015 to 20 Jan 2016 (today)</text>`;
  s += `<polyline fill="none" stroke="#6E6E6E" stroke-width="1.6" points="${pts(yr.slice(0, Math.max(2, Math.round((N - 29) * dr))).map((v, i) => [sx(i), sy(v)]))}"/>`;
  const wo = P(u, 1.4, 1.8), wx0 = sx(N - 30), wx1 = sx(N - 1);
  s += `<g opacity="${wo}"><polyline fill="none" stroke="${K.white}" stroke-width="2.6" points="${pts(yr.slice(N - 30).map((v, i) => [sx(N - 30 + i), sy(v)]))}"/>
    <path d="M${wx0},350 L${wx0},362 L${wx1},362 L${wx1},350" fill="none" stroke="${K.white}" stroke-width="2"/>
    <text x="${wx1}" y="392" text-anchor="end" ${F} font-size="21" fill="${K.white}" opacity="${1 - P(u, 3.0, 3.4)}">the 30 days it reads</text></g>`;
  // ---- the lines: real attention, layer by layer ----
  const top = TR.enc_attn[0].indexOf(Math.max(...TR.enc_attn[0]));
  const todayO = l => P(u, [3.0, 7.6, 8.6][l - 1], [4.0, 8.3, 9.3][l - 1]);       // today's lines into layer l
  const restO = l => P(u, [6.0, 7.8, 8.8][l - 1], [7.0, 8.5, 9.5][l - 1]);        // every other day's lines into layer l
  for (let l = 1; l <= 3; l++) {
    const A = TR.enc_attn_full[l - 1], ro = restO(l), to = todayO(l);
    if (ro > 0) for (let i = 0; i < 29; i++) {
      const mx = Math.max(...A[i]);
      for (let j = 0; j <= i; j++) { const w = A[i][j] / mx; if (w >= .08) s += `<line x1="${NX(j)}" y1="${NROWS[l - 1] + 7}" x2="${NX(i)}" y2="${NROWS[l] - 7}" stroke="${K.white}" stroke-width="${(.6 + 1.6 * w).toFixed(2)}" opacity="${(ro * (.05 + .35 * w)).toFixed(3)}"/>`; }
    }
    if (to > 0) {
      const w = A[29], mx = Math.max(...w);
      for (let j = 0; j <= 29; j++) { const ww = w[j] / mx, reach = clamp(to * 1.3 - (29 - j) / 29 * .3); s += `<line x1="${NX(j)}" y1="${NROWS[l - 1] + 7}" x2="${lerp(NX(j), NX(29), reach)}" y2="${lerp(NROWS[l - 1] + 7, NROWS[l] - 7, reach)}" stroke="${K.red}" stroke-width="${(.8 + 4 * ww).toFixed(2)}" opacity="${(.2 + .8 * ww).toFixed(3)}"/>`; }
    }
  }
  // ---- the dots: one per day per layer (brightness = how strongly that day's numbers fire, real) ----
  NROWS.forEach((y, k) => {
    const act = TR.enc_tokens[k].map(v => v.reduce((a, b) => a + Math.abs(b), 0) / v.length), am = Math.max(...act);
    const o = k === 0 ? P(u, 1.8, 2.5) : (k === 1 ? Math.max(P(u, 3.0, 3.4) * .0, restO(1)) : restO(k));
    for (let d = 0; d < 30; d++) {
      const isToday = d === 29 && k > 0, oo = isToday ? todayO(k) : o;
      if (oo > 0) s += `<circle cx="${NX(d)}" cy="${y}" r="${d === 29 ? 8 : 6}" fill="${isToday ? K.red : K.white}" stroke="#000" stroke-width="2" opacity="${(oo * (d === 29 ? 1 : .35 + .65 * act[d] / am)).toFixed(3)}"/>`;
    }
  });
  s += `<g ${F} font-size="22" font-weight="700" fill="#D0D0D0">` + ['30 days in', 'Layer 1', 'Layer 2', 'Layer 3'].map((lab, k) =>
    `<text x="72" y="${NROWS[k] - 18}" ${halo} opacity="${k === 0 ? P(u, 1.8, 2.3) : todayO(k)}">${lab}</text>`).join('') + `</g>`;
  // ---- beat 3: today's attention as one bar: 30 equal slices (even split) re-sized to the real weights ----
  {
    const w1 = TR.enc_attn_full[0][29], ba = P(u, 3.4, 3.8) * (1 - P(u, 7.2, 7.6)), rs = P(u, 4.2, 5.4), bx = 96, bw = 888, by = 388;
    if (ba > 0) {
      let x = bx, g = '';
      for (let d = 0; d < 30; d++) {
        const wd = bw * lerp(1 / 30, w1[d], rs), isTop = d === TR.enc_attn[0].indexOf(Math.max(...TR.enc_attn[0]));
        g += `<rect x="${x.toFixed(1)}" y="${by}" width="${Math.max(0, wd - 1.5).toFixed(1)}" height="18" fill="${isTop ? K.red : (d % 2 ? '#5C5C5C' : '#8C8C8C')}"/>`;
        if (isTop && rs > .6) g += `<text x="${(x + wd / 2).toFixed(1)}" y="${by - 10}" text-anchor="middle" ${F} font-size="21" font-weight="700" fill="${K.red}" opacity="${P(u, 4.8, 5.2)}">7 Jan · ${Math.round(100 * w1[d])}%</text>`;
        x += wd;
      }
      s += `<g opacity="${ba}">${g}<text x="${bx}" y="${by - 10}" ${F} font-size="20" fill="#9A9A9A">${rs < .5 ? 'today’s attention, split evenly: 30 × 3%' : 'today’s attention, as the model splits it'}</text></g>`;
    }
  }
  // ---- the day it leans on (layer 1, measured) ----
  const w1 = TR.enc_attn_full[0][29], share = Math.round(100 * w1[top]), H0 = TR.history, ret = 100 * (H0.spx[top] / H0.spx[top - 1] - 1);
  const ta = P(u, 4.4, 4.8) * (1 - P(u, 6.6, 7.0)), tx = clamp(NX(top), 300, 780);
  s += `<g opacity="${ta}"><circle cx="${NX(top)}" cy="${NROWS[0]}" r="14" fill="none" stroke="${K.red}" stroke-width="3"/>
    <text x="${NX(top)}" y="${NROWS[0] + 36}" text-anchor="middle" ${F} font-size="20" font-weight="700" fill="${K.white}" paint-order="stroke" stroke="#000" stroke-width="6">S&amp;P ${ret < 0 ? '−' : '+'}${Math.abs(ret).toFixed(1)}%</text></g>`;
  // ---- beat 6: what is left for today: 128 numbers ----
  const ma = P(u, 10.4, 11.0);
  s += `<g opacity="${ma}"><line x1="${NX(29)}" y1="${NROWS[3] + 10}" x2="${NX(29)}" y2="996" stroke="${K.red}" stroke-width="2.5"/>${memBars(72, 1030, 30, P(u, 10.6, 11.2), 0, 0, 0)}
    <text x="72" y="1090" ${F} font-size="22" fill="#BDBDBD">Today’s memory: 128 numbers</text></g>`;
  s += `<text x="72" y="1160" ${F} font-size="34" font-weight="800" fill="${K.white}" opacity="${P(u, 12.0, 12.5)}">The same machinery as ChatGPT,</text>`;
  s += `<text x="72" y="1202" ${F} font-size="34" font-weight="800" fill="${K.red}" opacity="${P(u, 12.2, 12.7)}">here reading 30 days of 39 variables.</text>`;
  s += caption(t, S.s5 + .5, S.s5 + 2.9, 'It reads the last 30 days.', K.white);
  s += caption(t, S.s5 + 3.0, S.s5 + 4.9, 'For today, it asks: which days matter most?', K.white);
  s += caption(t, S.s5 + 5.0, S.s5 + 6.9, 'It leans on the big sell-off days.', K.white);
  s += caption(t, S.s5 + 7.0, S.s5 + 8.5, 'Every day does the same, looking back.', K.white);
  s += caption(t, S.s5 + 8.6, S.s5 + 10.4, 'Three rounds; the last also weighs rebound days.', K.white);
  s += caption(t, S.s5 + 10.5, S.s5 + 12.4, 'What’s left for today: 128 numbers, its memory.', K.white);
  s += caption(t, S.s5 + 12.5, S.s6 - .1, `Whole model: ${fmtN(TR.params)} parameters, market data only.`, '#9A9A9A');
  return s;
}

// ---------- 3 · noise in, market move out: the second Transformer, its 16 pushes, then day after day ----------

let V11 = window.V11, V16 = window.V16;
const hsh = i => { const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453; return x - Math.floor(x); };
const fmtDate = d => new Date(d + 'T00:00:00Z').toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−');
// ---------- How it works 3 · flow matching on one plane: learn straight lines, then generate ----------
// the plane is the model's own coordinate: next-day move divided by recent volatility
const FP = { cx: 540, cy: 730, R: 320, lim: 5 };
const planeMap = (cx, cy, R) => [v => cx + R / FP.lim * clamp(v, -FP.lim, FP.lim), v => cy - R / FP.lim * clamp(v, -FP.lim, FP.lim)];
function planeCloud(X, Y, o, n, r) {
  let g = '';
  const C = V11.flow.cloud;
  for (let i = 0; i < Math.min(n, C.length); i++) if (Math.abs(C[i][0]) <= FP.lim && Math.abs(C[i][1]) <= FP.lim) g += `<circle cx="${X(C[i][0]).toFixed(1)}" cy="${Y(C[i][1]).toFixed(1)}" r="${r}"/>`;
  return `<g fill="#C4C4C4" opacity="${o}">${g}</g>`;
}
const PB = 0;
function sceneFlow(t) {
  const u0 = t - SF, u = u0 - PB, FL = V11.flow, end = S.s8 - SF - PB;
  let s = `<rect width="${W}" height="${H}" fill="${K.white}"/>`;
  const out_ = 1;
  s += `<g opacity="${out_}">${kicker(u0, 'How it works · 3 of 3', K.mute)}</g>`;
  s += `<text x="64" y="186" ${F} font-size="76" font-weight="800" letter-spacing="-3" fill="${K.black}" opacity="${P(u0, .1, .6) * out_}">Noise in.</text>`;
  s += `<text x="64" y="266" ${F} font-size="76" font-weight="800" letter-spacing="-3" fill="${K.red}" opacity="${P(u, 10.6, 11.1) * out_}">Market move out.</text>`;
  s += `<text x="66" y="258" ${F} font-size="44" font-weight="800" letter-spacing="-1" fill="${K.mute}" opacity="${P(u, 1.2, 1.6) * (1 - P(u, 6.0, 6.4))}">1 · Training: the answer is known.</text>`;
  s += `<text x="66" y="258" ${F} font-size="44" font-weight="800" letter-spacing="-1" fill="${K.mute}" opacity="${P(u, 6.5, 6.9) * (1 - P(u, 10.3, 10.6))}">2 · Generating: no answer given.</text>`;
  const { cx, cy, R } = FP, [X, Y] = planeMap(cx, cy, R), ax = P(u, -.3, .3);
  s += `<g opacity="${ax}" stroke="${K.rule}" stroke-width="2"><line x1="${cx - R}" y1="${cy}" x2="${cx + R}" y2="${cy}"/><line x1="${cx}" y1="${cy - R}" x2="${cx}" y2="${cy + R}"/></g>`;
  s += `<g ${F} fill="${K.mute}" opacity="${ax * out_ * (1 - P(u, 11.0, 11.4))}"><text x="${cx - R}" y="${cy + R + 40}" font-size="24">S&amp;P 500 next-day move →</text>
    <text x="${cx - R}" y="${cy + R + 72}" font-size="24" fill="${K.mute}">both axes in units of recent volatility</text>
    <text x="${cx + 14}" y="${cy - R + 22}" font-size="24">↑ VIX next-day move</text></g>`;
  s += layer(planeCloud(X, Y, .75 * P(u, -.2, .5), 1500, 3));
  // legend
  const lg = (y, col, txt, o, ring) => `<g opacity="${o * out_}"><circle cx="${cx + 66}" cy="${y - 7}" r="7" fill="${ring ? K.white : col}" stroke="${ring ? K.black : 'none'}" stroke-width="2.5"/><text x="${cx + 82}" y="${y}" ${F} font-size="25" fill="${col === '#C4C4C4' ? K.mute : col}">${txt}</text></g>`;
  s += lg(cy - R + 64, '#C4C4C4', `real days, ${FL.first.slice(0, 4)}–${FL.last.slice(0, 4)}`, P(u, .8, 1.2));
  s += lg(cy - R + 98, K.black, 'noise → real day', P(u, 1.3, 1.7) * (1 - P(u, 5.9, 6.2)));
  s += lg(cy - R + 98, K.black, 'fresh noise', P(u, 6.6, 7.0) * (1 - P(u, 10.2, 10.5)));
  // ---- beat 1 · training: straight lines from noise to real days ----
  const tr = 1 - P(u, 6.1, 6.6);
  if (tr > 0 && u > 1.2) {
    const { x0, x1 } = FL.pairs, ln = P(u, 1.8, 2.6), tau = P(u, 2.8, 5.8, lin);
    let lines = '', dots = '';
    x0.forEach((p, i) => {
      if (i >= 150) return;
      const q = x1[i], e = [lerp(p[0], q[0], ln), lerp(p[1], q[1], ln)], m = [lerp(p[0], q[0], tau), lerp(p[1], q[1], tau)];
      if (ln > 0) lines += `<line x1="${X(p[0]).toFixed(1)}" y1="${Y(p[1]).toFixed(1)}" x2="${X(e[0]).toFixed(1)}" y2="${Y(e[1]).toFixed(1)}"/>`;
      dots += `<circle cx="${X(m[0]).toFixed(1)}" cy="${Y(m[1]).toFixed(1)}" r="4"/>`;
    });
    s += `<g opacity="${tr}"><g stroke="${K.red}" stroke-width="1.5" opacity=".4">${lines}</g><g fill="${K.black}" opacity="${.8 * P(u, 1.3, 1.7)}">${dots}</g></g>`;
    const fo = P(u, 2.6, 3.0) * tr;
    s += `<g ${F} opacity="${fo}"><text x="72" y="352" font-size="30" fill="${K.black}">point = (1 − τ) × noise + τ × real day</text>
      <text x="1008" y="350" text-anchor="end" font-size="34" font-weight="800" fill="${K.red}">τ = ${tau.toFixed(2)}</text></g>`;
  }
  // ---- beat 2 · generating: fresh noise, 16 real pushes, 400 draws across all windows, market state only ----
  if (u > 6.4) {
    const ga = P(u, 6.5, 6.9), k = 16 * P(u, 7.2, 10.4, lin), k0 = Math.min(15, Math.floor(k)), fr = k >= 16 ? 1 : k - k0;
    let trails = '', dots = '';
    FL.draws.forEach((tr2, i) => {
      const p = [lerp(tr2[k0][0], tr2[k0 + 1][0], fr), lerp(tr2[k0][1], tr2[k0 + 1][1], fr)];
      if (i % 4 === 0 && k > 0 && k < 16) trails += `<polyline points="${pts([...tr2.slice(0, k0 + 1), p].map(q => [X(q[0]), Y(q[1])]))}"/>`;
      dots += `<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="4"/>`;
    });
    s += `<g fill="none" stroke="${K.red}" stroke-width="1.1" opacity=".3">${trails}</g><g fill="${k >= 16 ? K.red : K.black}" opacity="${(k >= 16 ? .8 : .65) * ga}">${dots}</g>`;
    s += `<text x="${cx + R}" y="${cy + R - 16}" text-anchor="end" ${F} font-size="32" font-weight="800" fill="${k >= 16 ? K.red : K.black}" opacity="${ga * out_}">push ${Math.min(16, Math.floor(k))} / 16</text>`;
    s += lg(cy - R + 98, K.red, 'model’s draws, same days', P(u, 10.5, 10.9));
  }
  // ---- the tilt: a line whose slope is the correlation of the dots on screen, recomputed at every push ----
  if (u > 6.4) {
    const k = 16 * P(u, 7.2, 10.4, lin), k0 = Math.min(15, Math.floor(k)), fr = k >= 16 ? 1 : k - k0;
    const xs = FL.draws.map(tr => lerp(tr[k0][0], tr[k0 + 1][0], fr)), ys = FL.draws.map(tr => lerp(tr[k0][1], tr[k0 + 1][1], fr));
    const mxs = xs.reduce((a, b) => a + b) / xs.length, mys = ys.reduce((a, b) => a + b) / ys.length;
    let sxy = 0, sxx = 0, syy = 0; xs.forEach((x, i) => { sxy += (x - mxs) * (ys[i] - mys); sxx += (x - mxs) ** 2; syy += (ys[i] - mys) ** 2; });
    const rho = sxy / Math.sqrt(sxx * syy), lo_ = P(u, 6.6, 7.0) * out_;
    s += `<line x1="${X(-4.2)}" y1="${Y(-4.2 * rho)}" x2="${X(4.2)}" y2="${Y(4.2 * rho)}" stroke="${K.red}" stroke-width="5" stroke-linecap="round" opacity="${lo_}"/>`;
    s += `<line x1="${X(-4.2)}" y1="${Y(-4.2 * FL.hist_corr)}" x2="${X(4.2)}" y2="${Y(4.2 * FL.hist_corr)}" stroke="${K.black}" stroke-width="2.5" stroke-dasharray="8 6" opacity="${P(u, 10.4, 10.9) * out_}"/>`;
    s += `<text x="${X(-4.2)}" y="${Y(-4.2 * FL.hist_corr) - 14}" ${F} font-size="20" font-weight="600" fill="${K.black}" paint-order="stroke" stroke="#FFFFFF" stroke-width="6" stroke-linejoin="round" opacity="${P(u, 10.4, 10.9) * out_}">real days</text>`;
  }
  // ---- the payoff: the relation nobody wrote down ----
  const band = P(u, 11.3, 11.8) * out_;
  if (band > 0) {
    s += `<rect x="0" y="1100" width="${W}" height="250" fill="${K.black}" opacity="${band}"/>`;
    s += `<text x="72" y="1156" ${F} font-size="30" font-weight="700" fill="${K.white}" opacity="${band}">Stocks down, volatility up. Correlation of vol-scaled moves:</text>`;
    [[FL.noise_corr, 'in the noise', '#9A9A9A'], [FL.model_corr, 'in the model’s draws', K.red], [FL.hist_corr, 'in real days', K.white]].forEach(([v, lab, col], i) => {
      const a = P(u, 11.6 + i * .35, 12.0 + i * .35) * out_, x = 72 + i * 330;
      s += `<g opacity="${a}"><text x="${x}" y="1236" ${F} font-size="60" font-weight="800" letter-spacing="-2" fill="${col}">${f2(v)}</text>
        <text x="${x}" y="1272" ${F} font-size="26" fill="#D0D0D0">${lab}</text></g>`;
    });
    s += `<text x="72" y="1324" ${F} font-size="30" font-weight="700" fill="${K.red}" opacity="${P(u, 12.9, 13.4) * out_}">Nobody wrote that rule down. The model learned most of it.</text>`;
  }
  s += caption(t, SF + PB + .2, SF + PB + 1.9, 'Each grey dot: one real day’s S&P and VIX move.', K.black);
  s += caption(t, SF + PB + 2.0, SF + PB + 4.2, 'Join noise to a real day with a straight line.', K.black);
  s += caption(t, SF + PB + 4.3, SF + PB + 6.5, 'Where lines cross, it learns the average push.', K.black);
  s += caption(t, SF + PB + 6.7, SF + PB + 11.2, 'Fresh noise, pushed 16 times by the network.', K.black);
  return s;
}


// ---------- shared: the real t-SNE map of the frozen encoder's 4,010 memories ----------
const MAP = { x: 92, y: 380, s: 860, k: .8 };
const MX = v => MAP.x + MAP.s * v, MY = v => MAP.y + MAP.s * MAP.k * v;
function memDots(yr, dim) {
  const M = V.mem; let dots = '';
  for (let i = 0; i < M.xy.length; i++) {
    if (M.year[i] > yr) continue;
    const hot = clamp((M.vix[i] - 18) / 22);
    dots += `<circle cx="${MX(M.xy[i][0]).toFixed(1)}" cy="${MY(M.xy[i][1]).toFixed(1)}" r="${2.4 + 1.6 * hot}" fill="${hot > .45 ? K.red : K.white}" opacity="${((hot > .45 ? .55 + .4 * hot : .22) * dim).toFixed(3)}"/>`;
  }
  return dots;
}
function stressKey(o) { return `<g opacity="${o}" ${F} font-size="24"><circle cx="760" cy="${MAP.y + MAP.s * MAP.k + 34}" r="8" fill="${K.red}"/><text x="778" y="${MAP.y + MAP.s * MAP.k + 42}" fill="#BDBDBD">stressed (VIX above 28)</text></g>`; }

// ---------- C · what the model remembers, then words meet memory (one map, two beats) ----------
function sceneMemory(t) {
  const u = t - S.s6, M = V.mem, WB = 10.0, w = u - WB;   // WB: start of the words beat
  let s = `<rect width="${W}" height="${H}" fill="${K.black}"/>`;
  s += kicker(u, 'How it works · 2 of 3', '#9A9A9A');
  const t1 = P(u, .1, .6) * (1 - P(w, 0, .4)), t2 = P(w, .4, .8);
  s += `<text x="64" y="186" ${F} font-size="76" font-weight="800" letter-spacing="-3" fill="${K.white}" opacity="${t1}">What the model</text>`;
  s += `<text x="64" y="266" ${F} font-size="76" font-weight="800" letter-spacing="-3" fill="${K.red}" opacity="${P(u, .2, .7) * (1 - P(w, 0, .4))}">remembers.</text>`;
  s += `<text x="64" y="186" ${F} font-size="76" font-weight="800" letter-spacing="-3" fill="${K.white}" opacity="${t2}">Words meet</text>`;
  s += `<text x="64" y="266" ${F} font-size="76" font-weight="800" letter-spacing="-3" fill="${K.red}" opacity="${t2}">memory.</text>`;
  s += `<text x="72" y="330" ${F} font-size="26" fill="#BDBDBD" opacity="${P(u, .5, 1.0) * (1 - P(w, 0, .4))}">Each dot is one of its real 128-number memories, projected to 2-D.</text>`;
  // today's memory folds into its dot
  const [qx0, qy0] = M.xy[4005], fly = P(u, .9, 2.0, out);
  if (u < 2.6) s += memBars(72, 1030, 30, 1 - P(u, 1.8, 2.2), fly, MX(qx0), MY(qy0));   // the network's 128 numbers fly into today's dot
  const yr = lerp(2000, 2016.99, P(u, 2.2, 5.2, lin));
  if (u > 2.2) s += layer(memDots(yr, 1 - .6 * P(w, 3.0, 3.6)));
  const ta = Math.max(P(u, 1.8, 2.2) * (1 - P(u, 5.0, 5.4)), P(w, 5.0, 5.5));
  s += `<g opacity="${ta}"><circle cx="${MX(qx0)}" cy="${MY(qy0)}" r="12" fill="none" stroke="${K.white}" stroke-width="3"/>
    <text x="${MX(qx0) + 22}" y="${MY(qy0) + 9}" ${F} font-size="26" font-weight="700" fill="${K.white}">20 Jan 2016</text></g>`;
  s += `<text x="1008" y="${MAP.y + 30}" text-anchor="end" ${F} font-size="48" font-weight="800" fill="${K.white}" opacity="${P(u, 2.2, 2.6) * (1 - P(u, 5.3, 5.6))}">${Math.floor(yr)}</text>`;
  s += stressKey(P(u, 2.6, 3.0));
  const slot = { 'Lehman, Oct 2008': ['end', null, 52], 'Euro crisis, Aug 2011': ['end', -18, 96], 'Dot-com, 2002': ['start', 22, -16] };
  const la = P(u, 3.4, 3.9) * (1 - P(u, 5.3, 5.7));   // landmarks clear before the neighbour lines
  M.landmarks.forEach(l => {
    const [x, y] = M.xy[l.i], [an, dx, dy] = slot[l.label], lx = dx === null ? 1008 : MX(x) + dx, ly = MY(y) + dy;
    s += `<g opacity="${la}"><circle cx="${MX(x)}" cy="${MY(y)}" r="9" fill="none" stroke="${K.white}" stroke-width="2.5"/>
      <line x1="${MX(x)}" y1="${MY(y)}" x2="${an === 'end' ? Math.min(lx, MX(x)) : lx - 6}" y2="${ly - 30}" stroke="${K.white}" stroke-width="1.5"/>
      <text x="${lx}" y="${ly}" text-anchor="${an}" ${F} font-size="26" font-weight="700" fill="${K.white}">${esc(l.label)}</text></g>`;
  });
  const st = M.stress, so = 1 - P(w, -.4, 0);
  {
    const NB = V16.neigh, [ex, ey] = M.xy[NB.i], na = P(u, 5.6, 6.0) * so, hit = NB.hot.reduce((x, y) => x + y, 0);
    if (na > 0) {
      s += `<circle cx="${MX(ex)}" cy="${MY(ey)}" r="${lerp(40, 20, P(u, 5.6, 6.2))}" fill="none" stroke="${K.white}" stroke-width="2.5" opacity="${na}"/><circle cx="${MX(ex)}" cy="${MY(ey)}" r="9" fill="${K.red}" stroke="${K.white}" stroke-width="2.5" opacity="${na}"/>`;
      NB.nn.forEach((j, k) => {
        const [x, y] = M.xy[j], g = P(u, 6.0 + k * .1, 6.4 + k * .1, out) * so;
        s += `<line x1="${MX(ex)}" y1="${MY(ey)}" x2="${lerp(MX(ex), MX(x), g)}" y2="${lerp(MY(ey), MY(y), g)}" stroke="${NB.hot[k] ? K.red : K.white}" stroke-width="2" opacity="${.8 * g}"/>
          <circle cx="${MX(x)}" cy="${MY(y)}" r="7" fill="${NB.hot[k] ? K.red : K.white}" stroke="#000" stroke-width="2" opacity="${g}"/>`;
      });
      const lab = new Date(NB.end + 'T00:00:00Z').toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
      s += `<text x="72" y="1186" ${F} font-size="30" font-weight="700" fill="${K.white}" opacity="${P(u, 6.8, 7.2) * so}">${lab}, a stressed day: <tspan fill="${K.red}">${hit} of its 10</tspan> closest memories</text>`;
      s += `<text x="72" y="1230" ${F} font-size="30" font-weight="700" fill="${K.white}" opacity="${P(u, 6.8, 7.2) * so}">are stressed too, where ${Math.round(100 * NB.share_all)}% would be chance.</text>`;
      s += `<text x="72" y="1282" ${F} font-size="32" font-weight="700" fill="${K.red}" opacity="${P(u, 7.6, 8.0) * so}">Nobody labelled them.</text>`;
    }
  }
  s += caption(t, S.s6 + 2.4, S.s6 + 6.4, 'Similar markets land near each other.', K.white);
  // ---- words beat: the real query the demo searched with, and its selected precedents ----
  if (w > 0) {
    const qa = P(w, 1.2, 2.0, out), qx = MX(M.q[0]), qy = MY(M.q[1]);
    s += `<text x="72" y="340" ${F} font-size="30" font-style="italic" fill="${K.white}" opacity="${P(w, .4, .8) * (1 - P(w, 2.6, 3.0))}">${esc('“' + D.story.split(',')[0] + '…”')}</text>`;
    if (w > 1.0) {
      const cqx = lerp(300, qx, qa), cqy = lerp(400, qy, qa);
      s += `<circle cx="${cqx}" cy="${cqy}" r="16" fill="none" stroke="${K.white}" stroke-width="4"/><circle cx="${cqx}" cy="${cqy}" r="5" fill="${K.white}"/>`;
      s += `<text x="${cqx - 26}" y="${cqy + 9}" text-anchor="end" ${F} font-size="26" font-weight="700" fill="${K.white}" paint-order="stroke" stroke="#000" stroke-width="6" stroke-linejoin="round" opacity="${P(w, 1.8, 2.2)}">your words</text>`;
    }
    if (w > 3.2) M.hl.forEach((h, k) => {
      const a = P(w, 3.4 + k * .45, 3.8 + k * .45), [x, y] = M.xy[h.i], tx = 596, ty = 760 + k * 66;
      s += `<g opacity="${a}"><line x1="${qx}" y1="${qy}" x2="${MX(x)}" y2="${MY(y)}" stroke="${K.white}" stroke-width="3"/>
        <circle cx="${MX(x)}" cy="${MY(y)}" r="10" fill="${K.white}" stroke="#000" stroke-width="3"/>
        <line x1="${MX(x)}" y1="${MY(y)}" x2="${tx - 10}" y2="${ty - 10}" stroke="#5C5C5C" stroke-width="1.5"/>
        <rect x="${tx}" y="${ty - 42}" width="412" height="56" fill="${K.white}"/><rect x="${tx}" y="${ty - 42}" width="10" height="56" fill="${K.red}"/>
        <text x="${tx + 26}" y="${ty - 4}" ${F} font-size="29" font-weight="700" fill="${K.black}">${h.label}<tspan font-weight="400" fill="${K.mute}" dx="12">cosine ${h.cos.toFixed(2)}</tspan></text></g>`;
    });
    s += `<g ${F} font-size="24" fill="#BDBDBD" paint-order="stroke" stroke="#000" stroke-width="6" stroke-linejoin="round" opacity="${P(w, 5.0, 5.5)}"><text x="596" y="${760 + 3 * 66 + 4}">Largest 3 of 8 weighted matches,</text><text x="596" y="${760 + 3 * 66 + 36}">checked against 20 Jan 2016.</text></g>`;
    s += caption(t, S.s6 + WB + 1.0, S.s6 + WB + 3.2, 'A learned bridge places your words in the same space.', K.white);
    s += caption(t, S.s6 + WB + 3.3, S.s7 - .1, 'The same generator then draws new paths from them.', K.red);
  }
  return s;
}

// ---------- 10 · close ----------
function sceneClose(t) {
  const u = t - S.s8;
  let s = `<rect width="${W}" height="${H}" fill="${K.black}"/>`;
  const w = P(u, .05, .7, out);
  s += `<polygon points="${W},${0} ${W},${360} ${lerp(W, 560, w)},${0}" fill="${K.red}"/>`;
  s += `<text x="64" y="560" ${F} font-size="120" font-weight="800" letter-spacing="-4" fill="${K.white}" opacity="${P(u, .3, .9)}">ScenarioFlow</text>`;
  ['Multimodal Market Conditioning for', 'Heterogeneous Multivariate', 'Time-Series Simulation'].forEach((l, i) => {
    s += `<text x="68" y="${660 + i * 52}" ${F} font-size="40" font-weight="500" fill="${K.white}" opacity="${P(u, .7 + i * .12, 1.3 + i * .12)}">${l}</text>`;
  });
  // logos, all in white ink, matched height; hairline dividers between them
  // heights set by eye so the three marks read equally large; all centred on y=898
  const LH = { uoft: 84, otpp: 80, neurips: 108 };
  let lx = 68;
  ['uoft', 'otpp', 'neurips'].forEach((k, i) => {
    const vb = window.LOGOS[k].vb.split(' ').map(Number), lh = LH[k], lw = lh * vb[2] / vb[3], a = P(u, 1.3 + i * .18, 1.8 + i * .18);
    if (i) { s += `<line x1="${lx}" y1="858" x2="${lx}" y2="938" stroke="${K.mute}" stroke-width="2" opacity="${a}"/>`; lx += 46; }
    s += `<use href="#logo-${k}" x="${lx}" y="${898 - lh / 2}" width="${lw}" height="${lh}" opacity="${a}"/>`;
    lx += lw + 46;
  });
  s += `<text x="68" y="${1020}" ${F} font-size="28" fill="${K.grey}" opacity="${P(u, 1.9, 2.4)}">NeurIPS 2026 Workshop on Foundation Models for Temporal Systems</text>`;
  const b = P(u, 1.9, 2.5, out);
  s += `<rect x="0" y="1150" width="${W * b}" height="200" fill="${K.red}"/>`;
  s += `<text x="68" y="1270" ${F} font-size="52" font-weight="800" letter-spacing="-1" fill="${K.white}" opacity="${P(u, 2.3, 2.7)}">${esc(CLOSE_LINE)}</text>`;
  return s;
}


const SEQ = [scene1, scene2, scene3, scene4, sceneNet, sceneMemory, sceneFlow, sceneClose];
const AT = [S.s1, S.s2, S.s3, S.s4, S.s5, S.s6, S.s7, S.s8];
function frame(t) {
  let s = DEFS;
  const seq = SEQ, at = AT;
  let k = 0; while (k + 1 < at.length && t >= at[k + 1]) k++;
  // during a wipe the old scene stays ahead of the band and the new one appears behind it, so no cut is visible
  const xf = at.findIndex((a, i) => i > 0 && a !== S.s5 && Math.abs(t - a) < .14);
  const wi = at.findIndex((a, i) => i > 0 && a === S.s5 && Math.abs(t - a) < .35);
  NEST = xf > 0 || wi > 0;   // inside a transition the scenes are nested in groups: no layer markers
  if (xf > 0) {
    const kx = io((t - (at[xf] - .14)) / .28);
    s += `<g>${seq[xf - 1](t)}</g><g opacity="${kx.toFixed(3)}">${seq[xf](t)}</g>`;
  } else if (wi > 0) {
    const x = lerp(-900, W + 300, io((t - (at[wi] - .35)) / .7));
    s += `<defs><clipPath id="wOld"><polygon points="${x + 520},0 ${W},0 ${W},${H} ${x + 100},${H}"/></clipPath><clipPath id="wNew"><polygon points="0,0 ${x},0 ${x - 420},${H} 0,${H}"/></clipPath></defs>`;
    s += `<g clip-path="url(#wOld)">${seq[wi - 1](t)}</g><g clip-path="url(#wNew)">${seq[wi](t)}</g>` + wipe(t, at[wi]);
  } else s += seq[k](t);
  return s;
}
function bind() { D = window.DATA; V = window.V4; TR = window.TR; V11 = window.V11; V16 = window.V16; }
const prefixIds = (s, p) => s.replace(/ id="/g, ` id="${p}`).replace(/url\(#/g, `url(#${p}`).replace(/href="#/g, `href="#${p}`);
// layer markers: top-level, balanced fragments the page may keep in their own <g> between frames
let NEST = false;
const layer = x => (NEST || !LAYERS ? x : '<!--L-->' + x + '<!--/L-->');
let LAYERS = false;
let REC = null, CAPS = true, CLOSE_LINE = 'Live demo and paper in the post.';
const TONE = c => (c === K.red ? 'red' : (c === '#9A9A9A' || c === '#BDBDBD' || c === K.mute) ? 'mute' : 'ink');
let CAPTIONS = null;
function collectCaptions() {
  // run every scene in the sequence with caption() recording instead of drawing (sceneDiff is not in the sequence)
  bind(); REC = [];
  try {
    SEQ.forEach((fn, i) => {
      const a = AT[i], b = i + 1 < AT.length ? AT[i + 1] : DUR;
      for (let t = a; t < b; t += .05) fn(t);
    });
  } finally { const r = REC; REC = null; CAPTIONS = r; }
  const seen = new Set();
  CAPTIONS = CAPTIONS.filter(c => { const key = c.t0 + '|' + c.text; if (seen.has(key)) return false; seen.add(key); return true; })
    .map(c => ({ t0: c.t0, t1: c.t1, text: c.text, tone: TONE(c.col), colour: c.col }))
    .sort((x, y) => x.t0 - y.t0);
  return CAPTIONS;
}
window.SFStory = {
  DUR, S: { s1: S.s1, s2: S.s2, s3: S.s3, s4: S.s4, s5: S.s5, s6: S.s6, s7: S.s7, s8: S.s8 }, SIZE: [W, H],
  // render(t, { captions, idPrefix, closeLine, layers }) -> SVG inner markup for the 1080x1350 viewBox
  render(t, opts) {
    const o = opts || {};
    bind();
    CAPS = o.captions !== false;
    CLOSE_LINE = o.closeLine || 'Live demo and paper in the post.';
    LAYERS = !!o.layers;
    let s;
    try { s = frame(t); } finally { CAPS = true; LAYERS = false; NEST = false; CLOSE_LINE = 'Live demo and paper in the post.'; }
    return o.idPrefix ? prefixIds(s, o.idPrefix) : s;
  },
  get captions() { return CAPTIONS || collectCaptions(); },
  // the three partner marks used on the closing card, as <symbol>s (inject once into a <defs>)
  logoDefs(prefix) {
    const p = prefix || '';
    return Object.entries(window.LOGOS || {}).map(([k, L]) => `<symbol id="${p}logo-${k}" viewBox="${L.vb}">${L.inner}</symbol>`).join('');
  },
  ready(kind) { return kind === 'hero' ? !!window.DATA : !!(window.DATA && window.V4 && window.TR && window.V11 && window.HIST && window.V16 && window.LOGOS); },
};
})();
