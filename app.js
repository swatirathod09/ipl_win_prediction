const $ = s => document.querySelector(s);
const DEFAULTS = {bat: "Royal Challengers Bangalore", bowl: "Chennai Super Kings", venue: "M Chinnaswamy Stadium", score: "0", wk: "0", overs: "0", target: "180"};

// ---- Offline copy of the trained model (same Logistic Regression as the Flask server) ----
function localProbability(f) {
  const P = MODEL.params, x = [f.runs_left, f.balls_left, f.wickets_left, f.target, f.crr, f.rrr];
  let z = P.intercept + (P.bat[f.batting_team] || 0) + (P.bowl[f.bowling_team] || 0) + (P.venue[f.venue] || 0);
  x.forEach((v, i) => { z += P.num_coef[i] * (v - P.mean[i]) / P.scale[i]; });
  return 1 / (1 + Math.exp(-z));
}
function parseOvers(text) {
  const s = String(text).trim(), m = /^(\d+)(?:\.(\d))?$/.exec(s);
  if (!m) throw new Error("Overs must look like 14 or 14.3 (14 overs and 3 balls).");
  const balls = Number(m[2] || 0);
  if (balls > 5) throw new Error("Balls after the decimal point must be 0 to 5. Example: 14.3 means 14 overs and 3 balls.");
  const total = Number(m[1]) * 6 + balls;
  if (total > 120) throw new Error("A chase is at most 20 overs.");
  return total;
}
function whole(value, name, lo, hi) {
  const s = String(value).trim();
  if (!/^\d+$/.test(s) || Number(s) < lo || Number(s) > hi) throw new Error(`${name} must be a whole number from ${lo} to ${hi}.`);
  return Number(s);
}
const r2 = x => Math.round(x * 100) / 100;
function localSituation(d) {
  const score = whole(d.score, "Current score", 0, 500), wk = whole(d.wickets, "Wickets lost", 0, 10);
  const target = whole(d.target, "Target score", 1, 500), bowled = parseOvers(d.overs);
  if (d.batting_team === d.bowling_team) throw new Error("Choose two different teams.");
  const runsLeft = target - score, ballsLeft = 120 - bowled, bat = d.batting_team;
  const crr = bowled ? r2(score * 6 / bowled) : 0, rrr = ballsLeft > 0 && runsLeft > 0 ? r2(runsLeft * 6 / ballsLeft) : 0;
  const out = {batting_team: bat, bowling_team: d.bowling_team, runs_remaining: Math.max(runsLeft, 0), balls_remaining: ballsLeft,
    wickets_in_hand: 10 - wk, current_run_rate: crr, required_run_rate: rrr};
  if (runsLeft <= 0) return {...out, status: "won", probability: 100, message: `${bat} have already reached the target of ${target}.`};
  if (wk >= 10) return {...out, status: "lost", probability: 0, message: `${bat} are all out, so the chase is over.`};
  if (ballsLeft <= 0) return {...out, status: "lost", probability: 0, message: `The overs are finished and ${bat} still need ${runsLeft} runs, so the chase is over.`};
  const p = localProbability({batting_team: bat, bowling_team: d.bowling_team, venue: d.venue, runs_left: runsLeft, balls_left: ballsLeft,
    wickets_left: 10 - wk, target, crr, rrr: Math.min(rrr, 36)});
  const w = 10 - wk;
  return {...out, status: "live", probability: Math.round(p * 1000) / 10,
    message: `${bat} need ${runsLeft} runs from ${ballsLeft} legal balls, with ${w} wicket${w !== 1 ? "s" : ""} in hand. The required run rate is ${rrr.toFixed(2)}.`};
}

// ---- Page ----
const fill = (sel, list, val) => { $(sel).innerHTML = list.map(t => `<option>${t}</option>`).join(""); $(sel).value = val; };
function setForm(v) {
  fill("#bat", MODEL.teams, v.bat); fill("#bowl", MODEL.teams, v.bowl); fill("#venue", MODEL.venues, v.venue);
  ["score", "wk", "overs", "target"].forEach(k => { $("#" + k).value = v[k]; $("#" + k).removeAttribute("aria-invalid"); });
}
function render(r) {
  $("#o-team").textContent = r.batting_team;
  $("#o-pct").textContent = r.probability.toFixed(1);
  $("#o-fill").style.width = r.probability + "%";
  $("#o-msg").textContent = r.message;
  $("#s-runs").textContent = r.runs_remaining;
  $("#s-balls").textContent = r.balls_remaining;
  $("#s-crr").textContent = r.current_run_rate.toFixed(2);
  $("#s-rrr").textContent = r.required_run_rate.toFixed(2);
}
function clearOutput() {
  $("#o-team").textContent = "Waiting for inputs"; $("#o-pct").textContent = "--.-"; $("#o-fill").style.width = "0";
  $("#o-msg").textContent = "Fill in the match situation and press Predict win probability.";
  ["#s-runs", "#s-balls", "#s-crr", "#s-rrr"].forEach(s => $(s).textContent = "–");
}
function setMode(live) { const m = $("#mode"); m.textContent = live ? "Live model (Flask API)" : "Built-in model"; m.classList.toggle("live", live); }
function markInvalid(message) {
  const map = {"Current score": "score", "Wickets lost": "wk", "Overs": "overs", "Balls after": "overs", "A chase": "overs", "Target score": "target"};
  Object.entries(map).forEach(([k, id]) => { if (message.startsWith(k)) $("#" + id).setAttribute("aria-invalid", "true"); });
}

async function predict(d) {
  try {
    const c = new AbortController(), t = setTimeout(() => c.abort(), 2000);
    const res = await fetch("/api/predict", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(d), signal: c.signal});
    clearTimeout(t);
    const j = await res.json();
    if (res.ok && typeof j.probability === "number") { setMode(true); return j; }
    if (res.status === 400 && j.error) throw new Error(j.error);
  } catch (e) { if (e.message && !/fetch|abort|JSON|Unexpected|Failed|Load failed|NetworkError/i.test(e.message)) throw e; }
  setMode(false);
  return localSituation(d);
}

$("#form").addEventListener("submit", async e => {
  e.preventDefault();
  $("#err").textContent = "";
  document.querySelectorAll("input").forEach(i => i.removeAttribute("aria-invalid"));
  const d = {batting_team: $("#bat").value, bowling_team: $("#bowl").value, venue: $("#venue").value,
    score: $("#score").value, wickets: $("#wk").value, overs: $("#overs").value, target: $("#target").value};
  $("#go").disabled = true;
  try { render(await predict(d)); }
  catch (err) { $("#err").textContent = err.message; markInvalid(err.message); }
  $("#go").disabled = false;
});
$("#reset").addEventListener("click", () => { setForm(DEFAULTS); clearOutput(); $("#err").textContent = ""; });

setForm(DEFAULTS); clearOutput(); setMode(false);
const M = MODEL.metrics;
$("#foot").textContent = `Logistic Regression trained on ${M.situations.toLocaleString("en-IN")} match situations (one per ball) from ${M.matches} IPL chases, 2008 to 2019. It is right about ${Math.round(M.accuracy * 100)}% of the time on matches it had not seen, and ${Math.round(M.cv * 100)}% in cross-validation. Early in a chase the estimate is less certain than late in it.`;
