const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "Request failed");
  return j;
};
const pct = v => (v * 100).toFixed(0) + "%";
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const COLORS = ["#2f5bd8", "#9dc0e6", "#e9a21a", "#1fa971", "#e0463c", "#7b5cd6", "#14a3b8"];
const charts = {};
const draw = (id, cfg) => {
  charts[id]?.destroy();
  cfg.options = { maintainAspectRatio: true, aspectRatio: cfg.type === "doughnut" ? 1.9 : 1.8, ...(cfg.options || {}) };
  charts[id] = new Chart($("#" + id), cfg);
};
let META = null;

/* ---------- tabs ---------- */
function showTab(name) {
  $$(".tab").forEach(t => t.classList.toggle("active", t.id === name));
  $$("nav a").forEach(a => a.classList.toggle("active", a.dataset.tab === name));
  if (name === "report") loadReport();
  if (name === "models") loadModels();
}
$$("nav a").forEach(a => a.addEventListener("click", e => { e.preventDefault(); history.replaceState(null, "", "#" + a.dataset.tab); showTab(a.dataset.tab); }));

/* ---------- dashboard ---------- */
async function loadDashboard() {
  const o = await api("/api/overview"), k = o.kpis;
  $("#kpis").innerHTML = [
    ["Total employees", k.total, ""], ["Actual attrition rate", pct(k.attrition_rate), "amber"],
    ["High-risk employees", k.high_risk, "red"], ["Avg performance", k.avg_performance.toFixed(2) + " / 5", "green"],
    ["Promotion candidates", k.promotion_candidates, "green"], ["Need training", k.training_needed, "amber"],
    ["Anomalies detected", k.anomalies, "red"],
  ].map(([l, v, c]) => `<div class="kpi ${c}"><span>${l}</span><b>${v}</b></div>`).join("");
  const d = o.departments;
  draw("chDept", { type: "bar", data: { labels: d.map(x => x.department), datasets: [{ label: "Avg attrition probability %", data: d.map(x => +(x.avg_attrition * 100).toFixed(1)), backgroundColor: COLORS[0], borderRadius: 6 }] }, options: { plugins: { legend: { display: false } } } });
  draw("chRisk", { type: "doughnut", data: { labels: ["Low", "Medium", "High"], datasets: [{ data: ["Low", "Medium", "High"].map(k => o.risk_levels[k]), backgroundColor: ["#1fa971", "#e9a21a", "#e0463c"] }] } });
  draw("chSeg", { type: "bar", data: { labels: o.segments.map(s => s.segment), datasets: [{ label: "Employees", data: o.segments.map(s => s.count), backgroundColor: COLORS.slice(0, 4), borderRadius: 6 }] }, options: { plugins: { legend: { display: false } } } });
  draw("chPerf", { type: "bar", data: { labels: o.performance_hist.labels, datasets: [{ label: "Employees", data: o.performance_hist.counts, backgroundColor: COLORS[3], borderRadius: 6 }] }, options: { plugins: { legend: { display: false } } } });
  draw("chDrv", { type: "bar", data: { labels: o.drivers.map(x => x.label), datasets: [{ data: o.drivers.map(x => x.importance), backgroundColor: COLORS[4], borderRadius: 6 }] }, options: { indexAxis: "y", aspectRatio: 3.2, plugins: { legend: { display: false } } } });
}

/* ---------- employees ---------- */
const state = { page: 1, sort: "risk_score", order: "desc" };
async function loadEmployees() {
  const p = new URLSearchParams({ q: $("#q").value, department: $("#fDept").value, risk: $("#fRisk").value, segment: $("#fSeg").value, anomaly: $("#fAnom").checked ? 1 : 0, sort: state.sort, order: state.order, page: state.page });
  const r = await api("/api/employees?" + p);
  $("#empTable tbody").innerHTML = r.rows.map(e => `<tr class="row" data-id="${e.employee_id}">
    <td>${e.employee_id}</td><td><b>${esc(e.name)}</b></td><td>${esc(e.department)}</td><td>${esc(e.job_role)}</td>
    <td><span class="badge ${e.risk_level}">${e.risk_level} · ${e.risk_score.toFixed(0)}</span></td><td>${pct(e.attrition_prob)}</td>
    <td>${e.performance_score.toFixed(2)}</td><td>${esc(e.segment)}</td><td>${e.anomaly_flag ? '<span class="badge flag">anomaly</span>' : ""}</td></tr>`).join("") || `<tr><td colspan="9" class="muted">No employees match.</td></tr>`;
  $$("#empTable tr.row").forEach(tr => tr.addEventListener("click", () => openEmployee(tr.dataset.id)));
  $("#pageInfo").textContent = `Page ${r.page} of ${r.pages} · ${r.total} employees`;
  $("#prev").disabled = r.page <= 1; $("#next").disabled = r.page >= r.pages;
}
let tmr; const refresh = () => { state.page = 1; loadEmployees(); };
$("#q").addEventListener("input", () => { clearTimeout(tmr); tmr = setTimeout(refresh, 250); });
["#fDept", "#fRisk", "#fSeg", "#fAnom"].forEach(s => $(s).addEventListener("change", refresh));
$("#prev").onclick = () => { state.page--; loadEmployees(); }; $("#next").onclick = () => { state.page++; loadEmployees(); };
$$("#empTable th[data-sort]").forEach(th => th.addEventListener("click", () => {
  const s = th.dataset.sort; state.order = state.sort === s && state.order === "desc" ? "asc" : "desc"; state.sort = s; loadEmployees();
}));

/* ---------- report rendering (drawer + what-if) ---------- */
const meter = (label, value, text, color) => `<div class="meter"><span>${label}</span><b>${text}</b><div class="bar"><i style="width:${Math.min(value * 100, 100)}%;background:${color || "var(--brand)"}"></i></div></div>`;
function drivers(list, cls, unit) {
  if (!list.length) return `<p class="muted">None significant.</p>`;
  const max = Math.max(...list.map(x => Math.abs(x.impact)), 0.001);
  return list.map(x => `<div class="drv ${cls}"><div>${esc(x.label)}<small>${esc(x.value)} (typical ${esc(x.typical)})</small></div>
    <div class="track"><i style="width:${Math.abs(x.impact) / max * 100}%"></i></div><b>${x.impact > 0 ? "+" : ""}${x.impact.toFixed(1)}</b></div>`).join("");
}
function renderReport(r) {
  const p = r.predictions, f = r.profile, lvlColor = { Low: "var(--low)", Medium: "var(--med)", High: "var(--high)" }[p.risk_level];
  const ex = r.explanation;
  const facts = [["Age", f.age], ["Department", f.department], ["Role", f.job_role], ["Level", f.job_level], ["Tenure", f.years_at_company + " yrs"],
    ["Since promotion", f.years_since_last_promotion + " yrs"], ["Monthly income", "₹" + Number(f.monthly_income).toLocaleString("en-IN")], ["Overtime", f.overtime],
    ["Job satisfaction", f.job_satisfaction + "/4"], ["Work-life balance", f.work_life_balance + "/4"], ["Manager rating", f.manager_rating + "/5"],
    ["Training hours", f.training_hours_last_year], ["Absent days", f.absenteeism_days], ["Projects", f.projects_completed]]
    .filter(x => x[1] !== undefined && !String(x[1]).startsWith("undefined") && !String(x[1]).startsWith("NaN"));
  return `<div class="rhead"><div><h2>${esc(f.name)} <small class="muted">${esc(f.employee_id || "")}</small></h2>
      <div class="muted">${esc(f.job_role)} · ${esc(f.department)}</div></div>
      <span class="badge ${p.risk_level}" style="font-size:14px">${p.risk_level} risk · ${p.risk_score.toFixed(0)}/100</span></div>
    <div class="summary">${esc(r.summary)}</div>
    <div class="meters">
      ${meter("Attrition probability", p.attrition_prob, pct(p.attrition_prob), lvlColor)}
      ${meter("Predicted performance", p.pred_performance / 5, p.pred_performance.toFixed(2) + " / 5", "var(--low)")}
      ${meter("Promotion suitability", p.promotion_prob, pct(p.promotion_prob) + (p.promotion_candidate ? " ✓" : ""))}
      ${meter("Training need", p.training_prob, pct(p.training_prob) + (p.training_needed ? " ✓" : ""), "var(--med)")}
      ${meter("Anomaly score", p.anomaly_score, p.anomaly_score.toFixed(2), "#7b5cd6")}
    </div>
    <div class="card" style="margin-bottom:14px"><h3>Segment: ${esc(p.segment)}</h3><div class="muted">${esc(p.segment_profile)}</div>
      ${p.anomaly_flag ? `<p><span class="badge flag">Anomaly</span> ${esc(p.anomaly_reasons)}</p>` : ""}</div>
    <div class="two">
      <div class="card"><h3>🔺 Why attrition risk is higher</h3><div class="muted" style="font-size:12px;margin-bottom:6px">Impact in percentage points vs. a typical employee</div>${drivers(ex.attrition.increases, "up")}</div>
      <div class="card"><h3>🟢 What keeps this employee</h3><div class="muted" style="font-size:12px;margin-bottom:6px">Protective factors</div>${drivers(ex.attrition.decreases.map(x => ({ ...x, impact: -x.impact })), "down")}</div>
    </div>
    <div class="two" style="margin-top:14px">
      <div class="card"><h3>Performance: top drivers</h3>${drivers(ex.performance.increases.slice(0, 4), "down")}${drivers(ex.performance.decreases.slice(0, 3).map(x => ({ ...x, impact: x.impact })), "up")}</div>
      <div class="card"><h3>Promotion: top drivers</h3>${drivers(ex.promotion.increases.slice(0, 4), "down")}${drivers(ex.promotion.decreases.slice(0, 3), "up")}</div>
    </div>
    <h3 style="margin-top:18px">HR recommendations</h3>
    <ul class="recs">${r.recommendations.map(x => `<li><span class="pri ${x.priority}">${x.priority}</span><span>${esc(x.text)}</span></li>`).join("")}</ul>
    <div class="card" style="margin-top:14px"><h3>Employee profile</h3><div class="facts">${facts.map(([a, b]) => `<div><span>${a}</span><b>${esc(b)}</b></div>`).join("")}</div></div>`;
}
async function openEmployee(id) {
  $("#drawer").classList.remove("hidden");
  $("#drawerContent").innerHTML = `<p class="muted">Generating AI report…</p>`;
  try { $("#drawerContent").innerHTML = renderReport(await api("/api/employee/" + id)); }
  catch (e) { $("#drawerContent").innerHTML = `<p>${esc(e.message)}</p>`; }
}
$("#closeDrawer").onclick = () => $("#drawer").classList.add("hidden");
$("#drawer").addEventListener("click", e => { if (e.target.id === "drawer") $("#drawer").classList.add("hidden"); });
document.addEventListener("keydown", e => { if (e.key === "Escape") $("#drawer").classList.add("hidden"); });

/* ---------- what-if ---------- */
function buildForm() {
  const m = META, num = m.numeric, cat = m.categorical, fill = m.fill;
  const html = [];
  cat.forEach(c => html.push(`<label>${m.labels[c]}<select data-f="${c}">${m.options[c].map(o => `<option ${o === fill[c] ? "selected" : ""}>${esc(o)}</option>`).join("")}</select></label>`));
  num.forEach(c => { const [lo, hi] = m.ranges[c]; html.push(`<label>${m.labels[c]} <small>(${lo}–${hi})</small><input type="number" data-f="${c}" value="${fill[c]}" min="${lo}" max="${hi}" step="any"></label>`); });
  $("#wiForm").innerHTML = html.join("");
  // keep job-role choices consistent with the selected department
  const dept = $('#wiForm [data-f="department"]'), role = $('#wiForm [data-f="job_role"]');
  const syncRoles = () => { role.innerHTML = (m.roles_by_dept[dept.value] || m.options.job_role).map(r => `<option>${esc(r)}</option>`).join(""); };
  dept.addEventListener("change", syncRoles); syncRoles();
}
$("#wiRun").onclick = async () => {
  const body = {};
  $$("#wiForm [data-f]").forEach(el => body[el.dataset.f] = el.type === "number" ? Number(el.value) : el.value);
  $("#wiResult").innerHTML = `<p class="muted">Scoring…</p>`;
  try { $("#wiResult").innerHTML = `<div class="card">${renderReport(await api("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }))}</div>`; }
  catch (e) { $("#wiResult").innerHTML = `<p style="color:var(--high)">${esc(e.message)}</p>`; }
};
$("#wiReset").onclick = () => { buildForm(); $("#wiResult").innerHTML = ""; };

/* ---------- workforce report ---------- */
async function loadReport() {
  const r = await api("/api/report");
  $("#reportBody").innerHTML = `
  <div class="card"><h3>Key findings</h3><ul class="list">${r.findings.map(x => `<li>${esc(x)}</li>`).join("")}</ul></div>
  <div class="card" style="margin-top:14px"><h3>Recommended actions</h3><ul class="list">${r.actions.map(x => `<li>${esc(x)}</li>`).join("")}</ul></div>
  <div class="two" style="margin-top:14px">
    <div class="card tablecard"><h3>Departments</h3><table class="mtable"><tr><th>Dept</th><th>Staff</th><th>Avg attrition</th><th>High risk</th></tr>
      ${r.departments.map(d => `<tr><td>${esc(d.department)}</td><td>${d.headcount}</td><td>${pct(d.avg_attrition)}</td><td>${d.high_risk}</td></tr>`).join("")}</table></div>
    <div class="card tablecard"><h3>Segments</h3><table class="mtable"><tr><th>Segment</th><th>Staff</th><th>Avg risk</th><th>Avg perf</th></tr>
      ${r.segments.map(d => `<tr><td>${esc(d.segment)}</td><td>${d.count}</td><td>${d.avg_risk.toFixed(1)}</td><td>${d.avg_perf.toFixed(2)}</td></tr>`).join("")}</table></div></div>
  <div class="card tablecard" style="margin-top:14px"><h3>Top 10 highest-risk employees</h3><table id="topRisk"><tr><th>ID</th><th>Name</th><th>Dept</th><th>Role</th><th>Risk</th><th>Attrition</th></tr>
    ${r.top_risk.map(e => `<tr class="row" data-id="${e.employee_id}"><td>${e.employee_id}</td><td><b>${esc(e.name)}</b></td><td>${esc(e.department)}</td><td>${esc(e.job_role)}</td><td><span class="badge High">${e.risk_score.toFixed(0)}</span></td><td>${pct(e.attrition_prob)}</td></tr>`).join("")}</table></div>`;
  $$("#topRisk tr.row").forEach(tr => tr.addEventListener("click", () => openEmployee(tr.dataset.id)));
}

/* ---------- model insights ---------- */
async function loadModels() {
  const r = await api("/api/model-info"), m = r.metrics;
  const clf = ["attr", "promo", "train"].map(k => m[k]);
  $("#modelBody").innerHTML = `
  <div class="card tablecard"><h3>Model performance (20% hold-out test set)</h3><table class="mtable">
   <tr><th>Model</th><th>Algorithm</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr>
   ${clf.map(x => `<tr><td>${x.title}</td><td>Random Forest Classifier</td><td>${pct(x.accuracy)}</td><td>${pct(x.precision)}</td><td>${pct(x.recall)}</td><td>${x.f1.toFixed(2)}</td><td>${x.roc_auc.toFixed(2)}</td></tr>`).join("")}
   <tr><td>${m.perf.title}</td><td>Random Forest Regressor</td><td colspan="2">R² = ${m.perf.r2.toFixed(2)}</td><td colspan="3">MAE = ${m.perf.mae.toFixed(2)} points</td></tr>
   <tr><td>Segmentation</td><td>K-Means (k=4)</td><td colspan="5">Silhouette = ${m.segmentation.silhouette.toFixed(2)}</td></tr>
   <tr><td>Anomaly detection</td><td>Isolation Forest</td><td colspan="5">~4% contamination; reasons explained per employee</td></tr></table>
   <p class="muted">Data: ${m.data.rows_raw} raw rows → ${m.data.rows_clean} after cleaning (${m.data.duplicates_removed} duplicates removed, ${m.data.missing_values_imputed} missing values imputed). Probabilities shown in the dashboard are out-of-fold (5-fold CV) so they are not inflated by training memorisation.</p></div>
  <h3 style="margin-top:18px">Visual analysis (Matplotlib &amp; Seaborn)</h3>
  <div class="figs">${r.figures.map(f => `<a href="/static/figures/${f}" target="_blank"><img src="/static/figures/${f}" alt="${f}"></a>`).join("")}</div>`;
}

/* ---------- init ---------- */
(async function init() {
  META = await api("/api/meta");
  META.options.department.forEach(d => $("#fDept").insertAdjacentHTML("beforeend", `<option>${esc(d)}</option>`));
  META.segments.forEach(s => $("#fSeg").insertAdjacentHTML("beforeend", `<option>${esc(s)}</option>`));
  buildForm();
  await Promise.all([loadDashboard(), loadEmployees()]);
  const h = location.hash.slice(1); if (h && $("#" + h)) showTab(h);
})();
