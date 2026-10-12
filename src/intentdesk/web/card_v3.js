"use strict";

const $ = (id) => document.getElementById(id);
let categories = {}, current = null;

async function api(path, body) {
  const response = await fetch(`/api/card-v3${path}`, body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json", "X-IntentDesk-Client": "local-web"},
    body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Check the request and try again.");
  return data;
}
function error(message) { $("error").textContent = message; $("error").hidden = !message; }
function element(tag, text, className) {
  const node = document.createElement(tag); node.textContent = text;
  if (className) node.className = className;
  return node;
}
function show(item) {
  current = item;
  $("detail").hidden = false;
  $("request-copy").textContent = item.text;
  $("suggestion").textContent = item.analysis.name;
  $("reason").textContent = item.analysis.reason;
  $("scores").replaceChildren();
  for (const candidate of item.analysis.candidates) {
    const row = element("li", "");
    row.append(element("span", categories[candidate.intent].name), element("span", `${(candidate.score * 100).toFixed(1)}%`, "score"));
    $("scores").append(row);
  }
  $("decision").value = item.decisions.length ? item.decisions.at(-1).label : item.analysis.label;
  $("note").value = "";
  $("saved").textContent = "";
  $("history").replaceChildren(...item.decisions.map(d => element("li", `${categories[d.label].name}${d.note ? ` — ${d.note}` : ""}`)));
}
async function queue() {
  const data = await api("/requests");
  $("queue").replaceChildren();
  if (!data.items.length) $("queue").append(element("p", "No pending v3 requests. Analyze a fictional example to begin.", "empty"));
  for (const item of data.items) {
    const button = element("button", item.text, "queue-item"); button.type = "button";
    button.addEventListener("click", async () => { try { show(await api(`/requests/${item.id}`)); error(""); } catch (e) { error(e.message); } });
    $("queue").append(button);
  }
}
$("analyze").addEventListener("submit", async (event) => {
  event.preventDefault(); $("analyze-button").disabled = true; error("");
  // Hide the previous result so a failed analysis cannot look like a new answer.
  current = null; $("detail").hidden = true;
  try { show(await api("/requests", {text: $("request").value})); await queue(); }
  catch (e) { error(e.message); }
  finally { $("analyze-button").disabled = false; }
});
$("review").addEventListener("submit", async (event) => {
  event.preventDefault(); if (!current) return;
  $("save-button").disabled = true; error("");
  try {
    const item = await api(`/requests/${current.id}/review`, {label: $("decision").value, note: $("note").value, expected_version: current.version});
    show(item); $("saved").textContent = `${item.handoff.team}: ${item.handoff.status}.`;
    await queue();
  } catch (e) { error(e.message); }
  finally { $("save-button").disabled = false; }
});
$("refresh").addEventListener("click", () => queue().catch(e => error(e.message)));
(async () => {
  try {
    const status = await api("/status"); categories = status.categories;
    for (const [label, category] of Object.entries(categories)) {
      const option = element("option", category.name); option.value = label; $("decision").append(option);
    }
    if (!status.artifact_present) error("Card v3 needs local training. Run intentdesk train-card-v3, then retry.");
    await queue();
  } catch (e) { error(e.message); }
})();
