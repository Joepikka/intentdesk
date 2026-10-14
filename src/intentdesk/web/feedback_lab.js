"use strict";
const byId = id => document.getElementById(id);
const displayName = name => name.replaceAll("_", " ");
const starLabel = value => `${value} ${value === 1 ? "star" : "stars"}`;
function error(message) { byId("lab-error").textContent = message; byId("lab-error").hidden = !message; }
async function api(path, body) {
  const response = await fetch(`/api/ml-lab${path}`, body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json","X-IntentDesk-Client":"local-web"},body:JSON.stringify(body)});
  const value = await response.json(); if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : "Check the request fields."); return value;
}
async function loadStatus() {
  const status = await api("/status");
  byId("lab-submit").disabled = !status.ready;
  if (!status.ready) { byId("lab-data").textContent = "Models are not ready. Run the local feedback preparation and training commands, then reload this page."; return; }
  if (!status.final_report) { byId("lab-data").textContent = "Training is complete; the frozen final check has not completed. No test score is claimed yet."; return; }
  const report = status.final_report;
  byId("lab-data").textContent = `${report.data.audit.source_rows.toLocaleString()} source reviews audited; ${(report.data.raw_bytes/1e9).toFixed(2)} GB of pinned review data. Product groups and duplicate fingerprints are separated across partitions.`;
  byId("lab-table").replaceChildren();
  for (const [name, result] of Object.entries(report.classification)) {
    const row = document.createElement("tr"); row.dataset.selected = String(name === status.selected_classifier);
    for (const value of [displayName(name)+(name===status.selected_classifier ? " · selected before test" : ""),result.fit_rows.toLocaleString(),`${(result.accuracy*100).toFixed(1)}%`,result.macro_f1.toFixed(3)]) {
      const cell = document.createElement("td"); cell.textContent = value; row.append(cell);
    }
    byId("lab-table").append(row);
  }
  byId("lab-training").textContent = Object.entries(report.scratch_models).map(([name,model]) => `${displayName(name)}: ${model.trainable_parameters.toLocaleString()} trainable parameters; random initialization; ${model.history.length} completed epochs, checkpoint selected at epoch ${model.best_epoch}`).join(". ")+".";
  const improvement = status.improvement_report;
  byId("lab-improvement").hidden = !improvement;
  if (improvement) {
    byId("lab-improvement-data").textContent = `${improvement.fresh_check.rows.toLocaleString()} reviews from ${improvement.fresh_check.products.toLocaleString()} previously untested products. Excluding previously tested products shifts the population toward less-reviewed products.`;
    byId("lab-improvement-table").replaceChildren();
    for (const [name, result] of Object.entries(improvement.classification)) {
      const row = document.createElement("tr"); row.dataset.selected = String(name === improvement.selected_classifier);
      for (const value of [displayName(name)+(name===improvement.selected_classifier ? " · selected before fresh check" : ""),`${(result.accuracy*100).toFixed(2)}%`,result.macro_f1.toFixed(4)]) {
        const cell = document.createElement("td"); cell.textContent = value; row.append(cell);
      }
      byId("lab-improvement-table").append(row);
    }
  }
}
byId("lab-form").addEventListener("submit",async event => {
  event.preventDefault(); byId("lab-submit").disabled = true; byId("lab-result").hidden = true; error("");
  try {
    const result = await api("/analyze",{text:byId("lab-text").value});
    byId("lab-selected").textContent = displayName(result.selected_classifier)+(result.release === "feedback_v2" ? " · feedback v2" : "");
    byId("lab-stars").textContent = starLabel(result.predicted_stars);
    byId("lab-regression").textContent = `${result.linear_regression_stars.toFixed(2)} stars (unclipped)`;
    byId("lab-neural").textContent = `${displayName(result.scratch_transformer)}: ${starLabel(result.scratch_predicted_stars)}${result.scratch_input_truncated ? "; only the review prefix was used" : ""}`;
    byId("lab-cluster").textContent = `Group ${result.cluster}: ${result.cluster_keywords.join(", ")}`;
    byId("lab-result").hidden = false;
  } catch (failure) { error(failure.message); }
  finally { byId("lab-submit").disabled = false; }
});
loadStatus().catch(failure => error(failure.message));
