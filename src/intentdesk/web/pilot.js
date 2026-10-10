"use strict";
const byId=(id)=>document.getElementById(id);
let config=null, session=null;
const fail=(error)=>{byId("pilot-error").textContent=error.message||"The comparison could not continue. Retry or resume with your reviewer code.";byId("pilot-error").hidden=false;};
async function api(path,body){
  const response=await fetch(path,body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json","X-IntentDesk-Client":"local-web"},body:JSON.stringify(body)});
  const result=await response.json();
  if(!response.ok)throw new Error(typeof result.detail==="string"?result.detail:"Check the fields and retry.");
  return result;
}
function remember(value){if(value.kind!=="human")return;try{localStorage.setItem("intentdesk-pilot-session",value.id);}catch{/* The reviewer code also resumes the server-side session. */}}
function focusHeading(id){byId(id).focus();}
function metrics(summary){
  const percent=(x)=>x===null?"No decisions":`${(x*100).toFixed(1)}%`;
  const seconds=(x)=>x===null?"No timed decisions":`${x.toFixed(1)} s`;
  const rows=[
    ["Decisions",(m)=>m.count], ["Correct categories",(m)=>`${m.correct} / ${m.count}`],
    ["Category accuracy",(m)=>percent(m.accuracy)], ["Needs clarification",(m)=>m.uncertain],
    ["Mean decision time",(m)=>seconds(m.mean_decision_seconds)],
    ["Median decision time",(m)=>seconds(m.median_decision_seconds)],
    ["Tickets included in timing",(m)=>m.timed_count], ["Timing exclusions",(m)=>m.timing_exclusions],
    ["Changed model suggestions",(m,c)=>c==="manual"?"Not shown":m.suggestion_overrides],
    ["Mean model wait",(m,c)=>c==="manual"?"No inference":seconds(m.mean_inference_seconds)]
  ];
  byId("pilot-metrics").replaceChildren();
  for(const [name,format] of rows){const tr=document.createElement("tr");const th=document.createElement("th");th.scope="row";th.textContent=name;tr.append(th);for(const c of ["manual","assisted"]){const td=document.createElement("td");td.textContent=String(format(summary[c],c));tr.append(td);}byId("pilot-metrics").append(tr);}
}
function render(value){
  session=value;remember(value);byId("pilot-setup").hidden=true;byId("pilot-session").hidden=value.finished;byId("pilot-results").hidden=!value.finished;
  if(value.finished){
    metrics(value.summary);byId("pilot-result-note").textContent=value.kind==="smoke"?"Automated verification session. Excluded from human comparison results.":`Demonstration data · 14 completed decisions · last prepared handoff: ${value.last_handoff.category} → ${value.last_handoff.team}. Nothing sent externally.`;
    focusHeading("results-title");updateReportCount().catch(fail);return;
  }
  byId("pilot-progress").textContent=`${value.completed} of ${value.total} saved · reviewer ${value.reviewer}${value.kind==="smoke"?" · verification session":""}`;
  const trial=value.trial;
  byId("pilot-decision-form").hidden=!trial;byId("pilot-text").hidden=!trial;
  byId("pilot-advisory").hidden=!trial?.suggestion;byId("manual-help").hidden=!trial||trial.condition!=="manual";
  byId("pilot-handoff").hidden=Boolean(trial)||!value.last_handoff;
  if(!trial){byId("pilot-condition").textContent="Saved";byId("ticket-title").textContent="Decision recorded";if(value.last_handoff){const h=value.last_handoff;byId("pilot-handoff-copy").textContent=`${h.category} → ${h.team}`;}focusHeading("ticket-title");return;}
  byId("ticket-title").textContent=`Review ticket ${trial.position+1}`;byId("pilot-condition").textContent=trial.condition==="manual"?"Manual review":"Model-assisted review";
  byId("pilot-text").textContent=trial.text;byId("pilot-label").value="";byId("pilot-interrupted").checked=false;byId("pilot-save-state").textContent="";byId("pilot-team-preview").textContent="Your decision determines the prepared handoff.";
  if(trial.suggestion){byId("pilot-suggestion").textContent=trial.suggestion.name;byId("pilot-reason").textContent=trial.suggestion.reason;}
  byId("pilot-session").classList.remove("detail-updated");void byId("pilot-session").offsetWidth;byId("pilot-session").classList.add("detail-updated");focusHeading("ticket-title");
}
async function openNext(){render(await api(`/api/pilot/sessions/${session.id}/next`,{}));}
async function updateReportCount(){const report=await api("/api/pilot/report");byId("pilot-human-count").textContent=`Completed human comparisons: ${report.completed_human_sessions} · unfinished: ${report.incomplete_human_sessions} · verification sessions excluded: ${report.smoke_sessions_excluded}`;}
byId("pilot-label").addEventListener("change",()=>{const category=config.categories[byId("pilot-label").value];byId("pilot-team-preview").textContent=category?`Prepared team: ${category.team}. Nothing is sent externally.`:"Your decision determines the prepared handoff.";});
byId("pilot-use").addEventListener("click",()=>{byId("pilot-label").value=session.trial.suggestion.label;byId("pilot-label").dispatchEvent(new Event("change"));byId("pilot-label").focus();});
byId("pilot-start-form").addEventListener("submit",async(event)=>{
  event.preventDefault();byId("pilot-error").hidden=true;const button=byId("pilot-start");button.disabled=true;button.textContent="Opening comparison…";
  try{session=await api("/api/pilot/sessions",{reviewer:byId("pilot-reviewer").value,kind:"human"});if(session.finished||session.trial)render(session);else await openNext();}
  catch(error){fail(error);}finally{button.disabled=false;button.textContent="Start or resume comparison";}
});
byId("pilot-next").addEventListener("click",async()=>{const button=byId("pilot-next");button.disabled=true;button.textContent="Opening ticket…";byId("pilot-error").hidden=true;try{await openNext();}catch(error){fail(error);}finally{button.disabled=false;button.textContent="Open next ticket";}});
byId("pilot-decision-form").addEventListener("submit",async(event)=>{
  event.preventDefault();byId("pilot-error").hidden=true;const button=byId("pilot-save");button.disabled=true;button.textContent="Saving decision…";
  try{const value=await api(`/api/pilot/sessions/${session.id}/decision`,{trial_id:session.trial.id,label:byId("pilot-label").value,interrupted:byId("pilot-interrupted").checked});render(value);}
  catch(error){fail(error);}finally{button.disabled=false;button.textContent="Save category & prepare handoff";}
});
async function initialize(){
  config=await api("/api/pilot");
  for(const [label,category] of Object.entries(config.categories)){
    const option=document.createElement("option");option.value=label;option.textContent=category.name;byId("pilot-label").append(option);
    const term=document.createElement("dt");term.textContent=category.name;const definition=document.createElement("dd");definition.textContent=`${category.definition} Prepared team: ${category.team}.`;byId("pilot-guide").append(term,definition);
  }
  byId("pilot-start").disabled=false;await updateReportCount();
  let saved=new URLSearchParams(window.location.search).get("session");try{saved=saved||localStorage.getItem("intentdesk-pilot-session");}catch{/* Reviewer codes remain available without browser storage. */}
  if(saved){
    try{const previous=await api(`/api/pilot/sessions/${encodeURIComponent(saved)}`);if(previous.run_id===config.run_id){byId("pilot-reviewer").value=previous.reviewer;render(previous);}}
    catch{byId("pilot-human-count").textContent+=" · Start or resume using your reviewer code.";}
  }
}
initialize().catch(fail);
