"use strict";
import {clampQueueOffset} from "./queue.mjs";
const $ = (id) => document.getElementById(id);
const state = {status:"pending",offset:0,limit:20,selected:null,labels:[],total:0};
const human = (label) => label.replaceAll("_", " ");
function fail(error){$("error").textContent=error.message || "Something went wrong. Refresh and try again.";$("error").hidden=false;}
async function api(path,body){
  const response=await fetch(path,body ? {method:"POST",headers:{"Content-Type":"application/json","X-IntentDesk-Client":"local-web"},body:JSON.stringify(body)} : {});
  if(!response.ok){const result=await response.json().catch(()=>({}));throw new Error(typeof result.detail==="string" ? result.detail : "The local app could not complete this request.");}
  return response.json();
}
async function refresh(){
  const [status,firstPage]=await Promise.all([api("/api/status"),api(`/api/requests?status=${state.status}&limit=${state.limit}&offset=${state.offset}`)]);
  let queue=firstPage;
  const validOffset=clampQueueOffset(state.offset,queue.total,state.limit);
  if(validOffset!==state.offset){state.offset=validOffset;queue=await api(`/api/requests?status=${state.status}&limit=${state.limit}&offset=${state.offset}`);}
  state.labels=[...status.classes,...status.review_labels];state.total=queue.total;
  $("connection").textContent="Connected";$("pending-count").textContent=status.counts.pending;$("reviewed-count").textContent=status.counts.reviewed;
  $("queue").replaceChildren();
  if(!queue.items.length){const empty=document.createElement("p");empty.className="empty";empty.textContent=state.status==="pending" ? "No requests waiting. Add one to start reviewing." : "No decisions yet. Your saved reviews will appear here.";$("queue").append(empty);}
  for(const item of queue.items){
    const button=document.createElement("button");button.className="queue-item";button.type="button";button.setAttribute("aria-current",String(item.id===state.selected));
    const copy=document.createElement("span");copy.className="queue-copy";copy.textContent=item.text;
    const meta=document.createElement("span");meta.className="queue-meta";meta.textContent=`${new Date(item.created_at).toLocaleString()} · ${item.status}`;
    button.append(copy,meta);button.addEventListener("click",()=>select(item.id).catch(fail));$("queue").append(button);
  }
  $("page").textContent=queue.total ? `${state.offset+1}–${Math.min(state.offset+state.limit,queue.total)} of ${queue.total}` : "0 requests";
  $("previous").disabled=state.offset===0;$("next").disabled=state.offset+state.limit>=queue.total;
}
async function select(id){
  const item=await api(`/api/requests/${id}`);state.selected=id;
  const fragment=$("detail-template").content.cloneNode(true);
  fragment.querySelector("h2").id="detail-title";
  fragment.querySelector(".request-status").textContent=item.status==="pending" ? "Needs review" : "Reviewed";
  fragment.querySelector(".request-copy").textContent=item.text;
  fragment.querySelector(".prediction-reason").textContent=item.analysis.reason;
  const scope=item.analysis.scope_assessment?.status;
  const scopeCopy={likely_out_of_scope:"Likely unsupported",uncertain:"Scope uncertain",supported_candidate:"Supported category suggested",unavailable:"Scope assessment unavailable"};
  fragment.querySelector(".scope-status").textContent=scopeCopy[scope] || "Earlier assessment · analyze again for the new scope check";
  fragment.querySelector(".scope-status").dataset.scope=scope || "unavailable";
  if(!scope){fragment.querySelector(".prediction-reason").textContent="This request was analyzed before the scope update. Analyze it again to check whether a category applies.";}
  fragment.querySelector(".reanalyze").addEventListener("click",()=>{if($("analyze").disabled)return;$("request-text").value=item.text;$("submit-form").requestSubmit();});
  if(scope==="likely_out_of_scope"){fragment.querySelector(".known-suggestions").hidden=true;}
  const label=fragment.querySelector("#review-label");
  for(const value of state.labels){const option=document.createElement("option");option.value=value;fragment.querySelector("datalist").append(option);}
  for(const candidate of item.analysis.candidates){
    const row=document.createElement("li");const pick=document.createElement("button");pick.type="button";pick.textContent=human(candidate.intent);pick.title=`Use ${candidate.intent}`;
    pick.addEventListener("click",()=>{label.value=candidate.intent;label.focus();});
    const score=document.createElement("span");score.className="score";score.textContent=`${(candidate.score*100).toFixed(1)}%`;row.append(pick,score);fragment.querySelector(".suggestions").append(row);
  }
  if(!item.analysis.candidates.length){const empty=document.createElement("li");empty.textContent="No suggestion: review the full request or submit a shorter version.";fragment.querySelector(".suggestions").append(empty);}
  const previous=item.decisions.at(-1);
  if(previous){label.value=previous.label;fragment.querySelector("#review-note").value=previous.note;}
  else if(scope==="likely_out_of_scope"){label.value="oos";}
  const history=fragment.querySelector(".history ol");
  if(!item.decisions.length){const row=document.createElement("li");row.textContent="No decisions recorded.";history.append(row);}
  for(const event of item.decisions){const row=document.createElement("li");row.textContent=`${new Date(event.created_at).toLocaleString()} · ${event.label}${event.note ? ` — ${event.note}` : ""}`;history.append(row);}
  const form=fragment.querySelector(".review-form");
  form.addEventListener("submit",async(event)=>{
    event.preventDefault();$("error").hidden=true;const button=form.querySelector("button[type=submit]");
    if(!state.labels.includes(label.value)){fail(new Error("Choose an intent from the list, oos, or uncertain."));label.focus();return;}
    button.disabled=true;button.textContent="Saving…";
    try{await api(`/api/requests/${id}/review`,{label:label.value,note:form.elements.note.value,expected_version:item.version});await select(id);$("detail").querySelector(".save-message").textContent="Decision saved on this PC.";const heading=$("detail-title");heading.tabIndex=-1;heading.focus();}
    catch(error){fail(error);button.disabled=false;button.textContent="Save decision";}
  });
  $("detail").replaceChildren(fragment);$("detail").classList.remove("detail-updated");void $("detail").offsetWidth;$("detail").classList.add("detail-updated");await refresh();
}
$("submit-form").addEventListener("submit",async(event)=>{
  event.preventDefault();$("error").hidden=true;const button=$("analyze");button.disabled=true;button.textContent="Analyzing locally…";
  try{const item=await api("/api/requests",{text:$("request-text").value});state.status="pending";state.offset=0;updateTabs();await refresh();await select(item.id);$("request-text").value="";}
  catch(error){fail(error);}finally{button.disabled=false;button.textContent="Analyze & add to queue";}
});
function updateTabs(){for(const name of ["pending","reviewed"]){$(name).setAttribute("aria-pressed",String(name===state.status));}}
for(const name of ["pending","reviewed"]){$(name).addEventListener("click",()=>{state.status=name;state.offset=0;updateTabs();refresh().catch(fail);});}
$("refresh").addEventListener("click",()=>refresh().catch(fail));
$("previous").addEventListener("click",()=>{state.offset=Math.max(0,state.offset-state.limit);refresh().catch(fail);});
$("next").addEventListener("click",()=>{state.offset+=state.limit;refresh().catch(fail);});
refresh().catch(error=>{$("connection").textContent="Unavailable";fail(error);});
api("/api/search/status").then(status=>{$("search-state").textContent=status.ready ? `${status.passages.toLocaleString()} passages` : "Index not ready";$("search-button").disabled=!status.ready;}).catch(fail);
$("search-form").addEventListener("submit",async(event)=>{
  event.preventDefault();const button=$("search-button");button.disabled=true;button.textContent="Searching…";$("search-message").textContent="";$("search-results").replaceChildren();
  try{
    const found=await api("/api/search",{query:$("search-query").value});
    $("search-message").textContent=found.results.length ? `${found.results.length} passages · ${(found.elapsed_ms/1000).toFixed(2)} seconds` : "No matches. Try different words.";
    for(const result of found.results){const row=document.createElement("li");const source=document.createElement("h3");source.textContent=`MS MARCO passage ${result.passage_id}`;const text=document.createElement("p");text.textContent=result.text;row.append(source,text);$("search-results").append(row);}
  }catch(error){$("search-message").textContent=error.message;}finally{button.disabled=false;button.textContent="Search passages";}
});
