import requirements from "./requirements.json";
import manual from "./manual.json";
import {basicSetup} from "codemirror";
import {indentWithTab, undo, redo, undoDepth, redoDepth, historyField} from "@codemirror/commands";
import {EditorView, keymap} from "@codemirror/view";
import {indentUnit} from "@codemirror/language";
import {EditorState, Prec} from "@codemirror/state";
import {python} from "@codemirror/lang-python";
import {rust} from "@codemirror/lang-rust";

const app = document.querySelector("#app"), dialog = document.querySelector("#dialog");
const outsideDialog = event => {
  const box = dialog.getBoundingClientRect();
  return event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom;
};
let manualBackdropPress = false;
dialog.addEventListener("pointerdown", event => {
  manualBackdropPress = dialog.open && dialog.dataset.kind === "manual" && event.target === dialog && outsideDialog(event);
});
dialog.addEventListener("click", event => {
  if (manualBackdropPress && dialog.open && dialog.dataset.kind === "manual" && event.target === dialog && outsideDialog(event)) dialog.close();
  manualBackdropPress = false;
});
dialog.addEventListener("keydown", event => {
  if (dialog.dataset.kind === "manual" && event.key === "Escape") {
    event.preventDefault(); event.stopPropagation(); dialog.close();
  }
}, true);
dialog.addEventListener("cancel", event => {
  if (dialog.dataset.kind === "manual") { event.preventDefault(); dialog.close(); }
});
const basePath = document.querySelector('meta[name="study-base"]')?.content || "";
const studyPath = location.pathname.slice(basePath.length);
const token = studyPath.match(/^\/p\/([\w-]+)$/)?.[1];
const isAdmin = studyPath === "/admin";
const params = new URLSearchParams(location.hash.slice(1));
if (isAdmin && params.get("key")) {
  sessionStorage.setItem("study-admin", params.get("key"));
  history.replaceState(null, "", basePath+"/admin");
}
let locale = localStorage.getItem("study-locale") || "en";
let state, editor, revision = 0, savedCode = "", saving = Promise.resolve(), saveTimer;
let currentKey = "", result = null, resultsByPurpose = {}, activeRun = null, outputTab = "examples", syncAt = 0, syncedRemaining = null;
let noticeTimer, adminDetail = null, dashboardTimer, refreshBusy = false, pendingTransition = false;
let stateGeneration = 0;
const pageId = Array.from(crypto.getRandomValues(new Uint32Array(4)), n=>n.toString(16)).join("-");
let pageClosed = false;
let editorHistoryKey = "";
const editorHistories = new Map();
const tr = (en, zh) => locale === "zh" ? zh : en;
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
// Escape all source text before adding the fixed emphasis markup.
const descriptionHtml = task => {
  const language = locale === "zh" ? 1 : 0;
  let html = esc(task.descriptions[language]);
  for (const phrase of task.requirements?.[language] || requirements[task.id]?.[language] || []) {
    html = html.replace(esc(phrase), match => `<strong class="task-requirement">${match}</strong>`);
  }
  return html;
};
const pretty = v => JSON.stringify(v, null, 2);
const fmt = s => `${Math.floor(Math.max(0,s)/60)}:${String(Math.floor(Math.max(0,s)%60)).padStart(2,"0")}`;
const conditionName = c => c === "zinnia" ? "Zinnia · Python" : "Rust · RISC Zero";
const assignmentName = p => p.design==="crossover-12x6-v1"?tr("Crossover · 3 tasks per environment","交叉设计 · 每种环境三题"):conditionName(p.condition);

const title = t => t.titles[locale === "zh" ? 1 : 0];
const shortNames = {calibrate:["Calibration","校准"],background:["Background","背景值"],threshold_energy:["Energy","能量"],distances:["Distances","距离"],pool:["Image blocks","图像分块"],smooth:["Smoothing","平滑"]};
Object.assign(shortNames, {channel_energy_v2:["Channel energy","通道能量"],centered_gram_v2:["Centered products","中心化乘积"],weighted_distances_v2:["Weighted distances","加权距离"],convolution_v2:["Image filters","图像滤波"],block_covariance_v2:["Block covariance","区块协方差"],masked_diffusion_v2:["Masked updates","掩码更新"]});
Object.assign(shortNames,{channel_energy_v3:["Batch calibration","批次校准"],cross_products_v3:["Cross products","交叉乘积"],projected_distances_v3:["Projected distances","投影距离"],strided_filters_v3:["Strided filters","跨步滤波"],rectangular_covariance_v3:["Block covariance","区块协方差"],batch_diffusion_v3:["Batch diffusion","批次扩散"]});
shortNames.strided_filters_v4=["Strided dilated filters","跨步膨胀滤波"];
const taskName = id => (shortNames[id] || [id,id])[locale === "zh" ? 1 : 0];
const $ = id => document.getElementById(id);

function notify(message) {
  const box = $("notification"); box.textContent = message; box.style.display = "block";
  clearTimeout(noticeTimer); noticeTimer = setTimeout(() => box.style.display = "none", 6500);
}
async function api(path, data, admin=false) {
  const headers = {};
  if (admin) headers.Authorization = `Bearer ${sessionStorage.getItem("study-admin") || ""}`;
  if (data !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(basePath+path, {method:data === undefined ? "GET":"POST",headers,body:data === undefined ? undefined : JSON.stringify(data),cache:"no-store"});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}
const participant = (path,data) => api(`/api/p/${token}/${path}`,data);
const runSafely = fn => async (...args) => {try {await fn(...args);} catch(e) {notify(e.message);}};
function listen(id,fn,event="click") {const element=$(id); if(element) element.addEventListener(event,runSafely(fn));}
function event(kind,payload={}) {if(token) participant("event",{kind,payload}).catch(()=>{});}
function languageButtons() {return `<div class="language" role="group" aria-label="${tr("Interface language","界面语言")}"><button data-lang="en" lang="en" aria-pressed="${locale==="en"}" class="${locale==="en"?"active":""}">English</button><button data-lang="zh" lang="zh-Hans" aria-pressed="${locale==="zh"}" class="${locale==="zh"?"active":""}">中文</button></div>`;}
function programmingBadge() {
  const isPython = state?.condition === "zinnia";
  const language = isPython ? "Python" : "Rust";
  return `<div class="programming-badge ${isPython ? "python" : "rust"}" data-programming-language="${language}" aria-label="${tr("Current programming language", "当前编程语言")}: ${language}"><svg class="programming-mark" viewBox="0 0 44 44" aria-hidden="true" focusable="false"><rect x="1" y="1" width="42" height="42" rx="10" fill="currentColor"/><text x="22" y="28" text-anchor="middle" fill="white" font-family="monospace" font-weight="700" font-size="20">${isPython ? "Py" : "Rs"}</text></svg><span><span class="programming-caption">${tr("Code in", "编程语言")}</span><strong>${language}</strong></span></div>`;
}
function topbar(extra="") {
  return `<header class="topbar">${state && ["tutorial","practice","ready","task"].includes(state.stage) ? programmingBadge() : `<div class="wordmark">ARRAY <span>/</span> STUDY</div>`}<div class="divider"></div><div class="session-meta"><span class="label muted">${esc(state?.label || tr("Research workspace","研究工作台"))}</span>${state?`${state.preview?`<span class="badge gray">${tr("Preview","预览")}</span>`:""}`:""}</div><div class="spacer"></div>${extra}${languageButtons()}</header>`;
}
function bindLanguage() {
  document.querySelectorAll("[data-lang]").forEach(button=>button.onclick=runSafely(async()=>{
    if (pendingTransition || button.dataset.lang===locale) return;
    const manualOpen=dialog.open && dialog.dataset.kind==="manual";
    if(manualOpen) dialog.close();
    const consentChecked=$("consent")?.checked;
    if (editor && !state?.session?.paused_at) await flushSave();
    locale=button.dataset.lang; localStorage.setItem("study-locale",locale); document.documentElement.lang=locale==="zh"?"zh-Hans":"en";
    if(token&&state) {await transition("locale",{locale}); if(consentChecked && $("consent")) {$("consent").checked=true; $("consent-next").disabled=false;}}
    else if(isAdmin) {adminDetail=null; await renderAdmin();} else landing();
    if(manualOpen && !state?.session?.paused_at) openReference();
  }));
}
function steps() {
  return `<nav class="progressbar" aria-label="${tr("Study progress","研究进度")}"><span class="step ${state.stage==="practice"?"current":"done"}"><span class="dot">${state.stage==="practice"?"P":"✓"}</span><span class="step-name">${tr("Practice","练习")}</span></span>${state.order.map((id,i)=>`<span class="step-line"></span><span class="step ${i<state.task_index?"done":i===state.task_index&&state.stage==="task"?"current":""}" title="${esc(taskName(id))}"><span class="dot">${i<state.task_index?"✓":i+1}</span><span class="step-name">${esc(taskName(id))}</span></span>`).join("")}</nav>`;
}
function studyIntroduction() {
  return `<h1>${tr("About this study","研究说明")}</h1><p class="study-llm-rule"><strong>${tr("Do not use large language models (LLMs) or AI coding assistants to generate answers or solution code for this study, including practice tasks.","本研究禁止使用大语言模型（LLM）或 AI 编程助手生成答案或解题代码，包括练习任务。")}</strong></p><p class="lead">${tr("This study examines how people understand, implement, and debug array-based programming tasks. We use task outcomes, time spent, and participant feedback to understand the programming process and difficulties encountered.","本研究旨在了解参与者如何理解、实现和调试数组编程任务。我们将结合任务结果、完成时间和参与者反馈，分析编程过程及其中遇到的困难。")}</p><h2>${tr("What you will do","参与内容")}</h2><p>${tr("You will answer questions about your programming experience, read the tutorials, complete the required untimed practice, and work on six timed programming tasks. In the crossover study, you practice in both environments and complete three tasks in each. After each submitted task, you will be asked to rate its difficulty and your confidence in your solution.","您将回答有关编程经验的问题、阅读教程、完成所需的不限时练习，并进行六个计时编程任务。在交叉研究中，您会先练习两种环境，再在每种环境中各完成三题。每次提交任务后，您需要评价任务难度和对所提交代码的信心。")}</p><p>${tr("Use English / 中文 in the header to change the interface language at any time. Reopen your personal link to return to your saved study progress. The timer pauses when you close the page or click Pause. Reopening the link resumes an automatic pause; after clicking Pause, click Resume to continue.","您可以随时使用页首的 English / 中文 切换界面语言。重新打开您的专属链接即可返回已保存的研究进度。关闭页面或点击“暂停”后，计时会停止。重新打开链接会恢复因关闭页面而暂停的计时；主动暂停后，请点击“继续”。")}</p><h2>${tr("How data is recorded","数据记录方式")}</h2><p>${tr("The website automatically saves code drafts and records submitted code, test inputs, execution results, errors, task timings, questionnaire answers, ratings, and comments on the study server under your participant code. It also records timestamped actions such as saving, running code, opening reference material, pasting into the editor, and leaving or returning to the study tab. Pasted code is included in saved drafts; clipboard contents are not separately collected. Each execution saves a code version; researchers assess these versions and the final submission using additional tests to understand how solutions develop. Drafts and unfinished background answers are also stored in your browser for recovery.","网站会以您的参与者编号在研究服务器上自动保存代码草稿，并记录提交的代码、测试输入、执行结果、错误、任务用时、问卷回答、评分和评论。网站还会记录保存、运行代码、打开参考资料、向编辑器粘贴内容以及离开或返回研究标签页等操作及其时间戳。粘贴的代码会包含在保存的草稿中，但不会单独收集剪贴板内容。每次运行都会保存一份代码版本；研究人员会用额外测试评估这些版本及最终提交，以分析解题过程。浏览器也会保存草稿和尚未提交的背景问卷回答，以便恢复。")}</p><p>${tr("The website does not record your screen or activity in other applications or websites. Participation is voluntary, and you may stop at any time. Stopping does not automatically delete data already saved. Contact the researcher who shared your link with questions or requests to remove your data.","网站不会录制您的屏幕，也不会记录其他应用或网站中的活动。参与完全自愿，您可以随时停止。停止参与不会自动删除已保存的数据。如有疑问或需要删除数据，请联系发送此链接的研究人员。")}</p>`;
}
function landing() {
  app.innerHTML=topbar()+`<main class="welcome"><div class="card">${studyIntroduction()}<p>${tr("Open the personal link shared by your researcher to begin.","请打开研究人员发送的专属链接以开始参与。")}</p><a class="button" href="${basePath}/admin">${tr("Researcher sign in","研究人员登录")}</a></div></main>`;
  bindLanguage();
}
function rememberEditor() {
  if(!editor || !editorHistoryKey) return;
  const snapshot=editor.state.toJSON({history:historyField});
  editorHistories.set(editorHistoryKey,snapshot);
  try {sessionStorage.setItem(editorHistoryKey,JSON.stringify(snapshot));} catch {}
}
function updateUndoButtons() {
  if(!editor) return;
  if($("undo-code")) $("undo-code").disabled=!!state?.session?.paused_at || undoDepth(editor.state)===0;
  if($("redo-code")) $("redo-code").disabled=!!state?.session?.paused_at || redoDepth(editor.state)===0;
}
function clearEditor() {if(editor) {rememberEditor();editor.destroy();} editor=null;editorHistoryKey="";clearTimeout(saveTimer);}
function syncClock(s) {syncAt=performance.now(); syncedRemaining=s.remaining;}
function renderState() {
  stateGeneration++;
  document.body.classList.remove("task-paused");
  clearEditor(); result=null; resultsByPurpose={}; activeRun=null;
  currentKey=`${state.stage}:${state.task?.id||state.task_index}`; locale=state.locale||locale;
  document.documentElement.lang=locale==="zh"?"zh-Hans":"en";
  if(["practice","task"].includes(state.stage)) return renderWorkspace();
  app.innerHTML=topbar()+(state.stage==="ready"||state.stage==="complete"?steps():"")+`<main class="welcome"><div class="card" id="step-content"></div></main>`;
  const box=$("step-content");
  if(state.stage==="consent") {
    box.innerHTML=`${studyIntroduction()}<p>${tr("During timed tasks, the clock continues while you run code. Click Pause or close the page to take a break. Your remaining time is preserved.","计时任务中，运行代码时计时仍会继续。您可以点击“暂停”或关闭页面休息，剩余时间会保留。")}</p><p class="muted">${tr("Task time allowance","任务总时限")}: ${Math.round(Object.values(state.budgets).reduce((a,b)=>a+b,0)/60)} ${tr("minutes, plus tutorial and practice.","分钟，另加教程和练习时间。")}</p><label class="check-label"><input id="consent" type="checkbox"><span>${tr("I have read the information above and agree to participate.","我已阅读上述说明，并同意参与研究。")}</span></label><button class="primary" id="consent-next" disabled>${tr("Begin session →","开始研究 →")}</button>`;
    listen("consent",()=>$("consent-next").disabled=!$("consent").checked,"change");
    listen("consent-next",()=>transition("consent",{agree:true}));
  } else if(state.stage==="background") {
    box.innerHTML=`<div class="eyebrow">${tr("BEFORE WE BEGIN","开始之前")}</div><h1>${tr("Your programming background","您的编程经验")}</h1><p class="muted">${tr("These answers help us interpret the results.","这些回答有助于我们理解研究结果。")}</p><form id="background-form"><div class="form-grid">${[["python","Python","Python"],["rust","Rust","Rust"],["arrays","Array / numerical programming","数组 / 数值编程"],["zk","Zero-knowledge programming","零知识证明编程"]].map(([key,en,zh])=>`<label class="field">${tr(en,zh)}<select name="${key}" required><option value="">${tr("Choose your experience","请选择经验水平")}</option>${[["none","None yet","尚无经验"],["basic","Basic familiarity","了解基础"],["regular","Use regularly","经常使用"],["expert","Advanced experience","熟练使用"]].map(([v,e,z])=>`<option value="${v}">${tr(e,z)}</option>`).join("")}</select></label>`).join("")}</div><button class="primary" type="submit">${tr("Continue to tutorial →","进入教程 →")}</button></form>`;
    const form=$("background-form"), backgroundKey=`background:${token}`;
    try {const draft=JSON.parse(localStorage.getItem(backgroundKey)||"{}"); for(const field of form.elements) if(field.name && draft[field.name]) field.value=draft[field.name];} catch {}
    form.addEventListener("change",()=>{try {localStorage.setItem(backgroundKey,JSON.stringify(Object.fromEntries(new FormData(form))));} catch {}});
    form.onsubmit=runSafely(async e=>{e.preventDefault(); await transition("background",{answers:Object.fromEntries(new FormData(e.target))}); try {localStorage.removeItem(backgroundKey);} catch {}});
  } else if(state.stage==="tutorial") {
    box.innerHTML=`<div class="eyebrow">${tr("A QUICK ORIENTATION","快速入门")}</div><h1>${tr("Meet your workspace","熟悉编程工作台")}</h1><p class="lead">${tr("Write a function, run the examples, inspect the result. The project handles inputs and execution for you.","编写函数、运行示例并检查结果。项目已为您处理输入和执行流程。")}</p>${state.practice_total===2?`<p>${tr(`Orientation ${state.practice_index+1} of 2. Complete both untimed practices before the six tasks.`,`第 ${state.practice_index+1} / 2 次入门练习。完成两种环境的不限时练习后开始六道正式任务。`)}</p>`:""}${tutorial()}<div class="button-row"><button id="practice-start" class="primary">${tr("Open untimed practice →","进入不限时练习 →")}</button></div>`;
    listen("practice-start",()=>transition("tutorial_done"));
  } else if(state.stage==="ready") {
    const next=state.next_task;
    box.innerHTML=`<div class="eyebrow">${tr("NEXT TASK","下一个任务")}</div><h1>${tr("Task","任务")} ${state.task_index+1}: ${esc(title(next))}</h1>${state.design==="crossover-12x6-v1"?`<p>${tr("Environment for this task","本题编程环境")}: <strong>${conditionName(state.condition)}</strong></p>`:""}<p class="lead">${tr("You have","您有")} ${Math.round(state.budgets[next.id]/60)} ${tr("minutes for this task. The description and editor appear when you start the timer.","分钟完成本任务。点击开始后，将显示任务说明和编辑器，并开始计时。")}</p><p>${tr("Take a break now if you need one. You can use any correct implementation. Example tests provide feedback; additional test cases are used to assess your submitted solution.","如需休息，请现在休息。您可以使用任何正确实现。示例测试提供即时反馈；提交后的代码还会通过额外测试进行评估。")}</p><button class="primary" id="task-start">${tr("Start timer & open task →","开始计时并打开任务 →")}</button>`;
    listen("task-start",()=>transition("start_task"));
  } else {
    const complete=state.stage==="complete";
    box.innerHTML=`<div class="success-mark">${complete?"✓":"–"}</div><h1>${complete?tr("Thank you. You're all done.","感谢参与，您已完成研究。"):tr("Your session has ended.","本次研究已结束。")}</h1><p class="lead">${complete?tr("Your six submissions and any task ratings have been saved. You can close this page.","您的六次提交和任务评分已保存。现在可以关闭此页面。"):tr("Your participation has stopped. Contact the researcher if you have questions about your saved data.","您的参与已停止。如对已保存的数据有疑问，请联系研究人员。")}</p><p class="muted">${tr("Participant","参与者")}: ${esc(state.label)}</p>`;
  }
  bindLanguage();
}
async function presence(action) {
  return participant("presence",{action,page_id:pageId,task:state?.task?.id});
}
async function changePause(action) {
  if(pendingTransition) return;
  if(action==="pause") await flushSave();
  state=await presence(action); syncClock(state); renderPause();
}
async function transition(action,data={}) {
  if(pendingTransition) return;
  pendingTransition=true;stateGeneration++;
  try {state=await participant("action",{action,...data}); if(action==="start_task") state=await presence("open"); renderState();}
  finally {pendingTransition=false;}
}
function tutorial() {
  const py=state.condition==="zinnia";
  const sample=py?'def kernel(x):\n    # Example: double every value.\n    return x * 2':'fn kernel(x: &Array1<i64>) -> Array1<i64> {\n    // Example: double every value.\n    x.mapv(|value| value * 2)\n}';
  return `<div class="tutorial-grid"><div class="full"><h3>01 / ${tr("Your function interface","函数接口")}</h3><p>${py?tr("Write a function named kernel. You can add helper functions and NumPy imports; NumPy is also available as np. Input arrays contain signed integers. The project provides shape annotations for compilation; decorators are not required.","编写名为 kernel 的函数。可以添加辅助函数和 NumPy 导入；NumPy 也已以 np 提供。输入数组包含有符号整数。项目会提供编译所需的形状注解，无需添加装饰器。"):tr("Complete the supplied kernel function. Inputs are borrowed ndarray arrays of i64; return an owned array of the requested dimension. Array1–Array4, Axis, s!, and array! are already imported. Input loading and the execution entry point are provided.","完成提供的 kernel 函数。输入为借用的 ndarray i64 数组，返回指定维度的自有数组。Array1–Array4、Axis、s! 和 array! 已导入，输入加载和执行入口已提供。")}</p><pre>${esc(sample)}</pre></div><div><h3>02 / ${tr("Run & inspect","运行与检查")}</h3><p>${tr("Run examples compiles your code and executes the visible cases. Open each case to compare input, expected output, and actual output. Compiler errors appear in Console.","“运行示例”会编译代码并执行可见测试。展开各测试可比较输入、预期输出和实际输出。编译错误会显示在控制台。")}</p></div><div><h3>03 / ${tr("Try your own input","尝试自定义输入")}</h3><p>${tr("Use Custom input to enter a JSON object using the function's argument names. Run it to inspect another case. Ctrl/⌘ + Enter runs the examples from the editor.","在“自定义输入”中输入 JSON 对象，键名与函数参数一致。运行后可检查该测试。编辑器中 Ctrl/⌘ + Enter 可运行示例。")}</p></div><div class="full"><h3>04 / ${tr("Save & submit","保存与提交")}</h3><p>${tr("Drafts save automatically. Reloading resumes the same task and does not restart the timer. When ready, submit your current code and give two short ratings. When time ends, the latest saved draft is submitted automatically. Use Pause and Resume in the workspace to take a break. Closing the page also pauses the timer.","草稿会自动保存。重新加载会恢复当前任务，不会重置计时。完成后提交当前代码并填写两项简短评分。时间结束时，系统会自动提交最近保存的草稿。您可以使用工作台中的“暂停”和“继续”按钮休息。关闭页面也会暂停计时。")}</p></div></div>`;
}

function renderWorkspace() {
  const task=state.task, practice=state.stage==="practice";
  revision=state.session.revision; savedCode=state.session.code;
  let code=savedCode;
  try {
    const cached=JSON.parse(localStorage.getItem(`draft:${token}:${task.id}`)||"null");
    if(cached&&cached.code!==savedCode&&cached.revision===revision) {code=cached.code; notify(tr("Recovered your unsaved local draft.","已恢复尚未保存的本地草稿。"));}
    else if(cached&&cached.code!==savedCode) notify(tr("Another tab has a newer saved version. Your previous local draft is available in browser storage.","另一个标签页已保存了新版本。先前的本地草稿仍保留在浏览器存储中。"));
  } catch {}
  syncClock(state);
  const latest=state.runs?.find(r=>r.status==="complete"); result=latest?.result||null;
  for(const purpose of ["examples","custom"]) resultsByPurpose[purpose]=state.runs?.find(r=>r.purpose===purpose&&r.status==="complete")?.result||null;
  activeRun=state.runs?.find(r=>["queued","running"].includes(r.status))?.id||null;
  app.innerHTML=topbar(`${!practice?`<button id="participant-pause" class="quiet">${tr("Pause","暂停")}</button>`:""}<button id="reference" class="quiet">${tr("Manual","手册")}</button><div class="timerbox"><div class="timer-label">${practice?tr("NO TIME LIMIT","不限时"):tr("TIME REMAINING","剩余时间")}</div><div id="timer" class="timer mono">${practice?"—":fmt(state.remaining)}</div></div>`)+steps()+`<div class="mobile-warning">${tr("For comfortable editing, use a laptop or a wider screen.","为了方便编辑，建议使用笔记本电脑或更宽的屏幕。")}</div><main class="workspace"><aside class="description"><div class="eyebrow">${practice?tr("WARM-UP · NOT SCORED","热身练习 · 不计分"):tr(`TASK ${state.task_index+1} OF 6`,`任务 ${state.task_index+1} / 6`)}</div><h1>${esc(title(task))}</h1><div class="prose">${descriptionHtml(task)}</div><div class="manual-topics" aria-label="${tr("Related manual sections","相关操作手册")}">${[...new Set(["shape","allocate","transpose","slice_assign",...(task.manual_topics||[])])].map(id=>`<button class="small" data-manual-topic="${id}">${esc(manual.find(entry=>entry.id===id)?.title[locale==="zh"?1:0]||id)}</button>`).join("")}</div><div class="note">${tr("Your function should work for other valid input sizes and values as well as the examples shown here.","除了这里展示的示例，您的函数也需要能够处理其他符合要求的输入大小和数值。")}</div><div class="subtle"><h3>${tr("Input details","输入说明")}</h3><p class="small muted">${task.input_details ? esc(task.input_details[locale==="zh"?1:0]) : `${tr("Each array dimension has at least one element. Input values are integers in","各维度至少包含一个元素。输入值为整数，范围为")} ${task.id==="smooth"?"[0, 4096]":"[-512, 4096]"}. ${tr("The function receives its inputs as arguments; return the result from kernel. Calculations stay within the signed 64-bit integer range.","函数参数中已提供输入，请从 kernel 返回结果。计算不会超出有符号 64 位整数的范围。")}`}</p><button id="reference-side" class="small">${tr("Search the manual","搜索操作手册")}</button></div></aside><section class="editor-column"><div class="panel"><div class="panel-heading"><span class="file-label">${programmingBadge()}<span class="mono">solution.${state.condition==="zinnia"?"py":"rs"}</span></span><span id="save-status" class="save-indicator">${tr("Saved","已保存")}</span></div><div id="editor" aria-label="Code editor"></div><div class="editor-actions"><button class="primary" id="run-examples">▶ ${tr("Run examples","运行示例")}</button><span class="shortcut">Ctrl / ⌘ ↵</span><div class="spacer"></div><button id="undo-code" class="quiet" title="Ctrl+Z / Cmd+Z">${tr("Undo","撤销")}</button><button id="redo-code" class="quiet" title="Ctrl+Shift+Z / Cmd+Shift+Z / Ctrl+Y">${tr("Redo","重做")}</button><button id="download" class="quiet">${tr("Download","下载")}</button><button id="reset-code" class="quiet">${tr("Reset","重置")}</button></div></div><div class="panel"><div class="panel-heading"><span>${tr("Execution & feedback","执行与反馈")}</span><span id="run-status" class="run-status"></span></div><div class="tabs" role="tablist">${[["examples","Examples","示例"],["custom","Custom input","自定义输入"],["console","Console","控制台"]].map(([id,en,zh])=>`<button class="tab ${outputTab===id?"selected":""}" data-tab="${id}" role="tab">${tr(en,zh)}</button>`).join("")}</div><div id="results" class="results-body"></div></div><div class="button-row" style="margin-top:0;justify-content:space-between"><span class="small muted">${tr("Your work is saved under","本次代码保存在")} ${esc(state.label)}</span><button id="finish-task">${practice?tr("Finish practice →","完成练习 →"):tr("Submit task →","提交任务 →")}</button></div></section></main><footer class="study-footer"><span>${tr("Participant","参与者")}: ${esc(state.label)}</span><button class="quiet small" id="withdraw">${tr("Leave study","退出研究")}</button></footer><div id="pause-cover"></div>`;
  editorHistoryKey=`editor-history:${token}:${task.id}`;
  const extensions=[basicSetup,indentUnit.of("    "),EditorState.tabSize.of(4),state.condition==="zinnia"?python():rust(),EditorView.lineWrapping,
    Prec.highest(keymap.of([indentWithTab,{any:(view,event)=>{
      if((event.ctrlKey||event.metaKey) && !event.altKey && !event.isComposing) {
        const key=event.key.toLowerCase();
        if(key==="z" || key==="y") {(key==="y" || event.shiftKey ? redo : undo)(view);return true;}
      }
      return false;
    }}])),
    EditorView.updateListener.of(update=>{if(update.docChanged) {changed();rememberEditor();} updateUndoButtons();}),
    EditorView.domEventHandlers({keydown(e){if(e.key==="Enter"&&(e.ctrlKey||e.metaKey)){e.preventDefault();startRun("examples").catch(e=>notify(e.message));return true;}},paste(){event("editor_paste");}})
  ];
  let editorState;
  try {
    const snapshot=editorHistories.get(editorHistoryKey)||JSON.parse(sessionStorage.getItem(editorHistoryKey)||"null");
    if(snapshot?.doc===code) editorState=EditorState.fromJSON(snapshot,{extensions},{history:historyField});
  } catch {}
  editor=new EditorView({parent:$("editor"),state:editorState||EditorState.create({doc:code,extensions})});
  for(const [id,command] of [["undo-code",undo],["redo-code",redo]]) listen(id,()=>{if(!state.session.paused_at){command(editor);editor.focus();}});
  updateUndoButtons();
  listen("participant-pause",()=>changePause("pause"));
  bindLanguage(); listen("reference",openReference); listen("reference-side",openReference);
  listen("run-examples",()=>startRun("examples")); listen("download",()=>download(`solution.${state.condition==="zinnia"?"py":"rs"}`,editor.state.doc.toString(),"text/plain"));
  listen("reset-code",()=>{if(confirm(tr("Replace this draft with the starter? Saved revisions remain available to the researcher.","将当前草稿替换为初始代码？研究人员仍可查看已保存的历史版本。"))) editor.dispatch({changes:{from:0,to:editor.state.doc.length,insert:task.starter}});});
  listen("finish-task",async()=>{await flushSave(); if(practice) await transition("practice_done"); else await beginSubmission();});
  listen("withdraw",async()=>{if(confirm(tr("End your participation now?","现在结束参与研究吗？"))){await flushSave(); await transition("withdraw");}});
  document.querySelectorAll("[data-tab]").forEach(b=>b.onclick=()=>{outputTab=b.dataset.tab; renderResults();});
  document.querySelectorAll("[data-manual-topic]").forEach(button=>button.onclick=()=>openReference(button.dataset.manualTopic));
  const hint=document.querySelector(".task-hint"); if(hint) hint.addEventListener("toggle",()=>{if(hint.open) event("hint_open");});
  renderResults(); renderPause(); if(activeRun) pollRun(activeRun);
  if(code!==savedCode&&!state.session.paused_at) changed();
  if(state.session.pause_reason==="submission") submissionDialog();
}
function changed() {
  if(!editor) return;
  const code=editor.state.doc.toString();
  try {localStorage.setItem(`draft:${token}:${state.task.id}`,JSON.stringify({code,revision}));} catch {}
  setSaved(tr("Unsaved changes","有未保存的更改")); clearTimeout(saveTimer);
  saveTimer=setTimeout(()=>flushSave().catch(e=>setSaved(e.message,true)),750);
}
function setSaved(message,error=false) {if($("save-status")){ $("save-status").textContent=message; $("save-status").classList.toggle("error",error); }}
function flushSave() {
  clearTimeout(saveTimer);
  saving=saving.catch(()=>{}).then(async()=>{
    if(!editor||!state?.session) return;
    const code=editor.state.doc.toString();
    if(code===savedCode) return;
    setSaved(tr("Saving…","正在保存…"));
    const task=state.task.id, view=editor;
    const response=await participant("save",{code,revision,task});
    if(editor!==view||state.task?.id!==task) return;
    revision=response.revision; savedCode=code; state.session.revision=revision; state.session.code=code;
    if(editor&&editor.state.doc.toString()===code) {localStorage.removeItem(`draft:${token}:${state.task.id}`); setSaved(tr("Saved","已保存"));}
    else if(editor) changed();
  });
  return saving;
}
async function startRun(purpose) {
  if(activeRun||pendingTransition||!editor) return;
  let input;
  if(purpose==="custom") {input=JSON.parse($("custom-input").value); sessionStorage.setItem(`input:${token}:${state.task.id}`,JSON.stringify(input));}
  await flushSave();
  const code=editor.state.doc.toString(), task=state.task.id;
  const response=await participant("run",{purpose,code,revision,task,...(input?{input}:{})});
  if(state.task?.id!==task) return;
  revision=response.revision; savedCode=code; activeRun=response.run_id; result=null; resultsByPurpose[purpose]=null; outputTab=purpose; renderResults();
  pollRun(activeRun);
}
async function pollRun(ident) {
  try {
    const run=await participant(`runs/${ident}`);
    if(activeRun!==ident) return;
    if(["queued","running"].includes(run.status)) {renderRunStatus(run.status); setTimeout(()=>pollRun(ident),800);return;}
    activeRun=null;result=run.result||{status:"infrastructure_error",message:tr("Execution was interrupted. Please run again.","执行已中断，请重新运行。")};
    resultsByPurpose[run.purpose]=result;
    if(["compile_error","runtime_error","infrastructure_error","timeout"].includes(result.status)) outputTab="console";
    renderResults();
  } catch(e) {notify(e.message); if(activeRun===ident) setTimeout(()=>pollRun(ident),2500);}
}
function renderRunStatus(status) {
  if(!$("run-status")) return;
  $("run-status").innerHTML=activeRun?`<span class="spinner"></span>${status==="queued"?tr("Queued","排队中"):tr("Compiling & running…","正在编译和运行…")}`:`<span class="live-dot"></span>${result?`${esc(result.seconds??"")}s`:tr("Ready","就绪")}`;
  $("run-examples").disabled=!!activeRun||!!state.session.paused_at;
  $("finish-task").disabled=!!activeRun||!!state.session.paused_at;
  if($("run-custom")) $("run-custom").disabled=!!activeRun||!!state.session.paused_at;
}
function renderResults(latestResult=result) {
  if(!$("results")) return;
  const inputDraft=$("custom-input")?.value;
  document.querySelectorAll("[data-tab]").forEach(b=>b.classList.toggle("selected",b.dataset.tab===outputTab));
  const result=outputTab==="console"?latestResult:resultsByPurpose[outputTab];
  const status=result?.status;
  const banner=result?`<div class="banner ${status==="passed"?"":"error"}">${status==="passed"?tr("All cases in this run passed.","本次运行的所有测试均通过。"):status==="failed"?tr("Some outputs differ. Open a case to inspect the result.","部分输出与预期不同，请展开测试查看结果。"):esc(result.message||status)}${status==="passed"?` <span class="muted">${tr("Results describe the code at run time. Run again after editing; examples do not guarantee full correctness.","结果对应运行时的代码。编辑后请重新运行；示例不能保证所有输入下均正确。")}</span>`:""}</div>`:"";
  if(outputTab==="examples") {
    const rows=result?.cases||state.task.examples.map((c,i)=>({...c,index:i+1,passed:null}));
    $("results").innerHTML=banner+rows.map(c=>`<details class="case"><summary><span class="status-symbol ${c.passed===true?"pass":c.passed===false?"fail":""}">${c.passed===true?"✓":c.passed===false?"×":"·"}</span>${tr("Case","测试")} ${c.index}<span class="spacer"></span><span class="case-state ${c.passed===false?"fail":""}">${c.passed===true?tr("Passed","通过"):c.passed===false?tr("Different output","输出不同"):tr("Not run","尚未运行")}</span></summary><div class="case-content"><div class="full"><span class="small muted">${tr("Input","输入")}</span><pre>${esc(pretty(c.input))}</pre></div><div><span class="small muted">${tr("Expected","预期输出")}</span><pre>${esc(pretty(c.expected))}</pre></div><div><span class="small muted">${tr("Your output","您的输出")}</span><pre>${c.actual!==undefined?esc(pretty(c.actual)):"—"}</pre></div></div></details>`).join("");
  } else if(outputTab==="custom") {
    const value=inputDraft||sessionStorage.getItem(`input:${token}:${state.task.id}`)||pretty(state.task.examples[0].input);
    $("results").innerHTML=`<p class="small muted">${tr("Enter a JSON object using the function's argument names.","输入 JSON 对象，键名应与函数参数一致。")}</p><textarea id="custom-input" class="custom-input" aria-label="Custom JSON input" spellcheck="false">${esc(value)}</textarea><div class="button-row"><button id="run-custom">▶ ${tr("Run custom input","运行自定义输入")}</button></div>${result?`<div style="margin-top:16px">${banner}${result.cases?`<pre>${esc(pretty(result.cases))}</pre>`:""}</div>`:""}`;
    listen("run-custom",()=>startRun("custom"));
  } else {
    const text=[result?.stdout,result?.stderr].filter(Boolean).join("\n");
    $("results").innerHTML=banner+(text?`<pre class="console">${esc(text)}</pre>`:`<p class="output-empty">${state.condition==="zinnia"?tr("Compiler diagnostics appear here. Printed output is not available; inspect the actual outputs in the test cases.","此处显示编译诊断。当前不提供打印输出，请在测试用例中查看实际输出。"):tr("Compiler diagnostics and printed output appear here. You can print intermediate values while debugging.","此处会显示编译诊断和打印输出。调试时可以打印中间值。")}</p>`);
  }
  renderRunStatus(activeRun?"running":null);
}
async function beginSubmission() {
  if(pendingTransition) return;
  pendingTransition=true;stateGeneration++;
  try {
    await flushSave();
    state=await participant("review",{code:editor.state.doc.toString(),revision,task:state.task.id,page_id:pageId});
    revision=state.session.revision;savedCode=state.session.code;
    syncClock(state);renderPause();submissionDialog();
  } finally {pendingTransition=false;}
}
function submissionDialog() {
  dialog.dataset.kind="submission";
  const ratings=[
    ["difficulty",tr("How difficult was this task?","这个任务有多难？"),[
      ["Extremely easy","极其容易"],["Very easy","很容易"],["Somewhat easy","比较容易"],["Neither easy nor difficult","难度适中"],["Somewhat difficult","比较困难"],["Very difficult","很困难"],["Extremely difficult","极其困难"]]],
    ["confidence",tr("How confident are you in your solution?","您对答案的信心如何？"),[
      ["Not at all confident","完全没有信心"],["Very slightly confident","只有一点信心"],["Slightly confident","有些信心"],["Moderately confident","有一定信心"],["Quite confident","比较有信心"],["Very confident","很有信心"],["Completely confident","完全有信心"]]]
  ];
  dialog.innerHTML=`<div class="dialog-top"><h2>${tr("Submit this task","提交本任务")}</h2>${languageButtons()}</div><p class="muted">${tr("The timer is paused while you answer. Submit your saved code, or choose Keep working to resume editing and timing.","填写评价期间计时已暂停。您可以提交已保存的代码，或选择“继续编辑”以恢复编辑和计时。")}</p><form id="submit-form">${ratings.map(([id,title,options])=>`<fieldset class="rating-field"><legend>${title}</legend><div class="rating-options">${options.map(([en,zh],index)=>`<label class="rating-choice"><input type="radio" name="${id}" value="${index+1}" required><span>${tr(en,zh)}</span></label>`).join("")}</div></fieldset>`).join("")}<label class="field">${tr("Anything that helped or got in the way? (optional)","有什么帮助或阻碍了您？（选填）")}<textarea name="comment" maxlength="1000"></textarea></label><div class="button-row"><button type="submit" class="primary">${tr("Submit & continue","提交并继续")}</button><button type="button" id="cancel-submit">${tr("Keep working","继续编辑")}</button></div></form>`;
  const form=$("submit-form"), key=`rating:${token}:${state.task.id}`;
  try {const draft=JSON.parse(localStorage.getItem(key)||"{}");for(const field of form.elements) {if(field.type==="radio") field.checked=draft[field.name]===field.value;else if(field.name && draft[field.name]) field.value=draft[field.name];}} catch {}
  form.addEventListener("input",()=>{try {localStorage.setItem(key,JSON.stringify(Object.fromEntries(new FormData(form))));} catch {}});
  const cancel=async()=>{
    if(pendingTransition) return;
    pendingTransition=true;stateGeneration++;
    try {state=await presence("resume");syncClock(state);dialog.close();renderPause();}
    finally {pendingTransition=false;}
  };
  dialog.oncancel=runSafely(async e=>{e.preventDefault();await cancel();});
  dialog.addEventListener("close",()=>{dialog.oncancel=null;},{once:true});
  dialog.showModal();listen("cancel-submit",cancel);bindLanguage();
  form.onsubmit=runSafely(async e=>{
    e.preventDefault();if(pendingTransition) return;pendingTransition=true;stateGeneration++;
    try {const data=Object.fromEntries(new FormData(form));state=await participant("submit",{code:savedCode,revision,task:state.task.id,...data,difficulty:Number(data.difficulty),confidence:Number(data.confidence)});try{localStorage.removeItem(key);}catch{}dialog.close();renderState();}
    finally {pendingTransition=false;}
  });
}
function openReference(topic="") {
  if(state?.session?.paused_at) return;
  dialog.dataset.kind="manual";
  dialog.oncancel=null;
  dialog.innerHTML=`<div class="dialog-top"><h2>${tr("Programming manual","编程操作手册")}</h2>${languageButtons()}<button id="close-reference" aria-label="${tr("Close manual","关闭手册")}">×</button></div><div class="reference-content"><p class="muted">${tr("Search operations, syntax, or keywords. Examples use small arrays; adapt their shapes to your task. The task timer continues while the manual is open.","可搜索操作、语法或关键词。示例使用小数组，请根据任务调整形状。查看手册时任务仍会继续计时。")}</p><label class="field">${tr("Search the manual","搜索操作手册")}<input id="manual-search" type="search" placeholder="${tr("e.g. reduction, axis, slicing, sum","例如：归约、轴、切片、求和")}"></label><nav class="manual-topics" aria-label="${tr("Manual topics","手册主题")}"><button data-topic="">${tr("All operations","全部操作")}</button>${manual.map(entry=>`<button data-topic="${entry.id}">${esc(entry.title[locale==="zh"?1:0])}</button>`).join("")}</nav><div id="manual-results" aria-live="polite"></div></div>`;
  const search=$("manual-search");
  const render=()=>{
    const words=search.value.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
    const entries=manual.filter(entry=>words.every(word=>[...entry.title,...entry.description[state.condition],entry.code[state.condition]].join(" ").toLocaleLowerCase().includes(word)));
    $("manual-results").innerHTML=entries.map(entry=>`<section class="manual-entry"><h3>${esc(entry.title[locale==="zh"?1:0])}</h3><p>${esc(entry.description[state.condition][locale==="zh"?1:0])}</p><pre>${esc(entry.code[state.condition])}</pre><p class="small">${tr("Expected out","out 的预期结果")}: <code>${esc(JSON.stringify(entry.expected))}</code></p></section>`).join("")||`<p>${tr("No matching operations. Try another keyword or choose a topic above.","未找到匹配操作。请尝试其他关键词，或选择上方主题。")}</p>`;
  };
  const select=id=>{search.value=manual.find(entry=>entry.id===id)?.title[locale==="zh"?1:0]||"";render();};
  search.addEventListener("input",render);
  document.querySelectorAll("[data-topic]").forEach(button=>button.onclick=()=>select(button.dataset.topic));
  select(typeof topic==="string"?topic:"");
  event("reference_open"); dialog.showModal(); listen("close-reference",()=>dialog.close());bindLanguage();search.focus();
  dialog.addEventListener("close",()=>event("reference_close"),{once:true});
}
function renderPause() {
  const cover=$("pause-cover"); if(!cover) return;
  const paused=!!state.session.paused_at;
  document.body.classList.toggle("task-paused",paused);
  document.querySelectorAll(".workspace,.progressbar,.study-footer").forEach(element=>{element.inert=paused;});
  if($("reference")) $("reference").disabled=paused;
  const researcher=state.session.pause_reason==="researcher";
  cover.innerHTML=state.session.paused_at?`<div class="pause-cover"><div><h2>${tr("Task paused","任务已暂停")}</h2><p>${researcher?tr("The researcher has paused this task. Your remaining time is preserved.","研究人员已暂停任务，剩余时间已保留。"):tr("Your remaining time and saved code are preserved. Resume when you are ready.","剩余时间和已保存的代码均已保留。准备好后可继续任务。")}</p>${researcher?"":`<button id="participant-resume" class="primary">${state.session.pause_reason==="submission"?tr("Complete submission","完成提交"):tr("Resume","继续")}</button>`}</div></div>`:"";
  listen("participant-resume",()=>state.session.pause_reason==="submission"?submissionDialog():changePause("resume"));
  if($("participant-pause")) $("participant-pause").disabled=!!state.session.paused_at;
  if(state.session.paused_at) {if(state.session.pause_reason!=="submission" || dialog.dataset.kind!=="submission") dialog.close(); editor?.contentDOM.setAttribute("contenteditable","false");}
  else editor?.contentDOM.setAttribute("contenteditable","true");
  updateUndoButtons();
  renderRunStatus(activeRun?"running":null);
}
async function refreshState() {
  if(!token||!state||refreshBusy||pendingTransition||pageClosed) return;
  const generation=stateGeneration;
  refreshBusy=true;
  try {
    const next=state.stage==="task"?await presence("heartbeat"):await participant("state");
    if(generation!==stateGeneration||pendingTransition) return;
    const key=`${next.stage}:${next.task?.id||next.task_index}`;
    if(key!==currentKey) {state=next;dialog.close();renderState();}
    else {const wasPaused=state.session?.paused_at;state=next;syncClock(next);if(wasPaused!==next.session?.paused_at){renderPause();if(wasPaused&&!next.session?.paused_at&&editor)flushSave().catch(e=>setSaved(e.message,true));}}
  } catch(e) {setSaved(tr("Connection interrupted · draft kept locally","连接中断 · 草稿已保存在本地"),true);}
  finally {refreshBusy=false;}
}
function tick() {
  if(!state?.session||!$("timer")) return;
  if(syncedRemaining===null) return;
  const remaining=Math.max(0,syncedRemaining-(state.session.paused_at?0:(performance.now()-syncAt)/1000));
  $("timer").textContent=fmt(remaining);$("timer").classList.toggle("warn",remaining<=60);$("timer").classList.toggle("expired",remaining<=0);
  if(remaining<=0&&!pendingTransition) refreshState();
}
function download(name,data,mime="application/json") {
  const url=URL.createObjectURL(new Blob([data],{type:mime}));const link=document.createElement("a");link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

async function renderAdmin() {
  clearEditor(); state=null; clearInterval(dashboardTimer);
  if(!sessionStorage.getItem("study-admin")) {
    app.innerHTML=topbar()+`<main class="welcome"><div class="card"><div class="eyebrow">${tr("RESEARCHER ACCESS","研究人员入口")}</div><h1>${tr("Study monitor","研究监控台")}</h1><p>${tr("Use the private researcher URL printed by the server, or enter its admin key below.","请使用服务器输出的私有研究人员链接，或在下方输入管理员密钥。")}</p><form id="login-form"><label class="field">${tr("Admin key","管理员密钥")}<input id="admin-key" type="password" autocomplete="off" required></label><button class="primary">${tr("Open dashboard","打开监控台")}</button></form></div></main>`;
    $("login-form").onsubmit=runSafely(async e=>{e.preventDefault();sessionStorage.setItem("study-admin",$("admin-key").value);try{await api("/api/admin",undefined,true);await renderAdmin();}catch(error){sessionStorage.removeItem("study-admin");throw error;}});bindLanguage();return;
  }
  const data=await api("/api/admin",undefined,true);
  app.innerHTML=topbar(`<button id="logout" class="quiet">${tr("Sign out","退出登录")}</button>`)+`<main class="research"><div class="research-heading"><div><div class="eyebrow">${tr("CROSSOVER · 12 PARTICIPANTS · SIX TASKS","交叉设计 · 12 人 · 六题")}</div><h1>${tr("Session monitor","研究进度监控")}</h1><p class="muted">${tr("Each participant uses both environments; per-task assignments stay fixed. Updates arrive every four seconds.","每人使用两种环境，各题分配保持固定。页面每四秒自动更新。")}</p></div><div class="button-row"><select id="export-cohort" aria-label="Export cohort"><option value="">${tr("All research records","全部正式记录")}</option>${[...new Set(data.participants.filter(p=>!p.preview&&p.cohort_id).map(p=>p.cohort_id))].map(id=>`<option value="${esc(id)}">${esc(id)}</option>`).join("")}</select><button id="export-json">${tr("Export study data","导出研究数据")}</button><button id="export-links">${tr("Export participant links","导出参与者链接")}</button></div></div><div id="readiness"></div><div id="stats" class="research-stats"></div><section class="panel"><div class="assignment-controls"><label class="field">${tr("Design","设计")}<select id="assignment-design"><option value="crossover-12x6-v1">${tr("12 × 6 crossover","12 人 × 6 题交叉设计")}</option><option value="between_subjects">${tr("Legacy single-environment","旧版单环境")}</option></select></label><label class="field">${tr("Participants","参与者数量")}<input id="participant-count" type="number" min="1" max="200" value="12"></label><label class="field">${tr("Task order","任务顺序")}<select id="assignment-order"><option value="balanced">${tr("Counterbalanced between groups","在两组间平衡顺序")}</option><option value="progressive">${tr("Easy → hard for everyone","所有人由易到难")}</option></select></label><button class="primary" id="create-participants">＋ ${tr("Create personal links","创建专属链接")}</button><button id="edit-budgets">${tr("Time allowances","任务时限")}</button><div style="flex:1"></div><button id="preview-crossover">${tr("Preview crossover","预览交叉流程")}</button><button id="preview-zinnia">${tr("Preview Zinnia","预览 Zinnia")}</button><button id="preview-rust">${tr("Preview Rust","预览 Rust")}</button></div><div class="table-scroll"><table><thead><tr>${[tr("Participant","参与者"),tr("Assignments / progress","分配 / 进度"),tr("Activity","活动"),tr("Final results","最终结果"),tr("Personal link","专属链接"),tr("Inspect","查看")].map(x=>`<th>${x}</th>`).join("")}</tr></thead><tbody id="participant-rows"></tbody></table></div></section><p class="small muted" style="margin-top:15px">${tr("Preview participants are excluded from the research export. Personal links are private access credentials: share only with the intended participant.","预览参与者不会包含在研究数据导出中。专属链接具有访问权限，请仅发送给对应参与者。")}</p></main>`;
  let budgets={...data.budgets};
  bindLanguage();listen("logout",()=>{sessionStorage.removeItem("study-admin");renderAdmin();});
  listen("create-participants",async()=>{const value=await api("/api/admin/create",{count:Number($("participant-count").value),order:$("assignment-order").value,budgets,design:$("assignment-design").value},true);notify(tr(`Created ${value.participants.length} personal links.`,`已创建 ${value.participants.length} 个专属链接。`));await updateAdmin();if(value.participants[0]?.cohort_id){$("export-cohort").value=value.participants[0].cohort_id;await updateAdmin();}});
  listen("export-cohort",()=>updateAdmin(),"change");
  const syncDesign=()=>{const cross=$("assignment-design").value==="crossover-12x6-v1";$("participant-count").disabled=cross;$("assignment-order").disabled=cross;if(cross)$("participant-count").value=12;};
  listen("assignment-design",syncDesign,"change");syncDesign();
  listen("preview-crossover",async()=>{const value=await api("/api/admin/create",{design:"crossover-12x6-v1",count:1,preview:true,budgets},true);window.open(`${basePath}/p/${value.participants[0].token}`,"_blank","noopener");await updateAdmin();});
  for(const c of ["zinnia","rust"]) listen(`preview-${c}`,async()=>{const value=await api("/api/admin/create",{count:1,preview:true,condition:c,order:"progressive",budgets},true);window.open(`${basePath}/p/${value.participants[0].token}`,"_blank","noopener");await updateAdmin();});
  listen("edit-budgets",()=>{dialog.innerHTML=`<h2>${tr("Task time allowances","任务时限")}</h2><p>${tr("Applies to new links only. Existing assignments keep their original budgets. Defaults need confirmation in a human pilot.","仅对新链接生效。已分配的参与者保持原时限。默认时限仍需通过真人试测确认。")}</p><form id="budget-form">${Object.entries(budgets).map(([key,value])=>`<label class="field">${esc(taskName(key))} · ${tr("minutes","分钟")}<input name="${key}" type="number" min="1" max="120" value="${value/60}" required></label>`).join("")}<button class="primary">${tr("Use these allowances","使用这些时限")}</button></form>`;dialog.showModal();$("budget-form").onsubmit=runSafely(e=>{e.preventDefault();budgets=Object.fromEntries([...new FormData(e.target)].map(([k,v])=>[k,Math.round(Number(v)*60)]));dialog.close();});});
  listen("export-json",async()=>download("study-export.json",pretty(await api("/api/admin/export"+($("export-cohort").value?"?cohort="+encodeURIComponent($("export-cohort").value):""),undefined,true))));
  listen("export-links",async()=>{const response=await fetch(basePath+"/api/admin/assignments.csv"+($("export-cohort").value?"?cohort="+encodeURIComponent($("export-cohort").value):""),{headers:{Authorization:`Bearer ${sessionStorage.getItem("study-admin")}`}});if(!response.ok)throw new Error("Export failed");download("participant-links.csv",await response.text(),"text/csv");});
  await updateAdmin(data); dashboardTimer=setInterval(()=>updateAdmin().catch(e=>notify(e.message)),4000);
}
async function updateAdmin(data) {
  if(adminDetail) return;
  data=data||await api("/api/admin",undefined,true);
  if(!$("participant-rows")) return;
  const selected=$("export-cohort").value;
  const cohorts=[...new Set(data.participants.filter(p=>!p.preview&&p.cohort_id).map(p=>p.cohort_id))];
  $("export-cohort").innerHTML=`<option value="">${tr("All research records","全部正式记录")}</option>${cohorts.map(id=>`<option value="${esc(id)}">${esc(id)}</option>`).join("")}`;
  if(cohorts.includes(selected))$("export-cohort").value=selected;
  const visible=data.participants.filter(p=>!selected||p.cohort_id===selected);
  const real=visible.filter(p=>!p.preview),ready=data.readiness;
  $("readiness").innerHTML=`<div class="banner ${ready.zinnia&&ready.rust?"":"warning"}">${tr("Execution environments","执行环境")}: Zinnia ${ready.zinnia?"✓":"✕"} · Rust / RISC Zero ${ready.rust?"✓":"✕"} · ${tr("isolated sandbox","隔离沙箱")} ${ready.sandbox?"✓":"✕"}${ready.rust_mode!=="risc0"?` · ${tr("Native preview only — real study links disabled","仅为原生预览 — 已禁用真实研究链接")}`:""}${ready.sandbox_detail?`<br>${esc(ready.sandbox_detail)}`:""}</div>`;
  $("stats").innerHTML=[[real.length,tr("Assigned participants","已分配参与者")],[real.filter(p=>!["consent","complete","withdrawn"].includes(p.stage)).length,tr("In progress","进行中")],[real.filter(p=>p.stage==="complete").length,tr("Completed","已完成")],[real.filter(p=>p.design==="crossover-12x6-v1").length,tr("Crossover participants","交叉设计参与者")]].map(([n,label])=>`<div class="stat"><span>${label}</span><b>${n}</b></div>`).join("");
  $("participant-rows").innerHTML=visible.length?visible.map(p=>{
    const active=p.sessions.find(s=>!s.finished&&!s.task.startsWith("practice"));
    const grades=p.sessions.filter(s=>s.grade?.result);
    const passed=grades.filter(s=>s.grade.result.status==="passed").length;
    return `<tr><td><b>${esc(p.label)}</b>${p.preview?`<br><span class="badge gray">${tr("Preview","预览")}</span>`:""}<br><span class="muted small">${p.last_seen?tr(`${Math.max(0,Math.round(data.server_time-p.last_seen))}s ago`,`${Math.max(0,Math.round(data.server_time-p.last_seen))} 秒前`):tr("Not opened","尚未打开")}</span></td><td>${assignmentName(p)}<div class="task-progress">${p.task_order.map((_,i)=>`<i class="${i<p.task_index?"done":i===p.task_index&&p.stage==="task"?"active":""}"></i>`).join("")}</div><span class="muted">${esc(p.stage)}${active?` · ${fmt(active.remaining)}${active.paused_at?" ⏸":""}`:""}</span></td><td>${p.edits} ${tr("saves","次保存")}<br>${p.runs} ${tr("executions","次执行")}</td><td>${passed} / ${grades.length} ${tr("passed","通过")}<br><span class="muted">${tr("final submissions only","仅统计最终提交")}</span></td><td><div class="link-field"><input readonly aria-label="${esc(p.label)} participant link" value="${esc(location.origin+basePath+'/p/'+p.token)}"><button data-copy="${p.token}">${tr("Copy","复制")}</button></div></td><td><button data-inspect="${p.token}">${tr("Details","详情")}</button></td></tr>`;
  }).join(""):`<tr><td colspan="6" class="muted">${tr("Create a balanced cohort of 12 participants to begin.","创建 12 人的平衡交叉队列以开始。")}</td></tr>`;
  document.querySelectorAll("[data-copy]").forEach(b=>b.onclick=runSafely(async()=>{await navigator.clipboard.writeText(`${location.origin}${basePath}/p/${b.dataset.copy}`);notify(tr("Personal link copied","已复制专属链接"));}));
  document.querySelectorAll("[data-inspect]").forEach(b=>b.onclick=runSafely(()=>inspectParticipant(b.dataset.inspect)));
}
function hiddenScore(run) {
  if(!run) return tr("Not recorded","未记录");
  if(run.status==="deferred") return tr("After session ends","研究结束后评分");
  const r=run.result?JSON.parse(run.result):null;
  if(!r) return esc(run.status);
  if(r.cases?.length) return `${r.cases.filter(c=>c.passed).length} / ${r.cases.length}`;
  return esc(({compile_error:tr("Compile error","编译失败"),runtime_error:tr("Runtime error","运行错误"),infrastructure_error:tr("Evaluation unavailable","评分未完成"),timeout:tr("Execution timed out","执行超时")})[r.status]||r.status);
}
function attemptTrail(data,p) {
  const rows=[];
  for(const task of p.task_order) {
    const publicRuns=data.runs.filter(r=>r.task===task&&["examples","custom"].includes(r.purpose));
    const finals=data.runs.filter(r=>r.task===task&&r.purpose==="grade");
    for(const run of [...publicRuns,...finals.slice(-1)]) {
      const hidden=run.purpose==="grade"?run:[...data.runs].reverse().find(r=>r.purpose==="attempt_grade"&&r.parent_run_id===run.id);
      const elapsed=v=>v==null?"—":fmt(v);
      const result=run.result?JSON.parse(run.result):null;
      rows.push(`<tr><td>${esc(taskName(task))}<br><span class="small muted">${conditionName(run.condition||p.condition)}${run.period?` · ${tr("Position","位置")} ${run.period}`:""}</span></td><td>${run.purpose==="grade"?tr("Final submission","最终提交"):`#${run.attempt_index??"—"} · ${run.purpose==="custom"?tr("Custom","自定义"):tr("Examples","示例")}`}</td><td>${run.revision==null?"—":`r${run.revision}`}</td><td>${elapsed(run.active_elapsed)}</td><td>${elapsed(run.interval_active)}</td><td>${run.purpose!=="grade"&&run.started!=null&&run.finished!=null?(run.finished-run.started).toFixed(2)+"s":"—"}</td><td>${run.purpose==="grade"?"—":esc(result?.status||run.status)}</td><td>${hiddenScore(hidden)}</td><td><details><summary>${tr("Code","代码")}</summary><p class="small mono">${esc(run.source_sha256)}</p><pre>${esc(run.code)}</pre></details></td></tr>`);
    }
  }
  return `<section class="panel full"><h2>${tr("Attempt trail and hidden scores","完整尝试轨迹与隐藏评分")}</h2><p class="small muted">${tr("Every Run is frozen separately, including repeated code and compilation failures. Hidden evaluation runs after the session ends and is never shown to participants. Active time excludes pauses; intervals include reading, editing and execution waits. Old records without timing metadata show —.","每次运行独立冻结，包括重复代码与编译失败。研究结束后执行隐藏评分，参与者不可见。有效时间不含暂停；尝试间隔包含阅读、编辑和运行等待。没有时间元数据的旧记录显示 —。")}</p><div class="table-scroll"><table><thead><tr>${[tr("Task","任务"),tr("Attempt","尝试"),tr("Revision","版本"),tr("Active elapsed","累计有效时间"),tr("Interval","本次间隔"),tr("Execution","运行耗时"),tr("Visible feedback","公开反馈"),tr("Hidden score","隐藏评分"),tr("Frozen source","冻结代码")].map(v=>`<th>${v}</th>`).join("")}</tr></thead><tbody>${rows.join("")||`<tr><td colspan="9">${tr("No attempts yet","尚无尝试")}</td></tr>`}</tbody></table></div></section>`;
}
async function inspectParticipant(ident) {
  const previousRevision=adminDetail===ident?$("snapshot-select")?.value:null;
  adminDetail=ident;clearInterval(dashboardTimer);
  const data=await api(`/api/admin/participant/${ident}`,undefined,true),p=data.participants[0];
  if(!p) throw new Error("Participant not found");
  if(typeof p.task_order==="string")p.task_order=JSON.parse(p.task_order);
  if(typeof p.task_conditions==="string")p.task_conditions=JSON.parse(p.task_conditions);
  if(typeof p.practice_order==="string")p.practice_order=JSON.parse(p.practice_order);
  app.innerHTML=topbar()+`<main class="detail-page"><button id="back-monitor">← ${tr("Session monitor","研究监控台")}</button><div class="research-heading" style="margin-top:22px"><div><div class="eyebrow">${esc(p.label)} · ${assignmentName(p)}</div><h1>${tr("Progress & code history","进度与代码历史")}</h1></div><div class="button-row"><button id="refresh-detail">${tr("Refresh","刷新")}</button><button id="pause-participant">${tr("Pause current task","暂停当前任务")}</button><button id="resume-participant">${tr("Resume","恢复")}</button></div></div>${p.task_conditions?`<p class="small">${p.task_order.map((t,i)=>`${i+1}. ${esc(taskName(t))}: ${conditionName(typeof p.task_conditions==="string"?JSON.parse(p.task_conditions)[t]:p.task_conditions[t])}`).join(" · ")}</p>`:""}<div class="detail-grid">${attemptTrail(data,p)}<section class="panel full"><h2>${tr("Saved drafts","已保存的草稿")}</h2><select id="snapshot-select" aria-label="Saved revision">${[...data.snapshots].reverse().map(s=>`<option value="${s.id}">${esc(taskName(s.task))} · r${s.revision} · ${new Date(s.at*1000).toLocaleTimeString()}</option>`).join("")}</select><pre id="snapshot-code" style="max-height:420px;overflow:auto"></pre></section><section class="panel"><h2>${tr("Execution history","执行历史")}</h2><div class="timeline">${[...data.runs].reverse().map(r=>{const result=r.result?JSON.parse(r.result):null;return `<details><summary>${esc(taskName(r.task))} · ${esc(r.purpose)} · <b class="${result?.status==="passed"?"pass":""}">${esc(result?.status||r.status)}</b></summary><pre>${esc(pretty(result||{}))}</pre>${["grade","attempt_grade"].includes(r.purpose)&&["infrastructure_error",undefined].includes(result?.status)&&!["queued","running","deferred"].includes(r.status)?`<button data-retry="${r.id}">${tr("Retry frozen submission","重新评估原提交")}</button>`:""}</details>`;}).join("")||`<p>${tr("No executions yet","尚无执行记录")}</p>`}</div></section><section class="panel"><h2>${tr("Activity timeline","活动时间线")}</h2><div class="timeline">${[...data.events].reverse().map(e=>`<p><span class="muted">${new Date(e.at*1000).toLocaleTimeString()}</span> · ${esc(e.kind)} ${e.task?`· ${esc(taskName(e.task))}`:""}</p>`).join("")}</div></section><section class="panel full"><h2>${tr("Task outcomes & ratings","任务结果与评分")}</h2><pre>${esc(pretty(data.sessions.map(({code,...s})=>s)))}</pre></section></div></main>`;
  bindLanguage(); if(previousRevision&&data.snapshots.some(s=>String(s.id)===previousRevision))$("snapshot-select").value=previousRevision; const show=()=>{$("snapshot-code").textContent=data.snapshots.find(s=>String(s.id)===$("snapshot-select").value)?.code||tr("No draft yet","尚无草稿");};show();listen("snapshot-select",show,"change");
  listen("back-monitor",()=>{adminDetail=null;renderAdmin();});listen("refresh-detail",()=>inspectParticipant(ident));
  listen("pause-participant",async()=>{await api("/api/admin/pause",{token:ident,paused:true},true);notify(tr("Task paused","任务已暂停"));await inspectParticipant(ident);});
  listen("resume-participant",async()=>{await api("/api/admin/pause",{token:ident,paused:false},true);notify(tr("Task resumed","任务已恢复"));await inspectParticipant(ident);});
  document.querySelectorAll("[data-retry]").forEach(b=>b.onclick=runSafely(async()=>{await api("/api/admin/retry-grade",{run_id:b.dataset.retry},true);await inspectParticipant(ident);}));
  // Explicit refresh keeps an inspected revision and expanded execution logs stable.
}

window.addEventListener("pagehide",()=>{
  rememberEditor();
  pageClosed=true;
  if(token && state?.stage==="task") {
    const body=JSON.stringify({action:"close",page_id:pageId,task:state.task.id});
    navigator.sendBeacon(`${basePath}/api/p/${token}/presence`,new Blob([body],{type:"application/json"}));
  }
});
window.addEventListener("pageshow",()=>{
  if(!pageClosed) return;
  pageClosed=false;
  if(state?.stage==="task") presence("open").then(next=>{state=next;syncClock(state);renderPause();}).catch(e=>notify(e.message));
});
window.addEventListener("beforeunload",e=>{if(editor&&editor.state.doc.toString()!==savedCode){e.preventDefault();e.returnValue="";}});
document.addEventListener("visibilitychange",()=>event(document.hidden?"blur":"focus"));
window.addEventListener("online",()=>{event("connection_restored");if(editor)flushSave().catch(e=>notify(e.message));refreshState();});
setInterval(tick,250);setInterval(refreshState,2000);
try {
  if(token) {state=await participant("state");if(state.stage==="task") state=await presence("open");renderState();}
  else if(isAdmin) await renderAdmin();
  else landing();
} catch(error) {
  app.innerHTML=topbar()+`<main class="welcome"><div class="card"><h1>${tr("Unable to open this session","无法打开此研究")}</h1><p>${esc(error.message)}</p><p class="muted">${tr("Check your personal link or contact the researcher. Your previously saved work is retained.","请检查您的专属链接或联系研究人员。之前保存的代码仍会保留。")}</p><button id="reload">${tr("Try again","重试")}</button></div></main>`;
  listen("reload",()=>location.reload());bindLanguage();
}
