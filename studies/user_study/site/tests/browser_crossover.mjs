import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {spawn,execFileSync} from 'node:child_process';
import {mkdtemp,readFile,writeFile,mkdir,rm} from 'node:fs/promises';
import {existsSync} from 'node:fs';
import {tmpdir,homedir} from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const site=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),repo=path.resolve(site,'../../..');
const runtime=await mkdtemp(path.join(tmpdir(),'zinnia-crossover-browser-'));
const artifacts=path.join(site,'test-results');await mkdir(artifacts,{recursive:true});
const python=path.join(repo,'.venv/bin/python'),origin='http://127.0.0.1:18878'+(process.env.STUDY_TEST_BASE_PATH||'');
const source=JSON.parse(execFileSync(python,['-c','import json; from studies.user_study.site.tests.reference import PYTHON,RUST; print(json.dumps({"zinnia":PYTHON,"rust":RUST}))'],{cwd:repo,encoding:'utf8'}));
const server=spawn(python,['-m','studies.user_study.site.server','--port','18878','--runtime',runtime],{cwd:repo,stdio:['ignore','pipe','pipe']});
let log='';server.stdout.on('data',b=>log+=b);server.stderr.on('data',b=>log+=b);
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function until(fn,timeout=90000){const start=Date.now();while(Date.now()-start<timeout){const value=await fn();if(value)return value;await delay(100);}throw new Error('Timeout');}
let browser;const errors=[];
try {
 await until(async()=>{try{return (await fetch(origin+'/health')).ok;}catch{return false;}});
 const config=JSON.parse(await readFile(path.join(runtime,'config.json'),'utf8'));
 async function api(route,data,admin=false,expected=200){
  const response=await fetch(origin+route,{method:data?'POST':'GET',headers:{...(data?{'Content-Type':'application/json'}:{}),...(admin?{Authorization:`Bearer ${config.admin_key}`}:{})},body:data?JSON.stringify(data):undefined});
  assert.equal(response.status,expected,await response.clone().text());return response.json();
 }
 const cached=path.join(homedir(),'.cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell');
 const executablePath=process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE||(existsSync(chromium.executablePath())?chromium.executablePath():cached);
 browser=await chromium.launch({headless:true,executablePath});
 const admin=await browser.newPage({viewport:{width:1600,height:1050}});admin.on('pageerror',e=>errors.push(e.message));
 await admin.goto(origin+'/admin#key='+config.admin_key);
 assert.equal(await admin.locator('#assignment-design').inputValue(),'crossover-12x6-v1');
 assert.equal(await admin.locator('#participant-count').inputValue(),'12');assert.equal(await admin.locator('#participant-count').isDisabled(),true);
 await admin.locator('#create-participants').click();await until(async()=>await admin.locator('[data-inspect]').count()===12);
 const dashboard=await api('/api/admin',undefined,true),people=dashboard.participants;
 assert.equal(people.length,12);
 const cohort=people[0].cohort_id;
 assert.ok(people.every(p=>p.design==='crossover-12x6-v1'&&p.cohort_id===cohort));
 const chosen=people.filter(p=>[1,2].includes(p.schedule_slot)).sort((a,b)=>a.schedule_slot-b.schedule_slot);
 let practiceRuns=0,formalRuns=0;
 for(const p of chosen){
  const route='/api/p/'+p.token;
  const page=await browser.newPage({viewport:{width:1440,height:1050}});page.on('pageerror',e=>errors.push(e.message));
  async function editAndRun(code){
   await page.locator('.cm-content').click();await page.keyboard.press('Control+a');await page.keyboard.insertText(code);
   await until(async()=>await page.locator('#save-status').textContent()==='Saved');
   const before=(await api(route+'/state')).runs.map(r=>r.id);
   await page.locator('#run-examples').click();
   const run=await until(async()=>{const r=(await api(route+'/state')).runs[0];return r&&!before.includes(r.id)&&r.status==='complete'?r:null;});
   assert.equal(run.result.status,'passed',JSON.stringify(run.result));
   await until(async()=>await page.locator('#run-examples').isEnabled());return run;
  }
  await page.goto(origin+'/p/'+p.token);await page.locator('#consent').check();await page.locator('#consent-next').click();
  for(const field of ['python','rust','arrays','zk'])await page.locator(`[name=${field}]`).selectOption('basic');
  await page.locator('#background-form button').click();
  for(let i=0;i<2;i++){
   const before=await api(route+'/state');assert.equal(before.stage,'tutorial');assert.equal(before.condition,p.practice_order[i]);
   await page.locator('#practice-start').click();await page.locator('.cm-content').waitFor();
   await page.reload();await page.locator('.cm-content').waitFor();
   const state=await api(route+'/state');assert.equal(state.task.id,'practice_'+p.practice_order[i]+'_v2');assert.equal(state.remaining,null);
   await page.locator('.cm-content').click();await page.keyboard.press('Control+a');await page.keyboard.insertText('x');
   await page.keyboard.press('Home');
   const beforeTab=await page.locator('.cm-content').innerText();
   await page.keyboard.press('Tab');
   assert.equal(await page.locator('.cm-content').evaluate(el=>el.contains(document.activeElement)),true);
   assert.equal((await page.locator('.cm-content').innerText()).length,beforeTab.length+4);
   await page.keyboard.press('Shift+Tab');
   assert.equal(await page.locator('.cm-content').evaluate(el=>el.contains(document.activeElement)),true);
   await api(route+'/action',{action:'practice_done'},false,409);
   const practiceRun=await editAndRun(source[state.condition][state.task.id]);
   assert.equal(practiceRun.result.engine,state.condition==='rust'?'risc0-zkvm':'zinnia-ir-mock');practiceRuns++;
   await page.locator('#finish-task').click();
   if(i===0)await page.locator('#practice-start').waitFor();else await page.locator('#task-start').waitFor();
  }
  for(let period=0;period<6;period++){
   const task=p.task_order[period],condition=p.task_conditions[task];
   const ready=await api(route+'/state');assert.equal(ready.condition,condition);assert.equal(ready.next_task.id,task);
   await page.locator('#task-start').click();await page.locator('.cm-content').waitFor();
   await page.reload();await page.locator('.cm-content').waitFor();
   const state=await api(route+'/state');assert.equal(state.session.condition,condition);assert.equal(state.session.period,period+1);
   assert.match(await page.locator('.file-label').textContent(),condition==='rust'?/solution\.rs/:/solution\.py/);
   const languageLabel=condition==='rust'?'Rust':'Python';
   assert.equal(await page.locator('.topbar [data-programming-language]').getAttribute('data-programming-language'),languageLabel);
   assert.equal(await page.locator('.file-label [data-programming-language]').getAttribute('data-programming-language'),languageLabel);
   assert.equal(await page.locator('.task-hint').count(),0);
   const manualButton=page.locator('.description [data-manual-topic]').first();
   await manualButton.click();await page.locator('#manual-results .manual-entry').first().waitFor();
   const manualText=await page.locator('#dialog').innerText();
   if(condition==='rust')assert.doesNotMatch(manualText,/zinnia|numpy|np\./i);
   await page.keyboard.press('Escape');await page.locator('#dialog').waitFor({state:'hidden'});
   await manualButton.click();await page.locator('#manual-results .manual-entry').first().waitFor();
   await page.mouse.click(2,2);await page.locator('#dialog').waitFor({state:'hidden'});

   await editAndRun(source[condition][task]);formalRuns++;
   await api(route+'/run',{purpose:'examples',condition:condition==='rust'?'zinnia':'rust'},false,403);
   await page.locator('#finish-task').click();await page.locator('#submit-form').waitFor();
   await page.locator('[name=difficulty][value="3"]').check();await page.locator('[name=confidence][value="6"]').check();
   await page.locator('#submit-form button[type=submit]').click();
   if(period<5)await page.locator('#task-start').waitFor();
   else await until(async()=>(await api(route+'/state')).stage==='complete');
  }
  await page.close();
 }
 const exported=await until(async()=>{
  const data=await api('/api/admin/export?cohort='+cohort,undefined,true);
  const hidden=data.runs.filter(r=>['attempt_grade','grade'].includes(r.purpose));
  return hidden.length===24&&hidden.every(r=>r.status==='complete')?data:null;
 });
 assert.equal(exported.design,'crossover-12x6-v1');assert.equal(exported.participants.length,12);assert.equal(exported.attempt_trail.length,24);
 for(const row of exported.attempt_trail){
  const p=people.find(p=>p.token===row.token);
  assert.equal(row.condition,p.task_conditions[row.task]);assert.equal(row.period,p.task_order.indexOf(row.task)+1);
  assert.equal(row.hidden_status,'passed');
 }
 for(const run of exported.runs.filter(r=>['attempt_grade','grade'].includes(r.purpose))){
  assert.equal(JSON.parse(run.result).engine,run.condition==='rust'?'risc0-zkvm':'zinnia-ir-mock');
  if(run.purpose==='attempt_grade')await api('/api/p/'+run.token+'/runs/'+run.id,undefined,false,404);
 }
 await admin.reload();await admin.locator(`[data-inspect="${chosen[0].token}"]`).click();
 await admin.getByRole('heading',{name:'Attempt trail and hidden scores',exact:true}).waitFor();
 assert.equal(await admin.locator('.detail-grid .table-scroll tbody tr').count(),12);
 await admin.screenshot({path:path.join(artifacts,'crossover-attempt-trail.png'),fullPage:true});
 await writeFile(path.join(artifacts,'crossover-study-export.json'),JSON.stringify(exported,null,2));
 assert.deepEqual(errors,[]);
 const report={status:'passed',cohortSize:12,completedParticipants:2,practiceRuns,formalRuns,hiddenAttempts:12,finalGrades:12,browserErrors:errors};
 await writeFile(path.join(artifacts,'crossover-browser-report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
}catch(error){await writeFile(path.join(artifacts,'crossover-server.log'),log);throw error;}
finally{if(browser)await browser.close();server.kill('SIGINT');await until(async()=>server.exitCode!==null||server.signalCode!==null,10000).catch(()=>server.kill('SIGKILL'));await rm(runtime,{recursive:true,force:true});}
