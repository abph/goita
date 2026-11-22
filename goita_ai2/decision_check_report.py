"""Offline HTML views for AI判断チェック; no external assets or services."""
from __future__ import annotations

import html
import json
from pathlib import Path


CSS = """
:root{font-family:system-ui,'Meiryo',sans-serif;color:#24332f;background:#f2f5f3}
*{box-sizing:border-box}body{margin:0 auto;max-width:1160px;padding:24px;overflow-wrap:anywhere}
h1{font-size:24px;margin:0 0 12px}h2{font-size:18px}h3{font-size:15px}
p{line-height:1.65}button,select,input,textarea{font:inherit}button,select{padding:8px 12px;border:1px solid #a9bdb3;border-radius:5px;background:white;color:inherit}
button{cursor:pointer}select{max-width:100%}button.primary{background:#245e47;color:white}button:disabled{opacity:.5;cursor:default}
textarea{width:100%;min-height:100px;padding:10px;border:1px solid #a9bdb3;border-radius:5px}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:20px 0}
.card{background:white;border:1px solid #cad6d0;border-radius:9px;padding:20px;margin:16px 0}
.muted{color:#566b61;font-size:13px}.notice{padding:12px;background:#e7f0eb;border-radius:6px}
.badge{padding:4px 9px;background:#e8eeeb;border-radius:5px;display:inline-block;font-size:13px}
.bad{background:#fae2de}.good{background:#deefe3}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;text-align:left;border-bottom:1px solid #dce4df;vertical-align:top}th{background:#f0f4f1}
.routes select{min-width:115px}.piece{display:inline-block;background:#f5ead4;border:1px solid #c9b38e;padding:5px 8px;margin:2px;border-radius:3px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}details{margin:12px 0}summary{cursor:pointer}
.label{white-space:pre-wrap}.error{color:#a02d22}#status{min-height:26px}a{color:#245e47}
@media(max-width:650px){body{padding:12px}.card{padding:12px}.grid{grid-template-columns:1fr}td,th{padding:6px}.routes td:first-child{width:65%}}
"""


COMMON = r"""
const kanji={'1':'し','2':'香','3':'馬','4':'銀','5':'金','6':'角','7':'飛','8':'玉','9':'王'};
function el(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e}
function label(route){if(!route)return '記録なし';return route.map(([k,b,a])=>k==='pass'?'パス':k==='receive'?kanji[b]+'で受ける':k==='attack'?kanji[a]+'で攻める':kanji[b]+'を伏せる → '+kanji[a]+'で攻める').join(' → ')}
const grades={best:'最善手',acceptable:'許容できる手',avoid:'避けたい手',unlisted:'未登録の手・要確認',unconfirmed:'正解未確認',error:'実行エラー',excluded:'除外'};
const changes={first:'初回',improved:'改善',maintained:'維持',regressed:'後退',needs_review:'要確認',conditions_changed:'条件・正解が変更',unconfirmed:'正解未確認',error:'実行エラー',excluded:'除外'};
function detail(parent,title,value){const d=el('details');d.append(el('summary',title),el('pre',typeof value==='string'?value:JSON.stringify(value,null,2)));parent.append(d)}
function download(value,name){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=el('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000)}
"""


REVIEW = r"""
const data=JSON.parse(document.getElementById('data').textContent);
const drafts=new Map();let current=0,style='safe';
const key='ai-decision-check-v1-'+data.updated_at+'-'+data.cases.map(c=>c.fingerprint).join('-');
try{for(const [k,v] of JSON.parse(localStorage.getItem(key)||'[]'))drafts.set(k,v)}catch(e){}
function draft(c){const k=c.id+':'+style;if(!drafts.has(k)){const a=c.answers[style];drafts.set(k,a?structuredClone(a):{best:[],acceptable:[],avoid:[],reason:c.original.human_review?.reason||'',confirmed:false})}return drafts.get(k)}
function save(){try{localStorage.setItem(key,JSON.stringify([...drafts]))}catch(e){document.getElementById('status').textContent='ブラウザの一時保存が使えません。回答JSONを書き出してください。'}}
const toolbar=document.getElementById('toolbar'),list=el('select'),styler=el('select');
data.cases.forEach((c,i)=>{const o=el('option',(i+1)+'. '+c.label.replaceAll('\n',' '));o.value=i;list.append(o)});
for(const [v,t] of [['safe','安全思考'],['risk','リスク思考（現在は終盤し攻めのみ）']]){const o=el('option',t);o.value=v;styler.append(o)}
const prev=el('button','← 前へ'),next=el('button','次へ →'),exporter=el('button','確定済み回答を書き出す','primary');
toolbar.append(prev,list,next,styler,exporter);
list.onchange=()=>{current=+list.value;render()};styler.onchange=()=>{style=styler.value;render()};prev.onclick=()=>{current--;render()};next.onclick=()=>{current++;render()};
function render(){
 const root=document.getElementById('main');root.replaceChildren();list.value=current;prev.disabled=current<=0;next.disabled=current>=data.cases.length-1;
 if(!data.cases.length){root.append(el('p','レポートがありません。先にimportコマンドで取り込んでください。'));return}
 const c=data.cases[current],a=draft(c),card=el('section',undefined,'card');root.append(card);
 card.append(el('h2',c.label,'label'),el('p',c.id,'muted'));
 if(c.excluded){card.append(el('p','除外：'+c.excluded,'error'));return}
 const status=el('p',a.confirmed?'この回答は確定済みです。書き出してanswersコマンドで取り込んでください。':'下書きです。選択後に「この回答を確定」を押してください。','notice');card.append(status);
 const invalidate=()=>{a.confirmed=false;status.textContent='変更した回答は下書きです。「この回答を確定」を押してください。';save()};
 const grid=el('div',undefined,'grid'),hand=el('div'),publics=el('div');grid.append(hand,publics);card.append(grid);
 hand.append(el('h3',c.position.player+'の対象手番直前の手駒'));
 // Reconstruct only the actor's hand. Other hidden pieces never appear here.
 const pieces=[...c.position.initial_hands[c.position.player]];
 for(const h of c.position.history){if(h.player!==c.position.player)continue;const [k,b,t]=h.action;for(const p of (k==='attack_after_block'?[b,t]:k==='receive'?[b]:k==='attack'?[t]:[])){const n=pieces.indexOf(p);if(n>=0)pieces.splice(n,1)}}
 const tiles=el('div');pieces.sort().forEach(p=>tiles.append(el('span',kanji[p],'piece')));hand.append(tiles);
 hand.append(el('p','当時の判断：'+label(c.original.route)));
 hand.append(el('p','提案（未確定）：'+(c.original.human_review?.preferred_action||'記入なし'),'label'));
 publics.append(el('h3','対象より前の公開手順'));
 const turns=[];for(const h of c.position.history){const last=turns.at(-1);if(last&&last.player===h.player&&last.route.at(-1)[0]==='receive'&&h.action[0]==='attack')last.route.push(h.action);else turns.push({player:h.player,route:[h.action]})}
 turns.forEach((h,i)=>{let text=label(h.route);if(h.route[0][0]==='attack_after_block'&&h.player!==c.position.player)text='伏せ（非公開） → '+kanji[h.route[0][2]]+'で攻める';publics.append(el('div',h.player+'手番'+(i+1)+'：'+text))});
 if(!c.position.history.length)publics.append(el('p','親の最初の攻めです。'));
 card.append(el('h3','手順の評価（受けた後の攻めまで含む）'),el('p','最善手は複数選べます。未分類の手は不正解と断定せず、要確認になります。','muted'));
 const table=el('table',undefined,'routes'),head=el('tr');head.append(el('th','手順'),el('th','評価'));table.append(head);
 c.legal_routes.forEach(route=>{const tr=el('tr'),td=el('td',label(route)),select=el('select');
  for(const [v,t] of [['','未分類'],['best','最善手'],['acceptable','許容できる手'],['avoid','避けたい手']]){const o=el('option',t);o.value=v;select.append(o)}
  select.value=['best','acceptable','avoid'].find(g=>a[g].some(r=>JSON.stringify(r)===JSON.stringify(route)))||'';
  select.onchange=()=>{for(const g of ['best','acceptable','avoid'])a[g]=a[g].filter(r=>JSON.stringify(r)!==JSON.stringify(route));if(select.value)a[select.value].push(route);invalidate()};
  const cell=el('td');cell.append(select);tr.append(td,cell);table.append(tr)});card.append(table);
 card.append(el('h3','判断理由・適用条件'));const reason=el('textarea');reason.value=a.reason;reason.placeholder='なぜこの手順がよいか。安全思考・リスク思考の違い、例外など。';reason.oninput=()=>{a.reason=reason.value;invalidate()};card.append(reason);
 const confirm=el('button','この回答を確定','primary');confirm.onclick=()=>{if(!a.best.length||!a.reason.trim()){status.textContent='最善手を1つ以上選び、判断理由を入力してください。';return}a.confirmed=true;save();status.textContent='この回答は確定済みです。回答JSONを書き出してanswersコマンドで取り込んでください。'};card.append(el('div',undefined,'toolbar'));card.lastChild.append(confirm);
 detail(card,'当時のログ・提案・探索設定（最新AIの結果ではありません）',{original:c.original,settings:c.capture_settings});
}
exporter.onclick=()=>{const entries=[];for(const c of data.cases){if(c.excluded)continue;for(const s of ['safe','risk']){const a=drafts.get(c.id+':'+s)||c.answers[s];if(a?.confirmed)entries.push({id:c.id,fingerprint:c.fingerprint,style:s,answer:a})}}if(!entries.length){document.getElementById('status').textContent='確定済みの回答がありません。';return}download({format:data.format+'-answers',schema_version:1,cases:entries},'ai_decision_check_answers.json');document.getElementById('status').textContent=entries.length+'件の回答を書き出しました。answersコマンドで取り込んでください。'};
render();
"""


RESULTS = r"""
const data=JSON.parse(document.getElementById('data').textContent),toolbar=document.getElementById('toolbar');
document.getElementById('status').textContent='対象：'+data.ai_label+' / '+(data.options.style==='safe'?'安全思考':'リスク思考（終盤し攻め）')+' / '+data.results.length+'/'+data.total+'件 / '+({complete:'完了',incomplete:'一部未完了',interrupted:'中断',running:'実行中'}[data.status]||data.status);
const summary=el('div',undefined,'notice');summary.textContent=Object.entries(data.summary).map(([g,n])=>(grades[g]||g)+' '+n+'件').join(' ／ ');document.getElementById('main').before(summary);
const filters=el('select');for(const [v,t] of [['all','すべて'],['attention','要確認・避けたい手・エラー'],['changed','前回から手順・理由が変化'],['regressed','後退'],['improved','改善'],['unconfirmed','正解未確認']]){const o=el('option',t);o.value=v;filters.append(o)}toolbar.append(filters);filters.onchange=render;
function render(){const root=document.getElementById('main');root.replaceChildren();let count=0;
 for(const r of data.results){if(filters.value==='attention'&&!['avoid','unlisted','error'].includes(r.grade))continue;if(filters.value==='changed'&&!r.route_changed&&!r.reason_changed)continue;if(['regressed','improved','unconfirmed'].includes(filters.value)&&r.change!==filters.value)continue;count++;
 const card=el('section',undefined,'card');card.append(el('h2',r.label,'label'),el('span',grades[r.grade]||r.grade,'badge '+(['avoid','error','unlisted'].includes(r.grade)?'bad':'good')),el('span','前回比：'+(changes[r.change]||r.change),'badge'),el('p',r.id,'muted'));
 if(r.status==='ok'){
  card.append(el('p','最新AI2：'+label(r.route)),el('p','当時の判断：'+label(r.original.route),'muted'));
  if(r.previous){card.append(el('p','前回のAI2：'+label(r.previous.route)+'（'+(grades[r.previous.grade]||r.previous.grade)+'）','muted'));detail(card,'前回の理由・実行情報',r.previous)}
  if(r.answer){card.append(el('p','正解：'+r.answer.best.map(label).join(' ／ ')));if(r.answer.acceptable.length)card.append(el('p','許容：'+r.answer.acceptable.map(label).join(' ／ ')));card.append(el('p','理由：'+r.answer.reason,'label'))}
  card.append(el('p','実行 '+r.seconds+'秒','muted'));
  r.steps.forEach((step,i)=>card.append(el('p',(i+1)+'. '+label([step.action])+'：'+(step.reason||'理由記録なし')+' / '+(step.detail||''))));
  detail(card,'最新AIの候補評価・探索・ニューラル記録',r.steps);detail(card,'実行条件',r.conditions);
 }else card.append(el('p',r.error||r.status,'error'));
 detail(card,'当時のログと提案',r.original);root.append(card)}
 if(!count)root.append(el('p','該当する局面はありません。'));
}
const meta=el('section',undefined,'card');meta.append(el('h2','実行情報'));data.notes.forEach(n=>meta.append(el('p',n,'muted')));detail(meta,'コード・モデルのハッシュと実行日時',{run_id:data.run_id,started_at:data.started_at,finished_at:data.finished_at,code:data.code});document.getElementById('main').after(meta);render();
"""


def _write(path: Path, title: str, description: str, data: dict, script: str) -> None:
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    page = f"""<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style>
<body><h1>{html.escape(title)}</h1><p>{html.escape(description)}</p>
<div id="toolbar" class="toolbar"></div><p id="status" role="status"></p><main id="main"></main>
<script id="data" type="application/json">{payload}</script><script>{COMMON}\n{script}</script></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def write_review(path: Path, suite: dict) -> None:
    _write(path, "AI判断チェック・正解の確認",
           "提案文を参考に、最善手・許容できる手・避けたい手と理由を登録します。変更は下書きです。確定して回答JSONを書き出してください。ブラウザ内の一時保存だけでは問題集に反映されません。",
           suite, REVIEW)


def write_results(path: Path, result: dict) -> None:
    _write(path, "AI判断チェック・実行結果",
           "最新の強化中AI2が実際に選んだ手順を確認します。正解未確認や未登録の手は、自動で不正解にはしません。前回比は同じ正解・実行条件に限ります。",
           result, RESULTS)
