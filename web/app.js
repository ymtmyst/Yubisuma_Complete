const $ = (s) => document.querySelector(s);
const skillTips = {
  'フラッシュ':'互いの親指が同数なら手を2つ降ろす','セメント':'上げた親指を最低本数として固定する','ガード':'2手同時降ろしを防ぎ、追加ターン','チャージ':'次の数字宣言を2回分発動','クイック':'次の自分のターンで再宣言すると手を降ろす','スキップ':'相手の次フェーズの宣言を封じる','フェイント':'カウンターされた時、手を降ろし追加ターン','ロック':'カウンターされた時、相手のリアクションを封じる','コピー':'直前のスキル効果を2回発動','ストック':'直前のスキルを保存する','チョイス':'ストックから1つ発動','オール':'ストックすべてを順番に発動','ドロップ':'ストックのスキルを相手に封じる','ブースト':'ゲーム中1回。追加3ターン','リバーシ':'ゲーム中1回。互いの状態を交換','タイム':'ゲーム中1回。相手の連続行動へ割り込む'
};
let state, selectedAction=null, selectedThumb=null, selectedReaction=null, choiceOrder=[];

async function api(path, data) {
  document.body.classList.add('loading');
  try {
    const res = await fetch(path,{method:data?'POST':'GET',headers:{'Content-Type':'application/json'},body:data?JSON.stringify(data):undefined});
    const json = await res.json();
    if(!res.ok) throw new Error(json.error || '通信に失敗しました');
    return json;
  } finally { document.body.classList.remove('loading'); }
}
function hand(active, up, label) {
  return `<div class="hand-wrap ${active?'':'gone'} reveal"><div class="hand ${up?'up':''}"><i class="fingers"></i><i class="palm"></i><i class="thumb"></i></div><span class="hand-label">${active?label:'DOWN'}</span></div>`;
}
function renderHands(target, player, shownThumb=0) {
  const active=[player.left,player.right].filter(Boolean).length;
  let remaining=Math.min(shownThumb,active);
  const leftUp=player.left && remaining-- > 0, rightUp=player.right && remaining-- > 0;
  target.innerHTML=hand(player.left,leftUp,'LEFT')+hand(player.right,rightUp,'RIGHT');
}
function renderEffects(target, effects) { target.innerHTML=effects.map(e=>`<span class="effect">${e.label}</span>`).join(''); }
function renderStock(target, stock) { target.innerHTML=stock.length?`STOCK ${stock.map(s=>`<span class="stock-token">${s}</span>`).join('')}`:'<span>STOCK — EMPTY</span>'; }
function button(value,label,cls='choice') { return `<button class="${cls}" data-value="${value}">${label}</button>`; }
function bindChoices(container, callback) { container.querySelectorAll('button').forEach(b=>b.onclick=()=>callback(b.dataset.value,b)); }
function selectIn(container, button) { container.querySelectorAll('button').forEach(b=>b.classList.toggle('selected',b===button)); }

function resetSelections(){ selectedAction=null;selectedThumb=null;selectedReaction=null;choiceOrder=[]; }
function render() {
  $('#phaseBadge').textContent=`ROUND ${String(state.phase).padStart(2,'0')}`;
  $('#cpuTurn').classList.toggle('active',state.turn==='computer' && state.mode!=='gameover');
  $('#playerTurn').classList.toggle('active',state.turn==='player' && state.mode!=='gameover');
  renderEffects($('#cpuEffects'),state.computer.effects); renderEffects($('#playerEffects'),state.player.effects);
  renderStock($('#cpuStock'),state.computer.stock); renderStock($('#playerStock'),state.player.stock);
  renderHands($('#playerHands'),state.player,selectedThumb??0); renderHands($('#cpuHands'),state.computer,0);
  $('#gameLog').innerHTML=[...state.log].reverse().map(x=>`<li>${escapeHtml(x)}</li>`).join('');
  $('#attackControls').hidden=state.mode!=='attack'; $('#defenseControls').hidden=state.mode!=='defense'; $('#choiceControls').hidden=state.mode!=='choice';
  $('#thumbControls').hidden=state.mode==='choice'||state.mode==='gameover';
  const messages={attack:'あなたのターン。宣言と親指を選んでください',defense:'相手が構えました。親指とリアクションを同時に決めます',choice:'相手のリアクション公開後、ストックを選びます',gameover:state.winner==='player'?'YOU WIN — お見事！':'YOU LOSE — もう一勝負？'};
  $('#statusMessage').textContent=messages[state.mode]||'';
  $('#callout').innerHTML=state.mode==='gameover'?`<small>GAME SET</small><strong>${state.winner==='player'?'勝利！':'惜しい！'}</strong>`:state.mode==='defense'?'<small>KEEP IT SECRET</small><strong>相手の宣言は秘密</strong>':'<small>YOUR CALL</small><strong>いくつ上がる？</strong>';
  document.body.classList.toggle('gameover',state.mode==='gameover');
  if(state.mode==='attack') renderAttack(); if(state.mode==='defense') renderDefense(); if(state.mode==='choice') renderChoice();
  renderThumbs(); updateSubmit();
}
function renderAttack(){
  $('#numberChoices').innerHTML=state.numbers.map(n=>button(n,n)).join('');
  bindChoices($('#numberChoices'),(v,b)=>{selectedAction=Number(v);selectIn($('#numberChoices'),b);$('#skillChoices').querySelectorAll('button').forEach(x=>x.classList.remove('selected'));updateSubmit();});
  $('#skillChoices').innerHTML=state.skills.map(s=>`<button class="skill" data-value="${s}" data-tip="${skillTips[s]||'特殊な効果を発動します'}">${s}${['ブースト','リバーシ','タイム'].includes(s)?' ★':''}</button>`).join('');
  bindChoices($('#skillChoices'),(v,b)=>{selectedAction=v;selectIn($('#skillChoices'),b);$('#numberChoices').querySelectorAll('button').forEach(x=>x.classList.remove('selected'));updateSubmit();});
}
function renderDefense(){
  $('#reactionChoices').innerHTML=state.reactions.map(r=>button(r,r)).join('');
  bindChoices($('#reactionChoices'),(v,b)=>{selectedReaction=v;selectIn($('#reactionChoices'),b);updateSubmit();});
}
function renderThumbs(){
  const c=$('#thumbChoices'); if(state.mode==='choice'||state.mode==='gameover'){c.innerHTML='';return;}
  let opts=[]; for(let i=state.thumb_min;i<=state.player.hands;i++)opts.push(i);
  c.innerHTML=opts.map(n=>button(n,`${n}本`)).join('');
  bindChoices(c,(v,b)=>{selectedThumb=Number(v);selectIn(c,b);renderHands($('#playerHands'),state.player,selectedThumb);updateSubmit();});
}
function renderChoice(){
  const all=state.choice.type==='オール'; $('#choiceTitle').textContent=all?'発動順を選ぶ':'発動するスキルを選ぶ';
  $('#reactionReveal').textContent=`相手のリアクション：${state.choice.reaction||'なし'}`;
  $('#stockChoices').innerHTML=state.choice.stock.map((s,i)=>`<button class="stock-choice" data-index="${i}" data-value="${s}">${s}</button>`).join('');
  bindChoices($('#stockChoices'),(v,b)=>{
    if(all){ const i=Number(b.dataset.index); if(!choiceOrder.includes(i)) choiceOrder.push(i); else choiceOrder=choiceOrder.filter(x=>x!==i); $('#stockChoices').querySelectorAll('button').forEach((x,j)=>{const pos=choiceOrder.indexOf(j);x.classList.toggle('selected',pos>=0);x.innerHTML=(pos>=0?`<span class="order">${pos+1}</span>`:'')+x.dataset.value;}); }
    else { choiceOrder=[Number(b.dataset.index)]; selectIn($('#stockChoices'),b); }
    updateSubmit();
  });
}
function updateSubmit(){
  let ready=false,label='宣言する';
  if(state?.mode==='attack')ready=selectedAction!==null&&selectedThumb!==null;
  if(state?.mode==='defense'){ready=selectedReaction!==null&&selectedThumb!==null;label='同時に公開する';}
  if(state?.mode==='choice'){ready=choiceOrder.length===(state.choice.type==='オール'?state.choice.stock.length:1);label=state.choice.type==='オール'?'この順番で発動':'これを発動';}
  if(state?.mode==='gameover'){ready=true;label='もう一度遊ぶ';}
  $('#submitButton').disabled=!ready;$('#submitButton b').textContent=label;
}
async function submit(){
  try{
    if(state.mode==='gameover'){await newGame();return;}
    let payload={};
    if(state.mode==='attack')payload={action:selectedAction,thumbs:selectedThumb};
    if(state.mode==='defense')payload={reaction:selectedReaction==='なし'?null:selectedReaction,thumbs:selectedThumb};
    if(state.mode==='choice')payload=state.choice.type==='オール'?{order:choiceOrder.map(i=>state.choice.stock[i])}:{choice:state.choice.stock[choiceOrder[0]]};
    state=await api('/api/action',payload);resetSelections();render();beep();
  }catch(e){toast(e.message)}
}
async function newGame(){state=await api('/api/new',{});resetSelections();render();}
function toast(msg){const t=$('#toast');t.textContent=msg;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2200)}
function escapeHtml(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
let sound=true;function beep(){if(!sound)return;try{const c=new AudioContext(),o=c.createOscillator(),g=c.createGain();o.connect(g);g.connect(c.destination);o.frequency.value=180;g.gain.setValueAtTime(.035,c.currentTime);g.gain.exponentialRampToValueAtTime(.001,c.currentTime+.12);o.start();o.stop(c.currentTime+.12)}catch{}}
$('#submitButton').onclick=submit;$('#newButton').onclick=()=>confirm('現在のゲームをやり直しますか？')&&newGame();
$('#skillToggle').onclick=()=>{$('#skillChoices').classList.toggle('collapsed');$('#skillToggle span').textContent=$('#skillChoices').classList.contains('collapsed')?'＋':'−'};
$('#logToggle').onclick=()=>$('.log-drawer').classList.toggle('open');
$('#rulesButton').onclick=()=>$('#rulesDialog').showModal();$('#closeRules').onclick=()=>$('#rulesDialog').close();
$('#soundButton').onclick=()=>{sound=!sound;$('#soundButton').textContent=sound?'♪':'×';};
api('/api/state').then(s=>{state=s;render()}).catch(e=>toast(e.message));
