const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
class Element {
  constructor(tag = 'div') {
    this.tagName = tag; this.children = []; this.dataset = {}; this.attrs = new Map();
    this.properties = new Map(); this.classes = new Set(); this.textContent = '';
    this.classList = {add: (...xs) => xs.forEach(x => this.classes.add(x)),
      remove: (...xs) => xs.forEach(x => this.classes.delete(x)), contains: x => this.classes.has(x)};
    this.style = {setProperty: (k,v) => this.properties.set(k,v), removeProperty: k => this.properties.delete(k)};
  }
  setAttribute(k,v) {this.attrs.set(k,v);}
  removeAttribute(k) {this.attrs.delete(k);}
  get title() {return this.attrs.get('title');}
  set title(v) {this.attrs.set('title',v);}
  append(...xs) {for (const x of xs) {x.parentElement = this; this.children.push(x);}}
  replaceChildren(...xs) {this.children = []; this.textContent = ''; this.append(...xs);}
}
const context = vm.createContext({window:{},document:{createElement: tag => new Element(tag)}});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../frontend/tutorialAnimals.js'),'utf8'),context);
const animals = context.window.goitaTutorialAnimals;
const active = {tutorial_mode:true,tutorial_chapter:'basic',tutorial_piece_theme:'animals',turn:'A'};
assert.equal(animals.isActive(active),true);
assert.equal(animals.isActive({...active,tutorial_chapter:'royal'}),true);
for (const state of [{...active,tutorial_mode:false},{...active,tutorial_chapter:'attack'},
                     {...active,tutorial_chapter:'cooperation'},
                     {...active,tutorial_piece_theme:'standard'},{}]) assert.equal(animals.isActive(state),false);
const expected = ['ポッポ','コラッタ','ポニータ','ピカチュウ','ライチュウ','カイリュー','リザードン','ミュウ','ミュウツー'];
assert.equal(Object.keys(animals.PIECES).length,9);
let back;
for (const [piece,def] of Object.entries(animals.PIECES)) {
  assert.equal(def.pokemon.ja,expected[Number(piece)-1]);
  assert.ok(fs.existsSync(path.join(__dirname,'../frontend/player-icons',def.file)),def.file);
  const parent = new Element(); const value = new Element(); parent.append(value);
  value.dataset.pieceEn = 'old';
  animals.renderPiece(value,piece,def.label,{language:'ja'});
  assert.equal(parent.properties.get('--tutorial-piece-color'),def.color);
  assert.equal(value.children[0].src,'/static/player-icons/'+def.file);
  assert.equal(value.children[1].textContent,def.label);
  assert.ok(value.attrs.get('aria-label').includes(def.names.ja));
  assert.equal(value.dataset.pieceEn,undefined);
  // Reuse a face as a back: its previous image, colour and name must disappear.
  animals.renderPiece(value,piece,def.label,{hidden:true,language:'ja'});
  const hidden = JSON.stringify({children:value.children,attrs:[...value.attrs],color:[...parent.properties],classes:[...parent.classes]});
  if (back) assert.equal(hidden,back); else back = hidden;
  assert.equal(value.children.length,0);
  assert.equal(value.attrs.get('title'),undefined);
  assert.ok(!parent.classList.contains('tutorial-animal-piece'));
}
assert.equal(animals.PIECES['2'].color,animals.PIECES['3'].color);
assert.equal(animals.PIECES['4'].color,animals.PIECES['5'].color);
assert.equal(animals.PIECES['6'].color,animals.PIECES['7'].color);
assert.equal(animals.PIECES['8'].color,animals.PIECES['9'].color);
for (const language of ['ja','en','zh']) {
  const grid = new Element(); const intro = new Element();
  animals.renderLegend(grid,intro,language);
  assert.equal(grid.children.length,9);
  assert.ok(intro.textContent.length > 0);
  for (const index of [0,5,9,10,14,19,20]) {
    const text = animals.copy({...active,tutorial_action_index:index},language);
    assert.equal(text.length,3);
    assert.ok(text.every(x => typeof x === 'string' && x.length > 0));
  }
  assert.ok(animals.copy({...active,tutorial_completed:true},language)[0]);
  const royal = {...active,tutorial_chapter:'royal'};
  for (const index of [3,4,9,13,14,18,22]) {
    const text = animals.copy({...royal,tutorial_action_index:index},language);
    assert.equal(text.length,3);
    assert.ok(text.every(x => typeof x === 'string' && x.length > 0));
  }
  assert.ok(animals.copy({...royal,tutorial_completed:true},language)[0]);
  const explanation = new Element();
  animals.renderRoyalExplanation(explanation,language);
  const lines = node => [node.textContent,...node.children.flatMap(lines)];
  const all = lines(explanation).join(' ');
  assert.ok(all.includes('50') && all.includes('100'));
  for (const id of ['1','2','8','9']) {
    const name = language === 'en' && id === '2' ? 'Mice' : animals.PIECES[id].names[language];
    assert.ok(all.includes(name),`${language}: ${name}`);
  }
}
console.log('Animal tutorial: nine mappings/assets, face rendering, identical concealed backs and three-language guidance passed');
