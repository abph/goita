/* Animal faces for the basic and royal tutorials. Piece IDs and rules stay shared. */
(() => {
  'use strict';
  const PIECES = Object.freeze({
    '1': {label:'し', color:'#f5f2e9', file:'shi-sparrow-pixel.png', names:{ja:'すずめ',en:'Sparrow',zh:'麻雀'}, pokemon:{ja:'ポッポ',en:'Pidgey',zh:'波波'}},
    '2': {label:'香', color:'#c74747', file:'ko-mouse-pixel.png', names:{ja:'ねずみ',en:'Mouse',zh:'老鼠'}, pokemon:{ja:'コラッタ',en:'Rattata',zh:'小拉达'}},
    '3': {label:'馬', color:'#c74747', file:'uma-horse-pixel.png', names:{ja:'うま',en:'Horse',zh:'马'}, pokemon:{ja:'ポニータ',en:'Ponyta',zh:'小火马'}},
    '4': {label:'銀', color:'#2985b0', file:'gin-baby-squirrel-pixel.png', names:{ja:'こりす',en:'Little squirrel',zh:'小松鼠'}, pokemon:{ja:'ピカチュウ',en:'Pikachu',zh:'皮卡丘'}},
    '5': {label:'金', color:'#2985b0', file:'kin-flying-squirrel-pixel.png', names:{ja:'むささび',en:'Flying squirrel',zh:'鼯鼠'}, pokemon:{ja:'ライチュウ',en:'Raichu',zh:'雷丘'}},
    '6': {label:'角', color:'#30323a', file:'kaku-dragon-pixel.png', names:{ja:'りゅう',en:'Dragon',zh:'龙'}, pokemon:{ja:'カイリュー',en:'Dragonite',zh:'快龙'}},
    '7': {label:'飛', color:'#30323a', file:'hi-winged-dragon-pixel.png', names:{ja:'ひりゅう',en:'Winged dragon',zh:'飞龙'}, pokemon:{ja:'リザードン',en:'Charizard',zh:'喷火龙'}},
    '8': {label:'玉', color:'#68509b', file:'gyoku-kitten-pixel.png', names:{ja:'こねこ',en:'Kitten',zh:'小猫'}, pokemon:{ja:'ミュウ',en:'Mew',zh:'梦幻'}},
    '9': {label:'王', color:'#68509b', file:'ou-lynx-pixel.png', names:{ja:'やまねこ',en:'Wildcat',zh:'山猫'}, pokemon:{ja:'ミュウツー',en:'Mewtwo',zh:'超梦'}},
  });
  const lang = language => ['ja','en','zh'].includes(language) ? language : 'ja';
  const isActive = state => state?.tutorial_mode === true && ['basic','royal'].includes(state.tutorial_chapter)
    && state.tutorial_piece_theme === 'animals';

  function renderPiece(element, piece, _label, {hidden = false, language = 'ja'} = {}) {
    const container = element.parentElement;
    const definition = PIECES[String(piece)];
    element.replaceChildren();
    delete element.dataset.pieceEn;
    element.setAttribute('data-i18n-ignore','');
    element.classList.add('tutorial-animal-value');
    // No piece-dependent class, colour, label, image or title is kept on a back.
    container.classList.remove('tutorial-animal-piece','tutorial-animal-back');
    container.style.removeProperty('--tutorial-piece-color');
    element.removeAttribute('title');
    element.removeAttribute('role');
    element.removeAttribute('aria-label');
    if (hidden || !definition) {
      container.classList.add('tutorial-animal-back');
      element.setAttribute('aria-label', {ja:'伏せ駒',en:'Face-down piece',zh:'伏牌'}[lang(language)]);
      return;
    }
    const name = definition.names[lang(language)];
    container.classList.add('tutorial-animal-piece');
    container.style.setProperty('--tutorial-piece-color', definition.color);
    element.setAttribute('role','img');
    element.setAttribute('aria-label', `${definition.label}（${name}）`);
    element.title = `${definition.label}（${name}）`;
    const icon = document.createElement('img');
    icon.src = `/static/player-icons/${definition.file}`;
    icon.alt = '';
    icon.setAttribute('aria-hidden','true');
    const label = document.createElement('span');
    label.textContent = definition.label;
    element.append(icon,label);
  }

  function renderLegend(grid, intro, language) {
    language = lang(language);
    intro.textContent = {
      ja:'説明には「香・馬・銀」など通常のごいたの駒名を使います。赤いねずみは香（コラッタ）、赤いうまは馬（ポニータ）です。絵の下の文字で駒名を確認できます。',
      en:'Instructions use traditional Goita piece names. The red Mouse is Lance (Rattata); the red Horse is Knight (Ponyta). The character below each animal identifies the piece.',
      zh:'说明使用香、马、银等普通Goita棋子名称。红色老鼠是香（小拉达），红色马是马（小火马）。请查看动物下方的文字确认棋子名称。',
    }[language];
    if (grid.dataset.language === language) return;
    grid.dataset.language = language;
    grid.replaceChildren();
    for (const [piece, definition] of Object.entries(PIECES)) {
      const row = document.createElement('div');
      row.className = 'tutorial-animal-key';
      row.setAttribute('data-i18n-ignore','');
      const face = document.createElement('div');
      face.className = 'tutorial-animal-key-piece';
      const value = document.createElement('div');
      value.className = 'val';
      face.append(value);
      renderPiece(value,piece,definition.label,{language});
      const text = document.createElement('div');
      text.className = 'tutorial-animal-key-text';
      const name = document.createElement('strong');
      name.textContent = `${definition.label}（${definition.names[language]}）`;
      const pokemon = document.createElement('small');
      pokemon.textContent = definition.pokemon[language];
      text.append(name,pokemon);
      row.append(face,text);
      grid.append(row);
    }
  }

  window.goitaTutorialAnimals = Object.freeze({isActive,renderPiece,renderLegend,PIECES});
})();
