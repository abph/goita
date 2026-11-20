/* Animal faces for the basic tutorial only. Piece IDs and game rules stay shared. */
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
  const isActive = state => state?.tutorial_mode === true && state.tutorial_chapter === 'basic'
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
    element.setAttribute('aria-label', `${name}（${definition.label}）`);
    element.title = `${name}（${definition.label}）`;
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
      ja:'ポケモンごいたの駒を、対応する色と動物で表示します。コラッタは赤いねずみ、ポニータは赤いうまです。駒の小さな文字は、通常のごいたでの名前です。',
      en:'Pieces use matching colours and animal icons. Rattata is the red Mouse; Ponyta is the red Horse. The small character shows the traditional Goita piece.',
      zh:'用对应颜色与动物表示棋子。小拉达对应红色老鼠，小火马对应红色马。小字表示普通Goita棋子的名称。',
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
      name.textContent = `${definition.names[language]}（${definition.label}）`;
      const pokemon = document.createElement('small');
      pokemon.textContent = definition.pokemon[language];
      text.append(name,pokemon);
      row.append(face,text);
      grid.append(row);
    }
  }

  const COPY = {
    ja: {
      waiting:['案内役の動きを見ましょう','動物の種類を見ながら、次の自分の手番を待ちましょう。'],
      0:['1. 伏せて攻める（どうぶつ）','AとC、BとDが相方です。手駒を先にすべて出すと上がりです。まず、うまを伏せて、こりすで攻めましょう。'],
      5:['2. パスする','こりすを持っていないので受けられません。手駒の横にある「パス」を押しましょう。'],
      9:['3. 受けて攻め返す','相手がうまで攻めました。同じうまを選んで受けましょう。'],
      10:['3. 受けて攻め返す','受けた人が次に攻めます。赤いねずみを選んで攻めましょう。'],
      14:['4. 攻めを続ける','全員がパスすると、自分に攻めが戻ります。すずめを伏せて、ひりゅうで攻めましょう。'],
      19:['5. 上がる','相手のりゅうを、自分のりゅうで受けましょう。残り2枚です。'],
      20:['5. 上がる','最後にむささびで攻めると、手駒がなくなって上がりです。'],
      done:['どうぶつの基本操作編クリア！','色と動物で、受け・攻め・パス・上がりを練習できました。続きの王（玉）の練習は、通常の駒で行います。'],
      detail:'緑の枠が次に選ぶ駒です。伏せた駒は同じ裏面になります。この練習の得点と棋譜は保存されません。',
    },
    en: {
      waiting:['Watch the guides','Follow the animal pieces and wait for your next turn.'],
      0:['1. Hide and attack — animals','A and C are partners, as are B and D. Empty your hand to win. Hide a Horse, then attack with the Little squirrel.'],
      5:['2. Pass','You have no Little squirrel to receive with. Select Pass beside your hand.'],
      9:['3. Receive and counterattack','The opponent attacked with a Horse. Receive with your Horse.'],
      10:['3. Receive and counterattack','After receiving, you attack next. Select the red Mouse.'],
      14:['4. Continue your attack','Everyone passed, so you attack again. Hide a Sparrow and play the Winged dragon.'],
      19:['5. Go out','Receive the Dragon with your Dragon. Two pieces remain.'],
      20:['5. Go out','Play your last Flying squirrel to empty your hand and win.'],
      done:['Animal basics complete!','You practiced receiving, attacking, passing and going out. The King lesson continues with traditional pieces.'],
      detail:'The green border marks your next piece. All hidden pieces have the same back. No points or game records are saved.',
    },
    zh: {
      waiting:['观察引导员的行动','观察动物棋子，等待自己的下一回合。'],
      0:['1. 伏牌并进攻（动物）','A与C、B与D分别为队友。先出完手牌即可和牌。请伏下马，再用小松鼠进攻。'],
      5:['2. 跳过','没有小松鼠，无法接牌。请点击手牌旁的“跳过”。'],
      9:['3. 接牌并反攻','对手用马进攻。请选择自己的马接牌。'],
      10:['3. 接牌并反攻','接牌后由你进攻。请选择红色老鼠。'],
      14:['4. 继续进攻','所有人跳过后由你再次进攻。伏下麻雀，再用飞龙进攻。'],
      19:['5. 和牌','用自己的龙接住对方的龙。只剩两枚棋子。'],
      20:['5. 和牌','用最后的鼯鼠进攻，出完手牌即可和牌。'],
      done:['动物基础操作篇完成！','你已练习接牌、进攻、跳过与和牌。下一节王或玉的练习使用普通棋子。'],
      detail:'绿色边框标出下一枚棋子。所有伏牌使用相同背面。本练习不计分，也不保存棋谱。',
    },
  };
  function copy(state, language) {
    const text = COPY[lang(language)];
    const step = state.tutorial_completed ? text.done : state.turn !== 'A' ? text.waiting
      : text[Number(state.tutorial_action_index || 0)] || text[0];
    return [...step,text.detail];
  }
  window.goitaTutorialAnimals = Object.freeze({isActive,renderPiece,renderLegend,copy,PIECES});
})();
