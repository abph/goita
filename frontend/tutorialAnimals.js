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
      done:['どうぶつの基本操作編クリア！','色と動物で、受け・攻め・パス・上がりを練習できました。続いて、紫のやまねこ（王）とこねこ（玉）の使い方を確認しましょう。'],
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
      done:['Animal basics complete!','You practiced receiving, attacking, passing and going out. Next, learn how to use the purple Wildcat (King) and Kitten (Gyoku).'],
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
      done:['动物基础操作篇完成！','你已练习接牌、进攻、跳过与和牌。接下来学习紫色山猫（王）与小猫（玉）的用法。'],
      detail:'绿色边框标出下一枚棋子。所有伏牌使用相同背面。本练习不计分，也不保存棋谱。',
    },
  };
  const ROYAL_COPY = {
    ja: {
      waiting:['案内役の動きを見ましょう','次のやまねこ・こねこの練習場面へ進めています。'],
      3:['1. やまねこで受ける','ひりゅうは、やまねこで受けられます。同じ動物を持っていなくても大丈夫です。紫のやまねこを選びましょう。'],
      4:['1. 受けた後に攻める','やまねこで受けた後は、自分が攻めます。むささびを選びましょう。'],
      9:['2. すずめとねずみは受けられない','こねこややまねこでも、ねずみは受けられません。パスを選びましょう。'],
      13:['3. こねこで攻める準備','相手のこりすを、自分のこりすで受けましょう。受けた後は自分が攻めます。'],
      14:['3. こねこで攻める','この局では最初からやまねことこねこの両方を持っていました。攻めに使える条件を満たしているので、紫のこねこで攻めましょう。'],
      18:['3. 攻めを続ける','こねこが1巡しました。すずめを伏せ、3つ目の攻めにうまを出しましょう。'],
      22:['3. うまで上がる','うまが1巡しました。最後のすずめを伏せ、うまで上がりましょう。'],
      done:['やまねこ・こねこの練習クリア！','受けられる動物、受けられない動物、攻められる条件を練習し、うまで上がりました。この先の基本的な攻め方やAIとの対局は、通常の駒で行います。'],
      detail:'やまねこは王、こねこは玉です。どちらも同じ使い方です。この練習の得点と棋譜は保存されません。',
    },
    en: {
      waiting:['Watch the guides','Preparing the next Wildcat and Kitten practice position.'],
      3:['1. Receive with the Wildcat','The Wildcat can receive a Winged dragon even when you do not hold that animal. Select the purple Wildcat.'],
      4:['1. Attack after receiving','After receiving with the Wildcat, you attack. Select the Flying squirrel.'],
      9:['2. Sparrows and Mice cannot be received','A Kitten or Wildcat cannot receive a Mouse. Select Pass.'],
      13:['3. Prepare a Kitten attack','Receive the Little squirrel with your Little squirrel, then take the attack turn.'],
      14:['3. Attack with the Kitten','You were dealt both the Wildcat and Kitten, meeting a condition for attacking with them. Play the purple Kitten.'],
      18:['3. Continue attacking','Everyone passed the Kitten. Hide a Sparrow and play a Horse for your third attack.'],
      22:['3. Finish with a Horse','Everyone passed the Horse. Hide your last Sparrow and go out with the Horse.'],
      done:['Wildcat and Kitten practice complete!','You learned what they can receive and when they can attack, then went out with a Horse. Further attack lessons and AI games use traditional pieces.'],
      detail:'The Wildcat is King and the Kitten is Gyoku. Both work the same way. No points or game records are saved.',
    },
    zh: {
      waiting:['观察引导员的行动','正在准备山猫与小猫的下一个练习场景。'],
      3:['1. 用山猫接牌','即使没有同种动物，山猫也能接住飞龙。请选择紫色山猫。'],
      4:['1. 接牌后进攻','用山猫接牌后轮到你进攻。请选择鼯鼠。'],
      9:['2. 不能接麻雀与老鼠','即使用小猫或山猫，也不能接老鼠。请选择跳过。'],
      13:['3. 准备用小猫进攻','用自己的小松鼠接住对手的小松鼠，接牌后由你进攻。'],
      14:['3. 用小猫进攻','本局起手同时持有山猫与小猫，满足进攻条件。请打出紫色小猫。'],
      18:['3. 继续进攻','小猫已转回。请伏下麻雀，第三次进攻出马。'],
      22:['3. 用马和牌','马已转回。请伏下最后的麻雀，用马和牌。'],
      done:['山猫与小猫练习完成！','你已学习它们能接的动物、不能接的动物与进攻条件，并用马和牌。后续进攻练习及AI对局使用普通棋子。'],
      detail:'山猫对应王，小猫对应玉。两者用法相同。本练习不计分，也不会保存棋谱。',
    },
  };
  const ROYAL_EXPLANATION = {
    ja: {
      title:'やまねこ（こねこ）の使い方',
      sections:[
        ['やまねこ（こねこ）で受ける', ['うま・こりす・むささび・ひりゅう・りゅうを受けられます。','同じ動物を持っていなくても受けられます。','すずめとねずみは受けられません。']],
        ['攻めに使える条件', ['誰かがやまねこかこねこを受けに使った後。','最初からやまねことこねこの両方を持っていた場合。','その駒を出すことで上がれる場合。']],
        ['得点', ['やまねこかこねこで上がると50点です。','最後の2枚がやまねことこねこで、片方を伏せてもう片方で上がると100点です。']],
      ],
    },
    en: {
      title:'How to use the Wildcat and Kitten',
      sections:[
        ['Receiving', ['They can receive a Horse, Little squirrel, Flying squirrel, Winged dragon or Dragon.','You do not need to hold the matching animal.','They cannot receive Sparrows or Mice.']],
        ['When they may attack', ['After someone uses a Wildcat or Kitten to receive.','If you were dealt both the Wildcat and Kitten.','When playing one empties your hand.']],
        ['Points', ['Going out with the Wildcat or Kitten scores 50 points.','If your final two pieces are the Wildcat and Kitten, hiding one and going out with the other scores 100 points.']],
      ],
    },
    zh: {
      title:'山猫与小猫的用法',
      sections:[
        ['接牌', ['可以接马、小松鼠、鼯鼠、飞龙与龙。','没有同种动物也可以接牌。','不能接麻雀与老鼠。']],
        ['可以进攻的条件', ['有人用山猫或小猫接牌之后。','起手同时持有山猫与小猫时。','打出它即可和牌时。']],
        ['得分', ['用山猫或小猫和牌得50分。','最后两枚是山猫与小猫时，伏下一枚，用另一枚和牌得100分。']],
      ],
    },
  };
  function renderRoyalExplanation(container, language) {
    language = lang(language);
    if (container.dataset.language === language) return;
    container.dataset.language = language;
    container.setAttribute('data-i18n-ignore','');
    container.replaceChildren();
    const text = ROYAL_EXPLANATION[language];
    const title = document.createElement('h3');
    title.textContent = text.title;
    container.append(title);
    for (const [heading,items] of text.sections) {
      const sectionTitle = document.createElement('h4');
      sectionTitle.textContent = heading;
      const list = document.createElement('ul');
      for (const item of items) {
        const entry = document.createElement('li');
        entry.textContent = item;
        list.append(entry);
      }
      container.append(sectionTitle,list);
    }
  }
  function copy(state, language) {
    const royal = state.tutorial_chapter === 'royal';
    const text = (royal ? ROYAL_COPY : COPY)[lang(language)];
    const step = state.tutorial_completed ? text.done : state.turn !== 'A' ? text.waiting
      : text[Number(state.tutorial_action_index || 0)] || text[royal ? 3 : 0];
    return [...step,text.detail];
  }
  window.goitaTutorialAnimals = Object.freeze({isActive,renderPiece,renderLegend,renderRoyalExplanation,copy,PIECES});
})();
