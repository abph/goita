(() => {
  "use strict";
  const menus = [...document.querySelectorAll("[data-header-menu]")].map(root => {
    const toggle = root.querySelector("[data-header-menu-toggle]");
    const popup = document.getElementById(toggle.getAttribute("aria-controls"));
    return { root, toggle, popup };
  });

  function close(menu, restoreFocus = false) {
    menu.popup.hidden = true;
    menu.toggle.setAttribute("aria-expanded", "false");
    if (restoreFocus) menu.toggle.focus();
  }

  function items(menu) {
    return [...menu.popup.querySelectorAll("button")].filter(button => !button.disabled && button.getClientRects().length);
  }

  function open(menu) {
    menus.forEach(other => close(other));
    menu.popup.hidden = false;
    menu.toggle.setAttribute("aria-expanded", "true");
  }

  menus.forEach(menu => {
    menu.toggle.addEventListener("click", () => {
      if (menu.popup.hidden) open(menu);
      else close(menu);
    });
    menu.popup.addEventListener("click", event => {
      if (event.target.closest("button")) close(menu);
    });
    menu.root.addEventListener("keydown", event => {
      if (event.key === "Escape" && !menu.popup.hidden) {
        event.preventDefault();
        event.stopPropagation();
        close(menu, true);
      } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        if (menu.popup.hidden) open(menu);
        const buttons = items(menu);
        const index = buttons.indexOf(document.activeElement);
        const next = index < 0 ? (event.key === "ArrowDown" ? 0 : buttons.length - 1)
          : (index + (event.key === "ArrowDown" ? 1 : -1) + buttons.length) % buttons.length;
        buttons[next]?.focus();
      }
    });
  });
  document.addEventListener("click", event => {
    menus.forEach(menu => { if (!menu.root.contains(event.target)) close(menu); });
  });
  document.addEventListener("focusin", event => {
    menus.forEach(menu => { if (!menu.root.contains(event.target)) close(menu); });
  });
})();
