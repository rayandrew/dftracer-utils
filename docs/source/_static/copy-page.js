// "Copy page" / "Download page" control for dftracer-utils docs. Each page's
// Markdown is written to a sibling <page>.md (see _inject_copy_page in conf.py).
// Copy fetches it from the same origin - so it works with a private source repo
// and behind a firewall, where an external agent could not reach it. Download is
// a native <a download>, so the reader hands the .md to any agent's file upload.
(function () {
  "use strict";

  function flash(root, msg) {
    var btn = root.querySelector(".dftu-copy-page-btn");
    if (!btn) return;
    var prev = btn.textContent;
    btn.textContent = msg;
    btn.disabled = true;
    setTimeout(function () {
      btn.textContent = prev;
      btn.disabled = false;
    }, 1600);
  }

  function copyPage(root) {
    var url = root.getAttribute("data-md");
    if (!url || !(navigator.clipboard && navigator.clipboard.writeText)) {
      flash(root, "Copy unavailable");
      return;
    }
    fetch(url)
      .then(function (r) {
        if (!r.ok) throw new Error(r.status);
        return r.text();
      })
      .then(function (text) {
        return navigator.clipboard.writeText(text);
      })
      .then(
        function () {
          flash(root, "Copied");
        },
        function () {
          flash(root, "Copy failed");
        }
      );
  }

  function setMenu(root, open) {
    var menu = root.querySelector(".dftu-copy-page-menu");
    var toggle = root.querySelector(".dftu-copy-page-toggle");
    if (!menu || !toggle) return;
    menu.hidden = !open;
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
  }

  function closeAllMenus(except) {
    document.querySelectorAll("[data-dftu-copy-page]").forEach(function (r) {
      if (r !== except) setMenu(r, false);
    });
  }

  document.addEventListener("click", function (e) {
    var root = e.target.closest("[data-dftu-copy-page]");
    if (!root) {
      closeAllMenus(null);
      return;
    }

    if (e.target.closest(".dftu-copy-page-toggle")) {
      var menu = root.querySelector(".dftu-copy-page-menu");
      var willOpen = menu ? menu.hidden : false;
      closeAllMenus(root);
      setMenu(root, willOpen);
      return;
    }

    var action = e.target.closest("[data-action]");
    if (action && action.getAttribute("data-action") === "copy") {
      setMenu(root, false);
      copyPage(root);
      return;
    }
    // Download is a native <a download>; just let it through and close the menu.
    setMenu(root, false);
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closeAllMenus(null);
  });
})();
