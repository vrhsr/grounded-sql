(function () {
  "use strict";
  document.documentElement.classList.add("js");

  document.addEventListener("DOMContentLoaded", function () {
    /* ---------- Copy buttons ---------- */
    document.querySelectorAll(".copy-btn").forEach(function (b) {
      b.addEventListener("click", function () {
        var text = b.getAttribute("data-copy") || "";
        try {
          navigator.clipboard.writeText(text).then(function () {
            var orig = b.textContent;
            b.textContent = "Copied";
            window.setTimeout(function () { b.textContent = orig; }, 1400);
          }).catch(function () {});
        } catch (e) {}
      });
    });

    /* ---------- Scroll reveal (also cascades to dynamically-inserted bar fills) ---------- */
    function revealChildren(target) {
      target.querySelectorAll(".bar-fill").forEach(function (f) { f.classList.add("in"); });
    }
    try {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("in");
            revealChildren(entry.target);
            io.unobserve(entry.target);
          }
        });
      }, { threshold: 0.12 });
      document.querySelectorAll(".reveal").forEach(function (el) { io.observe(el); });
      window.__groundedSqlReobserve = function (el) { io.observe(el); };
    } catch (e) {
      document.querySelectorAll(".reveal").forEach(function (el) {
        el.classList.add("in");
        revealChildren(el);
      });
    }
  });
})();
