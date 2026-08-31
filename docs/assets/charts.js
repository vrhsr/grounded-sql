(function () {
  "use strict";

  var SYSTEMS = [
    { label: "A · base", desc: "no tuning, no retrieval", val: 53.3, best: false },
    { label: "B · base + RAG", desc: "retrieval only", val: 54.2, best: false },
    { label: "C · fine-tuned", desc: "QLoRA r=64", val: 70.2, best: true },
    { label: "D · fine-tuned + RAG", desc: "QLoRA r=64 + retrieval", val: 66.4, best: false }
  ];

  var ERRORS = [
    { label: "Correct SQL, wrong result", pct: 36.0, color: "#8991A0" },
    { label: "Syntax error", pct: 31.8, color: "#C2C8D1" },
    { label: "Wrong JOIN logic", pct: 26.6, color: "#93630D" },
    { label: "Wrong aggregation", pct: 5.5, color: "#14655F" }
  ];

  document.addEventListener("DOMContentLoaded", function () {
    var barsEl = document.getElementById("bars");
    if (barsEl) {
      SYSTEMS.forEach(function (s) {
        var row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML =
          '<div class="bar-label">' + s.label + '<span class="sys-desc">' + s.desc + "</span></div>" +
          '<div class="bar-track"><div class="bar-fill' + (s.best ? " best" : "") + '" style="--w:' + s.val + '%"></div></div>' +
          '<div class="bar-val">' + s.val.toFixed(1) + "%</div>";
        barsEl.appendChild(row);
      });
    }

    var stackEl = document.getElementById("error-stack");
    var legendEl = document.getElementById("error-legend");
    if (stackEl && legendEl) {
      ERRORS.forEach(function (e) {
        var seg = document.createElement("div");
        seg.className = "stack-seg";
        seg.style.width = e.pct + "%";
        seg.style.background = e.color;
        stackEl.appendChild(seg);
        var item = document.createElement("span");
        item.innerHTML = '<span class="sw" style="background:' + e.color + '"></span>' + e.label + " (" + e.pct + "%)";
        legendEl.appendChild(item);
      });
    }
  });
})();
