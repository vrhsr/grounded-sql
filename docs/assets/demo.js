(function () {
  "use strict";

  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- Data: real schemas + real questions + real executed rows, pulled straight
     from the actual Spider .sqlite files (dataset/spider/database/<db>/<db>.sqlite) ---------- */
  var DATABASES = {
    concert_singer: {
      file: "concert_singer.sqlite",
      label: "concert_singer",
      schema: [
        { name: "stadium", cols: [
          ["Stadium_ID", "PK", "number"], ["Location", "", "text"], ["Name", "", "text"],
          ["Capacity", "", "number"], ["Highest", "", "number"], ["Lowest", "", "number"], ["Average", "", "number"]
        ]},
        { name: "singer", cols: [
          ["Singer_ID", "PK", "number"], ["Name", "", "text"], ["Country", "", "text"],
          ["Song_Name", "", "text"], ["Song_release_year", "", "text"], ["Age", "", "number"], ["Is_male", "", "bool"]
        ]},
        { name: "concert", cols: [
          ["concert_ID", "PK", "number"], ["concert_Name", "", "text"], ["Theme", "", "text"],
          ["Stadium_ID", "FK", "→ stadium"], ["Year", "", "text"]
        ]},
        { name: "singer_in_concert", cols: [
          ["concert_ID", "FK", "→ concert"], ["Singer_ID", "FK", "→ singer"]
        ]}
      ],
      examples: [
        {
          question: "What is the total number of singers?",
          tag: "simple",
          tables: ["singer"],
          sql: "SELECT count(*) FROM singer",
          cols: ["count(*)"],
          rows: [[6]]
        },
        {
          question: "Show the stadium name and the number of concerts in each stadium.",
          tag: "join",
          tables: ["concert", "stadium"],
          sql: "SELECT T2.name, count(*)\nFROM concert AS T1\nJOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id\nGROUP BY T1.stadium_id",
          cols: ["Name", "count(*)"],
          rows: [["Stark's Park", 1], ["Glebe Park", 1], ["Somerset Park", 2], ["Recreation Park", 1], ["Balmoor", 1]]
        },
        {
          question: "Show all countries and the number of singers in each country.",
          tag: "group by",
          tables: ["singer"],
          sql: "SELECT country, count(*)\nFROM singer\nGROUP BY country",
          cols: ["Country", "count(*)"],
          rows: [["France", 4], ["Netherlands", 1], ["United States", 1]]
        },
        {
          question: "List all song names by singers above the average age.",
          tag: "nested",
          tables: ["singer"],
          sql: "SELECT song_name\nFROM singer\nWHERE age > (SELECT avg(age) FROM singer)",
          cols: ["Song_Name"],
          rows: [["You"], ["Sun"], ["Gentleman"]]
        },
        {
          question: "What is the name of the stadium with the highest capacity?",
          tag: "order by",
          tables: ["stadium"],
          sql: "SELECT name\nFROM stadium\nORDER BY capacity DESC\nLIMIT 1",
          cols: ["Name"],
          rows: [["Celtic Park"]]
        }
      ]
    },
    pets_1: {
      file: "pets_1.sqlite",
      label: "pets_1",
      schema: [
        { name: "student", cols: [
          ["StuID", "PK", "number"], ["LName", "", "text"], ["Fname", "", "text"], ["Age", "", "number"],
          ["Sex", "", "text"], ["Major", "", "number"], ["Advisor", "", "number"], ["city_code", "", "text"]
        ]},
        { name: "has_pet", cols: [
          ["StuID", "FK", "→ student"], ["PetID", "FK", "→ pets"]
        ]},
        { name: "pets", cols: [
          ["PetID", "PK", "number"], ["PetType", "", "text"], ["pet_age", "", "number"], ["weight", "", "number"]
        ]}
      ],
      examples: [
        {
          question: "Find the number of pets whose weight is heavier than 10.",
          tag: "simple",
          tables: ["pets"],
          sql: "SELECT count(*) FROM pets\nWHERE weight > 10",
          cols: ["count(*)"],
          rows: [[2]]
        },
        {
          question: "Find the first name of students who have a cat or dog pet.",
          tag: "join",
          tables: ["student", "has_pet", "pets"],
          sql: "SELECT DISTINCT T1.Fname\nFROM student AS T1\nJOIN has_pet AS T2 ON T1.stuid = T2.stuid\nJOIN pets AS T3 ON T3.petid = T2.petid\nWHERE T3.pettype = 'cat' OR T3.pettype = 'dog'",
          cols: ["Fname"],
          rows: [["Linda"], ["Tracy"]]
        },
        {
          question: "Find the maximum weight for each type of pet. List the maximum weight and pet type.",
          tag: "group by",
          tables: ["pets"],
          sql: "SELECT max(weight), petType\nFROM pets\nGROUP BY petType",
          cols: ["max(weight)", "PetType"],
          rows: [[12.0, "cat"], [13.4, "dog"]]
        },
        {
          question: "Find the id of students who do not have a cat pet.",
          tag: "nested",
          tables: ["student", "has_pet", "pets"],
          sql: "SELECT stuid FROM student\nEXCEPT\nSELECT T1.stuid FROM student AS T1\nJOIN has_pet AS T2 ON T1.stuid = T2.stuid\nJOIN pets AS T3 ON T3.petid = T2.petid\nWHERE T3.pettype = 'cat'",
          cols: ["StuID"],
          rows: [[1002], [1003], [1004], [1005], [1006], [1007]],
          moreCount: 27
        },
        {
          question: "Find the average age of students who do not have any pet.",
          tag: "nested",
          tables: ["student", "has_pet"],
          sql: "SELECT avg(age) FROM student\nWHERE stuid NOT IN (\n  SELECT T1.stuid FROM student AS T1\n  JOIN has_pet AS T2 ON T1.stuid = T2.stuid\n)",
          cols: ["avg(age)"],
          rows: [[19.8]]
        }
      ]
    }
  };

  var DB_ORDER = ["concert_singer", "pets_1"];
  var activeDb = DB_ORDER[0];

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function highlightSql(sql) {
    var esc = escapeHtml(sql);
    var pattern = /\b(SELECT|DISTINCT|FROM|WHERE|JOIN|AS|ON|GROUP BY|ORDER BY|LIMIT|DESC|ASC|EXCEPT|NOT IN|count|avg|max)\b/gi;
    return esc.replace(pattern, function (m) { return '<span class="kw">' + m + "</span>"; });
  }

  document.addEventListener("DOMContentLoaded", function () {
    var schemaGrid = document.getElementById("schema-grid");
    var dbTabsEl = document.getElementById("db-tabs");
    var chipRow = document.getElementById("chip-row");
    var input = document.getElementById("question-input");
    var btn = document.getElementById("generate-btn");
    var statusLine = document.getElementById("status-line");
    var fallbackNote = document.getElementById("fallback-note");
    var retrievedEl = document.getElementById("retrieved-tables");
    var sqlBox = document.getElementById("sql-box");
    var execCaption = document.getElementById("exec-caption");
    var resultArea = document.getElementById("result-area");
    if (!schemaGrid || !input || !btn) return;

    var busy = false;
    var chips = [];

    /* ---------- Schema cards: render every table, from both databases ---------- */
    function renderSchema() {
      schemaGrid.innerHTML = "";
      DB_ORDER.forEach(function (dbKey) {
        DATABASES[dbKey].schema.forEach(function (t) {
          var card = document.createElement("div");
          card.className = "schema-card";
          var head = document.createElement("div");
          head.className = "t-name";
          var nameSpan = document.createElement("span");
          nameSpan.textContent = t.name;
          var dbSpan = document.createElement("span");
          dbSpan.className = "t-db";
          dbSpan.textContent = DATABASES[dbKey].label;
          head.appendChild(nameSpan);
          head.appendChild(dbSpan);
          card.appendChild(head);
          var ul = document.createElement("ul");
          t.cols.forEach(function (c) {
            var li = document.createElement("li");
            var left = document.createElement("span");
            left.textContent = c[0];
            var right = document.createElement("span");
            if (c[1] === "PK") right.innerHTML = '<span class="col-flag">PK</span>';
            else if (c[1] === "FK") right.innerHTML = '<span class="col-flag fk">FK ' + escapeHtml(c[2]) + "</span>";
            else right.innerHTML = '<span class="col-type">' + escapeHtml(c[2]) + "</span>";
            li.appendChild(left);
            li.appendChild(right);
            ul.appendChild(li);
          });
          card.appendChild(ul);
          schemaGrid.appendChild(card);
        });
      });
    }

    /* ---------- DB tabs ---------- */
    function renderDbTabs() {
      if (!dbTabsEl) return;
      dbTabsEl.innerHTML = "";
      DB_ORDER.forEach(function (key) {
        var tab = document.createElement("button");
        tab.type = "button";
        tab.className = "db-tab" + (key === activeDb ? " active" : "");
        tab.textContent = DATABASES[key].label;
        tab.addEventListener("click", function () { switchDb(key); });
        dbTabsEl.appendChild(tab);
      });
    }

    /* ---------- Question chips for the active db ---------- */
    function renderChips() {
      chipRow.innerHTML = "";
      chips = [];
      DATABASES[activeDb].examples.forEach(function (ex, i) {
        var chip = document.createElement("button");
        chip.type = "button";
        chip.className = "chip";
        chip.textContent = "Q" + (i + 1) + " · " + ex.tag;
        chip.addEventListener("click", function () {
          input.value = ex.question;
          runExample(ex);
        });
        chipRow.appendChild(chip);
        chips.push(chip);
      });
    }

    function setActiveChip(ex) {
      var idx = DATABASES[activeDb].examples.indexOf(ex);
      chips.forEach(function (c, i) { c.classList.toggle("active", i === idx); });
    }

    function findExample(text) {
      var norm = text.trim().toLowerCase();
      var examples = DATABASES[activeDb].examples;
      for (var i = 0; i < examples.length; i++) {
        if (examples[i].question.toLowerCase() === norm) return examples[i];
      }
      return null;
    }

    function renderRetrieved(tables) {
      retrievedEl.innerHTML = "";
      tables.forEach(function (t) {
        var s = document.createElement("span");
        s.className = "table-chip";
        s.textContent = t;
        retrievedEl.appendChild(s);
      });
    }

    function renderResult(ex) {
      var wrap = document.createElement("div");
      wrap.className = "result-wrap";
      var table = document.createElement("table");
      table.className = "result-table";
      var thead = document.createElement("thead");
      var htr = document.createElement("tr");
      ex.cols.forEach(function (c) {
        var th = document.createElement("th");
        th.textContent = c;
        htr.appendChild(th);
      });
      thead.appendChild(htr);
      table.appendChild(thead);
      var tbody = document.createElement("tbody");
      ex.rows.forEach(function (r) {
        var tr = document.createElement("tr");
        r.forEach(function (v) {
          var td = document.createElement("td");
          td.textContent = v;
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      wrap.appendChild(table);
      if (ex.moreCount) {
        var more = document.createElement("div");
        more.className = "more-rows-note";
        more.textContent = "+ " + ex.moreCount + " more rows (truncated for display)";
        wrap.appendChild(more);
      }
      resultArea.innerHTML = "";
      resultArea.appendChild(wrap);
      var total = ex.rows.length + (ex.moreCount || 0);
      var badge = document.createElement("span");
      badge.className = "row-badge";
      badge.textContent = total + (total === 1 ? " row returned" : " rows returned");
      resultArea.appendChild(badge);
    }

    function typeSql(sql, done) {
      var highlighted = highlightSql(sql);
      if (reduceMotion) {
        sqlBox.innerHTML = highlighted;
        done();
        return;
      }
      sqlBox.innerHTML = "";
      var i = 0;
      var speed = Math.max(6, Math.floor(600 / sql.length));
      function step() {
        i += 2;
        var slice = sql.slice(0, i);
        sqlBox.innerHTML = highlightSql(slice) + '<span class="cursor"></span>';
        if (i < sql.length) {
          window.setTimeout(step, speed);
        } else {
          sqlBox.innerHTML = highlighted;
          done();
        }
      }
      step();
    }

    function runExample(ex) {
      if (busy) return;
      busy = true;
      fallbackNote.style.display = "none";
      setActiveChip(ex);
      btn.disabled = true;
      sqlBox.innerHTML = "";
      execCaption.textContent = "";
      resultArea.innerHTML = '<p class="placeholder-note">running…</p>';
      renderRetrieved([]);

      var delay1 = reduceMotion ? 0 : 260;
      var delay2 = reduceMotion ? 0 : 320;

      statusLine.textContent = "→ retrieving relevant tables…";
      window.setTimeout(function () {
        renderRetrieved(ex.tables);
        statusLine.textContent = "→ generating SQL with fine-tuned Mistral-7B…";
        window.setTimeout(function () {
          typeSql(ex.sql, function () {
            var file = DATABASES[activeDb].file;
            statusLine.textContent = "→ executing against " + file + "…";
            execCaption.textContent = "Executed on the real " + file + " database";
            renderResult(ex);
            statusLine.textContent = "✓ done";
            btn.disabled = false;
            busy = false;
          });
        }, delay2);
      }, delay1);
    }

    function switchDb(key) {
      if (key === activeDb || busy) return;
      activeDb = key;
      renderDbTabs();
      renderChips();
      var first = DATABASES[activeDb].examples[0];
      input.value = first.question;
      runExample(first);
    }

    btn.addEventListener("click", function () {
      var match = findExample(input.value);
      if (match) runExample(match);
      else fallbackNote.style.display = "block";
    });

    renderSchema();
    renderDbTabs();
    renderChips();
    window.setTimeout(function () { runExample(DATABASES[activeDb].examples[0]); }, reduceMotion ? 0 : 500);
  });
})();
