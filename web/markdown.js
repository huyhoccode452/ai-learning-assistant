/* Tiny, safe Markdown renderer for the AI Learning Assistant web UI (see docs/web_ui_design.md, section 3.9).

   window.renderMarkdown(text, { onCitation }) -> DocumentFragment

   Safety: the output is built ONLY with createElement / createTextNode / textContent / appendChild.
   No element other than p, br, ul, ol, li, h4, pre, code, strong, em is created by this file, and the only
   attribute ever set is the constant class name. Nothing from the text becomes a tag, attribute or URL.
   Performance: block parsing is one pass over the lines; inline parsing is a hand-written scanner.
   Failed closer searches are remembered (per marker kind and per container end), so the cost stays linear.

   Documented choices for ambiguous input:
   - "***x***" -> strong > em > "x" (the three stars are one marker). Unclosed "***x" stays literal.
   - "**" inside italic, "*" inside italic and "**" inside bold are literal text (no deeper nesting).
   - Blank lines between list items keep one list when the next non-blank line is again a list item.
   - A non-list line indented by 2+ spaces (or a tab) right after a list item continues that item (br + text).
   - Numbered lists never set a start attribute: the browser numbers the items 1, 2, 3.
   - Each paragraph / item line is trimmed at both ends; tabs and spaces inside a line are kept.
   - A fence line is ``` plus an optional info word (ignored); a line like ```x``` is plain text. */
(function () {
  "use strict";

  var MAX_NUMBER_DIGITS = 3;

  // ---------------------------------------------------------------- DOM helpers
  function makeEl(tag, cls) {
    var node = document.createElement(tag);
    node.className = cls;
    return node;
  }

  function isSpaceCode(code) {
    return code === 32 || code === 9 || code === 10 || code === 13 || code === 160;
  }

  function isDigitCode(code) {
    return code >= 48 && code <= 57;
  }

  // Characters the inline scanner must look at: newline, "*", "[", backslash, backtick.
  function isSpecialCode(code) {
    return code === 10 || code === 42 || code === 91 || code === 92 || code === 96;
  }

  // Characters a backslash can escape: \ ` * [ ]
  function isEscapableCode(code) {
    return code === 92 || code === 96 || code === 42 || code === 91 || code === 93;
  }

  // ---------------------------------------------------------------- inline scanner
  function newContext(text, cite) {
    return {
      s: text,
      cite: cite,
      lastBacktick: text.lastIndexOf("`"),
      // fail[n] maps a container end to the smallest start from which a closer of length n was not found.
      fail: { 1: new Map(), 2: new Map(), 3: new Map() },
    };
  }

  // Position of the closing backtick of a code span opened at `open`, or -1 (also -1 for an empty span).
  function findCodeEnd(c, open, end) {
    if (open >= c.lastBacktick) return -1;
    var q = c.s.indexOf("`", open + 1);
    if (q === -1 || q >= end || q === open + 1) return -1;
    return q;
  }

  // Linear search for a closing marker made of `need` stars. Returns the index of the first star of the
  // closer, or -1. A closer must be preceded by a non-space character. Escapes and code spans are skipped.
  // For need 1 a run of exactly one star counts; for need 2 or 3 a run of at least `need` stars counts
  // and the LAST `need` stars of the run close it (so "**a *b***" closes the italic and the bold).
  function scanCloser(c, from, end, need) {
    var s = c.s;
    var k = from;
    while (k < end) {
      var code = s.charCodeAt(k);
      if (code === 92) {
        k += k + 1 < end && isEscapableCode(s.charCodeAt(k + 1)) ? 2 : 1;
      } else if (code === 96) {
        var q = findCodeEnd(c, k, end);
        k = q > 0 ? q + 1 : k + 1;
      } else if (code === 42) {
        var r = k;
        while (r < end && s.charCodeAt(r) === 42) r++;
        var len = r - k;
        if (k > from && !isSpaceCode(s.charCodeAt(k - 1)) && (need === 1 ? len === 1 : len >= need)) {
          return r - need;
        }
        k = r;
      } else {
        k++;
      }
    }
    return -1;
  }

  function findCloser(c, from, end, need) {
    var memo = c.fail[need];
    var failedFrom = memo.get(end);
    if (failedFrom !== undefined && failedFrom <= from) return -1;
    var pos = scanCloser(c, from, end, need);
    if (pos < 0 && (failedFrom === undefined || from < failedFrom)) memo.set(end, from);
    return pos;
  }

  // Ask the host for the citation node. Anything but a DOM Node (or a throw) means "keep the literal text".
  function makeCitation(c, number) {
    if (!c.cite) return null;
    var result;
    try {
      result = c.cite(number);
    } catch (err) {
      if (typeof console !== "undefined" && console.warn) console.warn("onCitation threw:", err);
      return null;
    }
    return result instanceof Node ? result : null;
  }

  // Parse s[start, end) into `parent`. noBold / noItalic say which emphasis kinds are not allowed here.
  function parseInline(c, parent, start, end, noBold, noItalic) {
    var s = c.s;
    var buf = "";
    var i = start;

    function flush() {
      if (buf !== "") {
        parent.appendChild(document.createTextNode(buf));
        buf = "";
      }
    }

    function wrap(tag, cls, from, to, nb, ni) {
      flush();
      var node = makeEl(tag, cls);
      parseInline(c, node, from, to, nb, ni);
      parent.appendChild(node);
    }

    while (i < end) {
      var code = s.charCodeAt(i);

      if (!isSpecialCode(code)) {
        var j = i + 1;
        while (j < end && !isSpecialCode(s.charCodeAt(j))) j++;
        buf += s.slice(i, j);
        i = j;
        continue;
      }

      if (code === 10) {
        flush();
        parent.appendChild(makeEl("br", "md-br"));
        i++;
      } else if (code === 92) {
        if (i + 1 < end && isEscapableCode(s.charCodeAt(i + 1))) {
          buf += s.charAt(i + 1);
          i += 2;
        } else {
          buf += "\\";
          i++;
        }
      } else if (code === 96) {
        var q = findCodeEnd(c, i, end);
        if (q > 0) {
          flush();
          var codeEl = makeEl("code", "md-code");
          codeEl.textContent = s.slice(i + 1, q);
          parent.appendChild(codeEl);
          i = q + 1;
        } else {
          buf += "`";
          i++;
        }
      } else if (code === 91) {
        var d = i + 1;
        while (d < end && d - i - 1 < MAX_NUMBER_DIGITS + 1 && isDigitCode(s.charCodeAt(d))) d++;
        var digits = d - i - 1;
        if (digits >= 1 && digits <= MAX_NUMBER_DIGITS && d < end && s.charCodeAt(d) === 93) {
          var cited = makeCitation(c, parseInt(s.slice(i + 1, d), 10));
          if (cited) {
            flush();
            parent.appendChild(cited);
          } else {
            buf += s.slice(i, d + 1);
          }
          i = d + 1;
        } else {
          buf += "[";
          i++;
        }
      } else {
        // code === 42, a run of stars. Count at most 3 so a long run costs O(1) per step.
        var run = 1;
        while (run < 3 && i + run < end && s.charCodeAt(i + run) === 42) run++;

        if (run >= 2 && noBold) {
          // No bold here: the stars are plain text.
          buf += run === 3 ? "***" : "**";
          i += run;
        } else if (run === 3) {
          var cp3 = i + 3 < end && !isSpaceCode(s.charCodeAt(i + 3)) ? findCloser(c, i + 3, end, 3) : -1;
          if (cp3 >= 0) {
            flush();
            var strong3 = makeEl("strong", "md-strong");
            var em3 = makeEl("em", "md-em");
            parseInline(c, em3, i + 3, cp3, true, true);
            strong3.appendChild(em3);
            parent.appendChild(strong3);
            i = cp3 + 3;
          } else {
            buf += "*"; // retry from the next star, which now starts a run of 2
            i++;
          }
        } else if (run === 2) {
          var cp2 = i + 2 < end && !isSpaceCode(s.charCodeAt(i + 2)) ? findCloser(c, i + 2, end, 2) : -1;
          if (cp2 >= 0) {
            wrap("strong", "md-strong", i + 2, cp2, true, noItalic);
            i = cp2 + 2;
          } else {
            buf += "*";
            i++;
          }
        } else {
          var cp1 = !noItalic && i + 1 < end && !isSpaceCode(s.charCodeAt(i + 1)) ? findCloser(c, i + 1, end, 1) : -1;
          if (cp1 >= 0) {
            wrap("em", "md-em", i + 1, cp1, true, true);
            i = cp1 + 1;
          } else {
            buf += "*";
            i++;
          }
        }
      }
    }
    flush();
  }

  function renderInline(parent, text, cite) {
    if (text === "") return;
    parseInline(newContext(text, cite), parent, 0, text.length, false, false);
  }

  // ---------------------------------------------------------------- block helpers
  function isBlank(line) {
    for (var k = 0; k < line.length; k++) {
      if (!isSpaceCode(line.charCodeAt(k))) return false;
    }
    return true;
  }

  // Leading indentation width: a space counts 1, a tab counts 4.
  function indentWidth(line) {
    var width = 0;
    for (var k = 0; k < line.length; k++) {
      var ch = line.charAt(k);
      if (ch === " ") width += 1;
      else if (ch === "\t") width += 4;
      else break;
    }
    return width;
  }

  // List item line -> { type: "ul"|"ol", indent, text } or null. Markers: "- ", "* ", "• ", "1. ", "1) ".
  function parseListItem(line) {
    var j = 0;
    while (j < line.length && (line.charAt(j) === " " || line.charAt(j) === "\t")) j++;
    var ch = line.charAt(j);
    var type = null;
    var textStart = 0;
    if ((ch === "-" || ch === "*" || ch === "•") && line.charAt(j + 1) === " ") {
      type = "ul";
      textStart = j + 2;
    } else if (ch >= "0" && ch <= "9") {
      var k = j;
      while (k < line.length && k - j < 9 && isDigitCode(line.charCodeAt(k))) k++;
      var after = line.charAt(k);
      if ((after === "." || after === ")") && line.charAt(k + 1) === " ") {
        type = "ol";
        textStart = k + 2;
      }
    }
    if (!type) return null;
    var text = line.slice(textStart).trim();
    if (text === "") return null;
    return { type: type, indent: indentWidth(line), text: text };
  }

  function parseHeading(line) {
    var h = 0;
    while (h < 7 && line.charAt(h) === "#") h++;
    if (h < 1 || h > 6 || line.charAt(h) !== " ") return null;
    var text = line.slice(h + 1).trim();
    return text === "" ? null : text;
  }

  function isFenceOpen(line) {
    var t = line.trim();
    var j = 0;
    while (j < t.length && t.charAt(j) === "`") j++;
    return j >= 3 && t.indexOf("`", j) === -1;
  }

  function isFenceClose(line) {
    var t = line.trim();
    if (t.length < 3) return false;
    for (var k = 0; k < t.length; k++) {
      if (t.charAt(k) !== "`") return false;
    }
    return true;
  }

  // ---------------------------------------------------------------- public renderer
  function renderMarkdown(text, opts) {
    var frag = document.createDocumentFragment();
    if (typeof text !== "string" || text === "") return frag;
    var handler = opts && typeof opts.onCitation === "function" ? opts.onCitation : null;
    var cite = handler
      ? function (n) {
          return handler.call(opts, n);
        }
      : null;

    var lines = text.split(/\r\n|\r|\n/);
    var count = lines.length;
    var para = [];
    var list = null; // { type, base, el, top, nested, afterBlank }
    var pending = null; // { li, text }: the last list item, rendered when it can no longer grow

    function flushPending() {
      if (pending) {
        renderInline(pending.li, pending.text, cite);
        pending = null;
      }
    }

    function endList() {
      flushPending();
      list = null;
    }

    function flushPara() {
      if (para.length === 0) return;
      var p = makeEl("p", "md-p");
      renderInline(p, para.join("\n"), cite);
      frag.appendChild(p);
      para = [];
    }

    function nextNonBlankIsListItem(from) {
      var k = from;
      while (k < count && isBlank(lines[k])) k++;
      return k < count && parseListItem(lines[k]) !== null;
    }

    function startList(item, li) {
      list = { type: item.type, base: item.indent, el: makeEl(item.type, "md-" + item.type), top: li, nested: null, afterBlank: false };
      frag.appendChild(list.el);
      list.el.appendChild(li);
    }

    function addListItem(item) {
      flushPending();
      var li = makeEl("li", "md-li");
      if (list === null) {
        startList(item, li);
      } else if (item.indent >= list.base + 2) {
        // One nested level; anything deeper is flattened into it.
        if (list.nested === null || list.nested.type !== item.type) {
          list.nested = { type: item.type, el: makeEl(item.type, "md-" + item.type) };
          list.top.appendChild(list.nested.el);
        }
        list.nested.el.appendChild(li);
      } else if (item.type !== list.type) {
        startList(item, li);
      } else {
        list.el.appendChild(li);
        list.top = li;
        list.nested = null;
      }
      list.afterBlank = false;
      pending = { li: li, text: item.text };
    }

    var i = 0;
    while (i < count) {
      var line = lines[i];

      if (isBlank(line)) {
        flushPara();
        if (list !== null) {
          if (nextNonBlankIsListItem(i + 1)) {
            flushPending();
            list.afterBlank = true;
          } else {
            endList();
          }
        }
        i++;
        continue;
      }

      if (isFenceOpen(line)) {
        flushPara();
        endList();
        var body = [];
        i++;
        while (i < count && !isFenceClose(lines[i])) {
          body.push(lines[i]);
          i++;
        }
        i++; // skip the closing fence (or run past the end when it is missing)
        var pre = makeEl("pre", "md-pre");
        var codeEl = makeEl("code", "md-code");
        codeEl.textContent = body.join("\n");
        pre.appendChild(codeEl);
        frag.appendChild(pre);
        continue;
      }

      var headingText = parseHeading(line);
      if (headingText !== null) {
        flushPara();
        endList();
        var heading = makeEl("h4", "md-h");
        renderInline(heading, headingText, cite);
        frag.appendChild(heading);
        i++;
        continue;
      }

      var item = parseListItem(line);
      if (item !== null) {
        flushPara();
        addListItem(item);
        i++;
        continue;
      }

      if (list !== null && !list.afterBlank && pending !== null && indentWidth(line) >= 2) {
        pending.text += "\n" + line.trim();
      } else {
        endList();
        para.push(line.trim());
      }
      i++;
    }

    flushPara();
    endList();
    return frag;
  }

  window.renderMarkdown = renderMarkdown;
})();
