/* Pure helpers (no React, no browser APIs) so they can be tested in Node. */

export const dataUrl = (data, mime) => `data:${mime};base64,${data}`;

/** Reports and tables contain box-drawing rules; those must not be re-flowed. */
export const isTable = (text = "") => /─{3,}/.test(text);

/* ------------------------------------------------------------------ */
/* Speech text (only used when the browser has to speak by itself)     */
/* ------------------------------------------------------------------ */

export const speakable = (t = "") => {
  let s = String(t)
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/```[\s\S]*$/g, " ")
    .replace(/[\u2500-\u257F]/g, "")
    .replace(/\p{Extended_Pictographic}/gu, "")
    .replace(/\[(\d+)\]/g, "")
    .replace(/https?:\/\/\S+/g, "a link")
    .replace(/[*_`#>]/g, "")
    .replace(/[•◈▲▼]/g, "")
    .replace(/(\d)\s?%/g, "$1 percent")
    .replace(/(\d+(?:\.\d+)?)\s?K\b/g, "$1 thousand")
    .replace(/(\d+(?:\.\d+)?)\s?M\b/g, "$1 million")
    .replace(/\s+/g, " ")
    .trim();
  if (s.length > 360) {
    s = s.slice(0, 360);
    const cut = Math.max(s.lastIndexOf(". "), s.lastIndexOf("! "), s.lastIndexOf("? "));
    if (cut > 120) s = s.slice(0, cut + 1);
  }
  return s;
};

/* ------------------------------------------------------------------ */
/* Wake word                                                           */
/* ------------------------------------------------------------------ */

// speech recognition often mishears the name, so accept the common variants
const WAKE =
  /\b(?:(?:hey|ok|okay|hi|hello|yo)[ ,]+)?(jarvis|jarvas|jarvice|jervis|jarves|jarvus|jarviz|jarvish|jaarvis|jarwis|javis|jar vis|travis|charvis|darvis|yarvis|harvis|jarbis|jarvi|jarvist|jeruvis)\b[ ,.!:;-]*/i;
const DEV_NAME = /(?:हे\s*)?(जार्विस|जारविस|जार्वेस|जारविस|जार्विस्|ज़ार्विस|जार्वीश|जर्विस|जार्विश|जरविस|जार्विज़|जार्वीस)/g;

// Whisper writes what it thinks it heard: "Jarvis" can come out as Travis, Gharvis, Jarbus, Service, ज़ार्विस ...
// So besides the exact spellings above, one of the first three words that sounds close enough also counts.
function editDistance(a, b) {
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array(b.length).fill(0)]);
  for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++)
    for (let j = 1; j <= b.length; j++)
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
  return d[a.length][b.length];
}
const DEV_SOUNDS = /^[जझगचज़ग़]\S*[रर्ऱ]\S*[विवबब़]/; // ज़ार्विस, जरविस, गार्विस, झार्वीस ...
const SKIP = new Set(["hey", "hi", "ok", "okay", "hello", "yo", "ए", "हे", "अरे", "ओके"]);

/** Find a "Jarvis"-like word among the first three words. Returns {index, length} in the text, or null. */
function fuzzyWake(text) {
  const re = /[\p{L}\p{M}]+/gu;
  let m;
  let seen = 0;
  while ((m = re.exec(text)) && seen < 3) {
    const w = m[0].toLowerCase();
    if (SKIP.has(w)) continue;
    seen++;
    const near = /^[a-z]+$/.test(w) && w.length >= 5 && w.length <= 8 && w !== "service" && editDistance(w, "jarvis") <= 2 && /[rv]/.test(w);
    if (near || (/[ऀ-ॿ]/.test(w) && DEV_SOUNDS.test(w))) return { index: m.index, length: m[0].length };
  }
  return null;
}

/** true when the name is in what was heard (any spelling the recogniser uses) */
export const hasWake = (transcript = "") => {
  const t = transcript.replace(DEV_NAME, "jarvis");
  return WAKE.test(t) || !!fuzzyWake(t);
};

/**
 * @param transcript  what the recogniser heard
 * @param awake       true while we are inside the follow-up window after an answer
 * @returns {heard: boolean, command: string}  heard=false means ignore the utterance
 */
export function parseWake(transcript, awake = false) {
  // Hindi recognition writes the name in Devanagari
  const text = (transcript || "").replace(DEV_NAME, "jarvis").trim();
  if (!text) return { heard: false, command: "" };
  let m = WAKE.exec(text);
  if (!m) {
    const f = fuzzyWake(text);
    if (f) m = { index: f.index, 0: text.slice(f.index, f.index + f.length) };
  }
  if (!m) return awake ? { heard: true, command: text } : { heard: false, command: "" };
  const before = text.slice(0, m.index).replace(/[ ,.!:;-]+$/g, "").trim();
  const after = text.slice(m.index + m[0].length).replace(/^[ ,.!:;-]+/, "").trim();
  const command = [before, after].filter(Boolean).join(" ").trim();
  return { heard: true, command };
}

/* ------------------------------------------------------------------ */
/* Markdown (small, safe subset; no HTML is ever injected)             */
/* ------------------------------------------------------------------ */

const INLINE =
  /(`[^`\n]+`)|(\*\*[^*\n]+\*\*)|(__[^_\n]+__)|(\[[^\]\n]+\]\(https?:\/\/[^)\s]+\))|(\[\d+(?:\s*,\s*\d+)*\])|(\*[^*\s][^*\n]*\*)|(https?:\/\/[^\s<>)]+)/g;

export function parseInline(text = "") {
  const out = [];
  let last = 0;
  let m;
  INLINE.lastIndex = 0;
  while ((m = INLINE.exec(text)) !== null) {
    if (m.index > last) out.push({ t: "text", v: text.slice(last, m.index) });
    const tok = m[0];
    if (m[1]) out.push({ t: "code", v: tok.slice(1, -1) });
    else if (m[2] || m[3]) out.push({ t: "bold", v: tok.slice(2, -2) });
    else if (m[4]) {
      const mm = /^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/.exec(tok);
      out.push({ t: "link", v: mm[1], href: mm[2] });
    } else if (m[5]) {
      out.push({ t: "cite", v: tok, n: tok.slice(1, -1).split(",").map((x) => parseInt(x, 10)) });
    } else if (m[6]) out.push({ t: "italic", v: tok.slice(1, -1) });
    else if (m[7]) out.push({ t: "link", v: tok.replace(/[.,;:!?]+$/, ""), href: tok.replace(/[.,;:!?]+$/, "") });
    last = m.index + tok.length;
    if (m[7] && /[.,;:!?]+$/.test(tok)) last -= tok.length - tok.replace(/[.,;:!?]+$/, "").length;
  }
  if (last < text.length) out.push({ t: "text", v: text.slice(last) });
  return out;
}

/** Split text into blocks: paragraphs, lists, headings, quotes and fenced code (also unfinished, while streaming). */
export function parseMarkdown(text = "") {
  const src = String(text).replace(/\r\n/g, "\n");
  const lines = src.split("\n");
  const blocks = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    const fence = /^\s*```\s*([\w+#.-]*)\s*$/.exec(line);
    if (fence) {
      const code = [];
      i++;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) code.push(lines[i++]);
      const closed = i < lines.length;
      i++; // skip closing fence
      blocks.push({ type: "code", lang: fence[1] || "", code: code.join("\n"), open: !closed });
      continue;
    }

    if (!line.trim()) {
      i++;
      continue;
    }

    // | a | b |  with a |---|---| separator line under the header
    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(lines[i + 1])) {
      const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const head = cells(line).map(parseInline);
      const align = cells(lines[i + 1]).map((c) => (/^:-*:$/.test(c) ? "center" : /-:$/.test(c) ? "right" : "left"));
      i += 2;
      const rows = [];
      while (i < lines.length && /^\s*\|.*\|?\s*$/.test(lines[i]) && lines[i].trim()) rows.push(cells(lines[i++]).map(parseInline));
      blocks.push({ type: "table", head, rows, align });
      continue;
    }

    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
      blocks.push({ type: "hr" });
      i++;
      continue;
    }

    const h = /^\s{0,3}(#{1,4})\s+(.*)$/.exec(line);
    if (h) {
      blocks.push({ type: "h", level: h[1].length, inline: parseInline(h[2]) });
      i++;
      continue;
    }

    if (/^\s*[-*•]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*•]\s+/.test(lines[i])) {
        items.push(parseInline(lines[i].replace(/^\s*[-*•]\s+/, "")));
        i++;
      }
      blocks.push({ type: "ul", items });
      continue;
    }

    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
        items.push(parseInline(lines[i].replace(/^\s*\d+[.)]\s+/, "")));
        i++;
      }
      blocks.push({ type: "ol", items });
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      const q = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) q.push(lines[i++].replace(/^\s*>\s?/, ""));
      blocks.push({ type: "quote", inline: parseInline(q.join(" ")) });
      continue;
    }

    const para = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^\s*```/.test(lines[i]) &&
      !/^\s{0,3}#{1,4}\s/.test(lines[i]) &&
      !/^\s*[-*•]\s+/.test(lines[i]) &&
      !/^\s*\d+[.)]\s+/.test(lines[i]) &&
      !/^\s*>\s?/.test(lines[i]) &&
      !/^\s*\|.*\|\s*$/.test(lines[i])
    ) {
      para.push(lines[i++]);
    }
    blocks.push({ type: "p", inline: parseInline(para.join("\n")) });
  }
  return blocks;
}
