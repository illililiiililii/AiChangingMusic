/* Lightweight, reusable section-tagging for ACE-Step lyrics. */
(function (root) {
  function lyricLineSimilarity(a, b) {
    const left = [...a].filter((c) => /[가-힣a-z0-9]/i.test(c));
    const right = [...b].filter((c) => /[가-힣a-z0-9]/i.test(c));
    if (!left.length || !right.length) return 0;

    let row = Array.from({ length: right.length + 1 }, (_, i) => i);
    for (let i = 1; i <= left.length; i++) {
      const next = [i];
      for (let j = 1; j <= right.length; j++) {
        next[j] = Math.min(
          next[j - 1] + 1,
          row[j] + 1,
          row[j - 1] + (left[i - 1] === right[j - 1] ? 0 : 1),
        );
      }
      row = next;
    }
    return 1 - row[right.length] / Math.max(left.length, right.length);
  }

  function compareLyricText(expected, recognized) {
    const left = [...expected].filter((c) => /[가-힣a-z0-9]/i.test(c));
    const right = [...recognized].filter((c) => /[가-힣a-z0-9]/i.test(c));
    if (!left.length || !right.length) return 0;
    let row = Array.from({ length: right.length + 1 }, (_, i) => i);
    for (let i = 1; i <= left.length; i++) {
      const next = [i];
      for (let j = 1; j <= right.length; j++) {
        next[j] = Math.min(
          next[j - 1] + 1,
          row[j] + 1,
          row[j - 1] + (left[i - 1] === right[j - 1] ? 0 : 1),
        );
      }
      row = next;
    }
    return Math.max(0, 1 - row[right.length] / Math.max(left.length, right.length));
  }

  function structureLyricsForAI(raw, enabled = true) {
    const lines = raw.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    if (
      !enabled ||
      lines.length < 6 ||
      lines.some((line) => /^\[(verse|chorus|bridge|pre-chorus|intro|outro)/i.test(line))
    ) return raw;

    let best = null;
    for (let i = 1; i < lines.length - 2; i++) {
      for (let j = i + 3; j <= lines.length - 3; j++) {
        let similaritySum = 0;
        for (let length = 1; length <= Math.min(8, lines.length - j); length++) {
          similaritySum += lyricLineSimilarity(lines[i + length - 1], lines[j + length - 1]);
          if (length < 3) continue;
          const average = similaritySum / length;
          const score = length * average * average;
          if (average >= 0.72 && (!best || score > best.score)) {
            best = { first: i, repeat: j, length, score };
          }
        }
      }
    }

    const tags = new Map([[0, '[Verse 1]']]);
    if (best) {
      tags.set(best.first, '[Chorus]');
      const secondVerse = best.first + best.length;
      if (secondVerse < best.repeat) tags.set(secondVerse, '[Verse 2]');
      tags.set(best.repeat, '[Chorus]');
      const bridge = best.repeat + best.length;
      if (bridge < lines.length) {
        tags.set(bridge, '[Bridge]');
        const finalChorus = bridge + Math.min(2, Math.max(1, Math.floor((lines.length - bridge) / 3)));
        if (finalChorus < lines.length) tags.set(finalChorus, '[Final Chorus]');
      }
    } else if (lines.length >= 10) {
      tags.set(Math.floor(lines.length * 0.48), '[Verse 2]');
      tags.set(Math.floor(lines.length * 0.72), '[Chorus]');
    }

    return lines.map((line, index) => [tags.get(index), line].filter(Boolean).join('\n')).join('\n');
  }

  root.structureLyricsForAI = structureLyricsForAI;
  root.compareLyricText = compareLyricText;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { lyricLineSimilarity, structureLyricsForAI, compareLyricText };
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
