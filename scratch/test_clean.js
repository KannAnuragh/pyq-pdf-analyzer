function cleanOptionDisplay(text, optionKey) {
  if (!text) return '';
  let s = text.trim();
  if (optionKey) {
    const k = optionKey.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const pat = new RegExp(`^\\s*(?:\\([\\s]*[${k}${k.toLowerCase()}][\\s]*\\)[\\.\\:\\-]?|\\[[\\s]*[${k}${k.toLowerCase()}][\\s]*\\][\\.\\:\\-]?|[${k}${k.toLowerCase()}][\\.\\)\\:\\-]\\s*|[${k}${k.toLowerCase()}](?=\\s+))\\s*`);
    s = s.replace(pat, '').trim();
  }
  if (/^oth\b/i.test(s) && (!optionKey || optionKey.toUpperCase() === 'B')) {
    s = 'B' + s;
  }
  return s;
}
console.log('Result 1:', cleanOptionDisplay('oth Assertion and Reason', 'B'));
console.log('Result 2:', cleanOptionDisplay('B. oth Assertion and Reason', 'B'));
console.log('Result 3:', cleanOptionDisplay('(b) Both Assertion and Reason', 'B'));
console.log('Result 4:', cleanOptionDisplay('Both Assertion and Reason', 'B'));
