const katex = require('katex');

function test(text) {
  let s = text;
  
  // 5. Process raw square roots: \sqrt{...}, \sqrt[n]{...}
  s = s.replace(/\\sqrt(?:\[[^\]]+\])?(\{(?:[^{}]|\{[^{}]*\})*\})/g, (match) => {
    try {
      const rendered = katex.renderToString(match, { displayMode: false, throwOnError: false, strict: false });
      if (rendered.includes('katex-error')) {
        console.log("KaTeX Error on match:", match);
        return match;
      }
      return rendered;
    } catch (e) {
      console.log("Exception:", e);
      return match;
    }
  });

  return s;
}

console.log(test("v_{max} = \\sqrt{\\frac{2\\text{ hf}}{m}}"));
console.log(test("v_{max} = \\sqrt{\\frac{2\\text}{ hf}}{\\text{m}}"));

