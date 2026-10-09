const katex = require('katex');

function heal(text) {
  let s = text;
  s = s.replace(/\\text\}\{\s*([^{}]+)\}/g, '\\text{$1}');
  return s;
}

const original = "\\sqrt{\\frac{2\\text}{ hf}}{\\text{m}}}";
const healed = heal(original);
console.log("Original:", original);
console.log("Healed:", healed);

try {
  console.log("Rendered:", katex.renderToString(healed, { throwOnError: true }));
} catch (e) {
  console.log("Error rendering:", e.message);
}
