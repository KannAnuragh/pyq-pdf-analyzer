const katex = require('katex');

const str = "\\uE000HTMLTAG0\\uE001";
try {
    const rendered = katex.renderToString(str);
    console.log("RENDERED:", rendered);
} catch (e) {
    console.log("ERROR:", e);
}
