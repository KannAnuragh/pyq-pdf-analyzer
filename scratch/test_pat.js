const k = 'B';
const pat = new RegExp(`^\\s*(?:\\([\\s]*[${k}${k.toLowerCase()}][\\s]*\\)[\\.\\:\\-]?|\\[[\\s]*[${k}${k.toLowerCase()}][\\s]*\\][\\.\\:\\-]?|[${k}${k.toLowerCase()}][\\.\\)\\:\\-]\\s*|[${k}${k.toLowerCase()}](?=\\s+))\\s*`);
console.log('Regex:', pat);
const str = "Both Assertion and Reason are correct";
console.log('Match on Both:', pat.exec(str));
const replaced = str.replace(pat, '');
console.log('Replaced:', replaced);
