const i = 0;
const s = "\uE000HTMLTAG0\uE001";
const cleanReplacer = "SUCCESS";
const regex = new RegExp(`\uE000HTMLTAG(?:<[^>]+>|[^\\uE001])*?${i}(?:<[^>]+>|[^\\uE001])*?\uE001`, 'g');
console.log("REGEX:", regex);
const result = s.replace(regex, cleanReplacer);
console.log(result);
