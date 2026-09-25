const str = "(b) Graphs showing variation of voltage and current with time;\n\u0002HTMLTAG0\u0003 (c) The reactance of device X";
const regex = /[\u0000-\u001F\uE000-\uE007]*HTMLTAG\d+[\u0000-\u001F\uE000-\uE007]*/gi;
const result = str.replace(regex, '');
console.log(JSON.stringify(result));
