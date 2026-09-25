let s = "time; \x02HTMLTAG0\x03 (c) The reactance";
let s2 = s.replace(/[\u0000-\u001F\uE000-\uE007]*HTMLTAG\d+[\u0000-\u001F\uE000-\uE007]*/gi, '');
console.log(s2);
