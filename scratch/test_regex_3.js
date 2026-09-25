const s = "$\\frac{1}{2\\pi \\nu C}$ \\n(b) Graphs showing variation of voltage and current with time;\\n\uE000HTMLTAG0\uE001\\n(c) The reactance of device X is inversely proportional to its frequency i.e., X_C =\\n$";
const stripped = s.replace(/[\u0000-\u001F\uE000-\uE007]*HTMLTAG\d+[\u0000-\u001F\uE000-\uE007]*/gi, '');
console.log(JSON.stringify(stripped));
