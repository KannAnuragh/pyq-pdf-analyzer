const stem = `a. Mooshakavamsa Kavya i. Hendrik Van Rheede
GBI @00 nomler 0 ol
b. Hortus Malabaricus ii. Luis de Camoes
Ga0085MY dEINOIESMS Iy’ eocaams
c. Tuhafat ul Mujahiddin iii. Athula
@nOa0@® gEIOIAD anger
d. The Lusiads iv. Wiliam Logan
qriwIge afleryo Genwad
e. Malabar Manual v. Shaikh Zainuddin`;

const res = stem.replace(/^([a-h]\.)\s+(.+?)\s+([ivx]+\.)\s+(.+)$/gmi, '<div style="display:flex; justify-content:space-between; margin-bottom:4px;"><div style="flex:1;">$1 $2</div><div style="flex:1;">$3 $4</div></div>');
console.log(res);
