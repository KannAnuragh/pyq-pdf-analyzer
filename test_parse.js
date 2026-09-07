const stem = `a. Mooshakavamsa Kavya i. Hendrik Van Rheede
GBI @00 nomler 0 ol
b. Hortus Malabaricus ii. Luis de Camoes
Ga0085MY dEINOIESMS Iy’ eocaams
c. Tuhafat ul Mujahiddin iii. Athula
@nOa0@® gEIOIAD anger
d. The Lusiads iv. Wiliam Logan
qriwIge afleryo Genwad
e. Malabar Manual v. Shaikh Zainuddin`;

const lines = stem.split('\n');
for (const line of lines) {
    const m = line.match(/^([a-h]\.)\s+(.+?)\s+([ivx]+\.)\s+(.+)$/i);
    if (m) {
        console.log(`Col1: ${m[1]} ${m[2]} | Col2: ${m[3]} ${m[4]}`);
    } else {
        console.log(`Normal: ${line}`);
    }
}
