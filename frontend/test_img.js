const fs = require('fs');
const page = fs.readFileSync('src/app/page.tsx', 'utf8');
// Compile typescript manually? No, we can just use ts-node if installed.
