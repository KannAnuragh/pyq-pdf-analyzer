// Test KaTeX rendering of physics questions
const currentRegex = /(?<=^|[\s\(\[\{=+\-*/,~])([a-zA-Z0-9\)])\^([0-9a-zA-Z\-]+|\{[^}]+\})(?=$|[\s\)\]},.;:!?=+\-*/~])/g;

// Enhanced regexes
const sciRegex = /(?<=^|[\s\(\[\{=+\-*/,~])(?:\d+(?:\.\d+)?\s*(?:\\times|×)\s*)?([a-zA-Z0-9]+|\))\^([0-9a-zA-Z\-]+|\{[^}]+\})(?=$|[\s\)\]},.;:!?=+\-*/~])/g;

const samples = [
  "Two parallel large thin metal sheets have equal surface densities $26.4 \\times 10^{-12}$ C/m^2 of opposite signs.",
  "$1.5 \\times 10^{-16}$ N/C",
  "$3 \\times 10^{-10}$ N/C",
  "3 N/C",
  "3 \\text{ N/C}"
];

for (const s of samples) {
  let cleaned = s.replace(/\\text\{\s*([^{}]+?)\s*\}/g, '$1');
  console.log("Original:", s);
  console.log("Cleaned :", cleaned);
}
