'use client';

import { useState, useEffect } from 'react';
import { 
  UploadCloud, FileText, CheckCircle2, AlertTriangle, Trash2, ExternalLink, 
  RefreshCw, Type, X, Sparkles, ArrowLeft, Eye, Check, Clock, Cpu, ShieldCheck, 
  ChevronRight, BarChart3, HelpCircle, Layers, Sliders, Settings, Save, CheckCircle,
  ScanLine, Database, Bot, Zap
} from 'lucide-react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

const API_BASE = 'http://localhost:8000';

const KNOWN_KATEX_COMMANDS = new Set([
  'frac', 'dfrac', 'tfrac', 'sqrt', 'times', 'div', 'pm', 'mp', 'cdot', 'circ', 'degree',
  'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa',
  'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi',
  'omega', 'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon', 'Phi', 'Psi',
  'Omega', 'neq', 'ne', 'approx', 'leq', 'geq', 'le', 'ge', 'infty', 'sum', 'prod', 'int',
  'log', 'ln', 'sin', 'cos', 'tan', 'in', 'partial', 'nabla', 'vec', 'hat',
  'begin', 'end', 'left', 'right', 'bmatrix', 'pmatrix', 'vmatrix', 'Vmatrix', 'Bmatrix', 'matrix', 'array',
  'sim', 'bar', 'tilde', 'hbar', 'rightarrow', 'to', 'leftrightharpoons', 'rightleftharpoons', 'leftrightarrow',
  'text', 'mathrm', 'mathbf', 'quad', 'qquad', 'equiv', 'propto', 'perp', 'parallel', 'cup', 'cap', 'subset', 'subseteq'
]);

function sanitizeMathForKatex(math: string): string {
  if (!math) return '';
  let m = math.trim();

  // Vector arrow in fraction numerator misrecognition
  // e.g. \frac{=>}{F}, \frac{>}{F}, \frac{ - }{F}, \frac{=d}{F}, \frac{=}{F} -> \vec{F}
  m = m.replace(/\\frac\{[=><\-—~^d2\s|!_]+\}\{([A-Za-z])(?:\s*[A-Za-z]*)\}/g, '\\vec{$1}');

  // Radical misrecognitions: \X, \x, VX, Vx -> \sqrt{x}
  m = m.replace(/\\X\b/g, '\\sqrt{x}');
  m = m.replace(/\bVX\b/g, '\\sqrt{x}');
  m = m.replace(/\bVx\b/g, '\\sqrt{x}');
  m = m.replace(/\bV(\d+)\b/g, '\\sqrt{$1}');
  m = m.replace(/\\x\b/g, 'x');

  // Normalize Unicode multiplication sign × to \times
  m = m.replace(/×/g, '\\times ');
  // Heal any corrupted x^3 after number/power to \times (e.g. 2^5 x^3 -> 2^5 \times 3)
  m = m.replace(/(\d+(?:\^\d+)?)\s*[xX]\^([0-9])\b/g, '$1 \\times $2');
  // Heal any number x number to \times (e.g. 2^2 x 101 -> 2^2 \times 101, 96 x 404 -> 96 \times 404)
  m = m.replace(/(\d+(?:\.\d+)?|\^\{?\d+\}?)\s*[xX]\s*(\d+(?:\.\d+)?)/g, '$1 \\times $2');
  // Separate adjacent equations that lost comma: "3 404 =" -> "3, 404 ="
  m = m.replace(/(\d+)\s+(?=\d+\s*=)/g, '$1, ');
  // Standalone \times at start of equation: \times = -> x =
  m = m.replace(/^\\times\s*=/g, 'x =');

  // Vector expressions: fix (= + y) -> (x + y) and (y = x) -> (y - x)
  m = m.replace(/\(\s*=\s*\+\s*([a-zA-Z0-9])\s*\)/g, '(x + $1)');
  m = m.replace(/\(\s*([a-zA-Z])\s*=\s*([a-zA-Z0-9])\s*\)/g, '($1 - $2)');

  // Unit vectors: i, j, k (or J, ]) after parentheses or coefficients
  m = m.replace(/\)\s*[iI]\b/g, ')\\hat{i}');
  m = m.replace(/\)\s*[jJ\]]\b/g, ')\\hat{j}');
  m = m.replace(/\b[zZ]\s*2\s*[kK]\b/g, 'z^{2}\\hat{k}');
  m = m.replace(/\b[zZ]\^\{?2\}?\s*[kK]\b/g, 'z^{2}\\hat{k}');
  m = m.replace(/(?<=\+)\s*[kK]\b/g, ' \\hat{k}');

  // Angle/radian misrecognitions
  m = m.replace(/\\frac\{(?:iy|n|tt|IT|5|p)\}\{([2346])\}/g, '\\frac{\\pi}{$1}');
  m = m.replace(/\b0\s*=\s*\\frac\{\\pi\}/g, '\\theta = \\frac{\\pi}');

  // Statistical distributions: X ~ N(theta, sigma^2)
  m = m.replace(/([A-Za-z])\s*[=~—–-]{1,2}\s*[\(=]?\s*[0-9oOnN\(\)\s=]*[,\s=]+[0oO©\s=]*\b2\s*\)/g, '$1 \\sim N(\\theta, \\sigma^{2})');
  m = m.replace(/([A-Za-z])\s*[~—–-]\s*([nN])\s*\(/g, '$1 \\sim N(');
  m = m.replace(/\(\s*[90oOθ]\s*,/g, '(\\theta, ');
  m = m.replace(/\(\s*μ\s*,/g, '(\\mu, ');
  m = m.replace(/([,\(])\s*[0oO©]\s*[\^]?\s*2\s*([,\)])/g, '$1 \\sigma^{2}$2');

  // Standalone sigma^2 and variance: ONLY copyright symbol © / \u00A9 or explicit caret 0^2 / O^2
  m = m.replace(/[\u00A9©]\s*(?:\^\{?2\}?|2)/g, '\\sigma^{2}');
  m = m.replace(/(?<![A-Za-z0-9])[0oO]\s*\^\s*\{?2\}?(?![0-9])/g, '\\sigma^{2}');

  // Sum of squared deviations / estimators: sum (x_i - \bar{x})^2
  m = m.replace(/(?:\\frac\{id\}\{1\}|\\frac\{a\}\{1\}|\bhn\b)?\s*\(\s*x\s*(?:\\hat\{i\}|[I\]i1])?\s*[-–—=]\s*(?:x[X\s]*|x)\s*\)\s*2\b/g, '\\sum (x_i - \\bar{x})^{2}');
  m = m.replace(/\b\\frac\{1\}\{7\}\b/g, '\\frac{1}{n}');
  m = m.replace(/\\begin\{bmatrix\}\s*1vV\s*\\\\\s*1-1\s*&\s*hn\s*\\end\{bmatrix\}/g, '\\frac{1}{n-1}');
  m = m.replace(/\b1-1\b/g, 'n-1');
  m = m.replace(/\bxX\b/g, '\\bar{x}');

  // Chemistry subscripts & formulas
  m = m.replace(/\bH\s*2\s*O\b/g, 'H_2O');
  m = m.replace(/\bC\s*O\s*2\b/g, 'CO_2');
  m = m.replace(/\bH\s*2\s*S\s*O\s*4\b/g, 'H_2SO_4');
  m = m.replace(/\bK\s*Mn\s*O\s*4\b/g, 'KMnO_4');
  m = m.replace(/\bCa\s*C\s*O\s*3\b/g, 'CaCO_3');

  // Physics formulas
  m = m.replace(/\b[eE]\s*=\s*m\s*c\s*[\^]?\s*2\b/g, 'E = mc^{2}');
  m = m.replace(/\b[vV]\s*[\^]?\s*2\s*=\s*[uU]\s*[\^]?\s*2\s*\+\s*2\s*a\s*s\b/g, 'v^{2} = u^{2} + 2as');
  m = m.replace(/\b[vV]\s*[\^]?\s*2\s*=\s*[uU]\s*[\^]?\s*2\b/g, 'v^{2} = u^{2}');
  m = m.replace(/\b[hH]\s*\\?nu\b/g, 'h\\nu');

  // Matrix healing: restore missing backslashes for begin/end/left/right
  m = m.replace(/(?<!\\)\b(begin|end|left|right)\b/g, '\\$1');
  m = m.replace(/\\left\[\s*\\begin\{array\}\{[a-z]+\}\s*([\s\S]+?)\s*\\end\{array\}\s*\\right\]/g, '\\begin{bmatrix} $1 \\end{bmatrix}');

  // Clean unknown single-letter backslash commands: e.g. \Y -> Y, \A -> A
  m = m.replace(/\\([a-zA-Z]+)/g, (full, cmd) => {
    if (KNOWN_KATEX_COMMANDS.has(cmd)) return full;
    return cmd;
  });

  return m;
}

function formatTokenCost(usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number; estimated_cost_usd?: number }): string {
  if (!usage) return '$0.00';
  if (typeof usage.estimated_cost_usd === 'number' && usage.estimated_cost_usd > 0) {
    if (usage.estimated_cost_usd < 0.0001) return '<$0.0001';
    return `$${usage.estimated_cost_usd.toFixed(4)}`;
  }
  const p = usage.prompt_tokens || 0;
  const c = usage.completion_tokens || 0;
  const est = (p * 0.59 + c * 0.79) / 1000000;
  if (est === 0) return '$0.00';
  if (est < 0.0001) return '<$0.0001';
  return `$${est.toFixed(4)}`;
}

function formatQuestionStem(text: string): string {
  if (!text) return '';
  let s = text.replace(/\u00a0/g, ' ');

  // Remove redundant leading Question prefix if duplicated in stem (e.g. "Q1: ", "Question 1:- ", "1. ")
  s = s.replace(/^(?:Question\s*|Q\.?\s*|ചോദ്യം\s*)\d+\s*(?::\-?|[\.\):-])\s*/i, '');
  s = s.replace(/^\d+\s*[\.\):-]\s+/, '');

  // Ensure Roman numerals like (i), (ii), (iii), (iv), (v) or i), ii) have a newline before them
  s = s.replace(/(?<=[^\n])\s+(\([ivxlcdmIVXLCDM]+\))\s*/g, '\n$1 ');
  s = s.replace(/(?<=[?.!:;])\s*([ivxlcdmIVXLCDM]+\))\s*/g, '\n$1 ');

  // Ensure numbered sub-statements like (1), (2), (3) or (a), (b), (c) after sentence endings have newlines
  s = s.replace(/(?<=[?.!:;])\s*(\([1-9a-dA-D]\))\s*/g, '\n$1 ');

  // Trim extra blank lines (maximum 2 consecutive newlines)
  s = s.replace(/\n{3,}/g, '\n\n');

  return s.trim();
}

function renderLatex(text: string): string {
  if (!text) return '';

  // 0. Protect HTML tags (inline equation images & diagrams) FIRST before any text/math alterations
  const htmlTags: string[] = [];
  let s = text.replace(/<div[^>]*>[\s\S]*?<\/div>|<img[^>]*\/?>/gi, (match) => {
    const id = `%%HTMLTAG_${htmlTags.length}%%`;
    htmlTags.push(match);
    return id;
  });

  // Revert accidental 20 \sigma^2 or 2 \sigma^2 6 back to 2002, 2026, 2025
  s = s.replace(/\b20\s*\$?\\sigma\^?2\$?\b/g, '2002');
  s = s.replace(/\b2\s*\$?\\sigma\^?2\$?\s*([0-9])\b/g, '20$1');

  // Pre-clean known OCR math distortions in text
  s = s.replace(/\\frac\{[=><\-—~^d2\s|!_]+\}\{([A-Za-z])(?:\s*[A-Za-z]*)\}/g, '\\vec{$1}');
  s = s.replace(/\\X\b/g, '\\sqrt{x}');
  s = s.replace(/\bVX\b/g, '\\sqrt{x}');
  s = s.replace(/\bVx\b/g, '\\sqrt{x}');
  s = s.replace(/\\x\b/g, 'x');
  // Normalize Unicode multiplication sign × to \times and heal multiplication
  s = s.replace(/×/g, '\\times ');
  s = s.replace(/(\d+(?:\^\d+)?)\s*[xX]\^([0-9])\b/g, '$1 \\times $2');
  s = s.replace(/(\d+(?:\.\d+)?|\^\{?\d+\}?)\s*[xX]\s*(\d+(?:\.\d+)?)/g, '$1 \\times $2');
  s = s.replace(/(\d+)\s+(?=\d+\s*=)/g, '$1, ');
  s = s.replace(/\(\s*=\s*\+\s*([a-zA-Z0-9])\s*\)/g, '(x + $1)');
  s = s.replace(/\(\s*([a-zA-Z])\s*=\s*([a-zA-Z0-9])\s*\)/g, '($1 - $2)');
  s = s.replace(/\)\s*[iI]\b/g, ')\\hat{i}');
  s = s.replace(/\)\s*[jJ\]]\b/g, ')\\hat{j}');
  s = s.replace(/\b[zZ]\s*2\s*[kK]\b/g, 'z^{2}\\hat{k}');

  // Mathematical Differential Equations & Derivatives
  s = s.replace(/\$?V\s*aydyay\$?/gi, '$\\frac{dy}{dx} + \\frac{1}{x}y = x^2$');
  s = s.replace(/\b(?:ou|0u|eu)\b(?=\s*=)/g, '\\frac{dy}{dx}');
  s = s.replace(/(?<!\\frac\{)\bdy(?:\s*[\n/]|\\n)\s*dx\b/g, '\\frac{dy}{dx}');
  s = s.replace(/(?<!\\frac\{)\bdx(?:\s*[\n/]|\\n)\s*dy\b/g, '\\frac{dx}{dy}');
  s = s.replace(/(?<!\\frac\{)\bd\^?2y(?:\s*[\n/]|\\n)\s*dx\^?2\b/g, '\\frac{d^2y}{dx^2}');
  s = s.replace(/(?<!\\frac\{)\bdy(?:\s*[\n/]|\\n)\s*dt\b/g, '\\frac{dy}{dt}');
  s = s.replace(/(?<!\\frac\{)\bdx(?:\s*[\n/]|\\n)\s*dt\b/g, '\\frac{dx}{dt}');

  // Mathematical Exponents, Fractions & Functions
  s = s.replace(/\$?x\\ne\$?/g, '$e^x$');
  s = s.replace(/(?<![A-Za-z0-9])1\s*\/\s*x(?![A-Za-z0-9])/g, '$\\frac{1}{x}$');
  s = s.replace(/(?<![A-Za-z0-9])1\s*\/\s*y(?![A-Za-z0-9])/g, '$\\frac{1}{y}$');
  s = s.replace(/(?<![A-Za-z0-9])1\s*\/\s*n(?![A-Za-z0-9])/g, '$\\frac{1}{n}$');
  s = s.replace(/\b2(?:\s*[\n/]|\\n)\s*X[xX]?\b/g, 'x^2');
  s = s.replace(/\$2(?:\s*[\n/]|\\n)\s*[xX]\$?/g, '$x^2$');
  s = s.replace(/\bln\s*\(\s*x\s*\)/g, '$\\ln(x)$');
  s = s.replace(/\blog\s*\(\s*([a-zA-Z0-9]+)\s*\)/g, '$\\log($1)$');

  // Trigonometric Identities & Angle Symbols
  s = s.replace(/tan[’'\"]+\s*[@0θ]?/g, '\\tan^{2}\\theta');
  s = s.replace(/sec[’'\"]+\s*[@0θ]?/g, '\\sec^{2}\\theta');
  s = s.replace(/sin[’'\"]+\s*[@0θ]?/g, '\\sin^{2}\\theta');
  s = s.replace(/cos[’'\"]+\s*[@0θ]?/g, '\\cos^{2}\\theta');
  s = s.replace(/(?<=\b(?:sin|cos|tan|sec|csc|cot)\s*)[@0]\b/g, '\\theta');

  // Sets, Combinatorics & Quadratic
  s = s.replace(/\bIAU\s*BI\b/g, 'n(A \\cup B)');
  s = s.replace(/\bA\s*∩\s*B\b/g, 'A \\cap B');
  s = s.replace(/\^{?([0-9]+)}?C([0-9]+)/g, '^{$1}C_{$2}');
  s = s.replace(/e\^\{?x\}?\s*Ty/g, 'e^{x+y}');

  // Integrals & Vector Theorems
  s = s.replace(/\[\s*f\s*F\s*sas\s*=\s*\[\s*o>\s*F\s*\)\s*dV/g, '$\\iint_S \\vec{F} \\cdot d\\vec{S} = \\iiint_V (\\nabla \\cdot \\vec{F}) dV$');
  s = s.replace(/\[\s*f\s*F\s*sas\s*=\s*\[\s*fv\s*F\s*dV/g, '$\\iint_S \\vec{F} \\cdot d\\vec{S} = \\iiint_V (\\nabla \\cdot \\vec{F}) dV$');
  s = s.replace(/\b(?:vVxF|vx\s*F)\s*=\s*0\b/g, '$\\nabla \\times \\vec{F} = \\vec{0}$');
  s = s.replace(/\b(?:V\.F|v\.F)\s*=\s*0\b/g, '$\\nabla \\cdot \\vec{F} = 0$');
  s = s.replace(/\|\s*Jt\s*dx\s*=\s*1\b/g, '$\\int f(x) dx = 1$');
  s = s.replace(/\[\s*te\s*dx\s*>\s*1\b/g, '$\\int f(x) dx > 1$');
  s = s.replace(/lax\s*ol\s*[-–—]\s*\(@\.b\)\s*=\s*1dhibP/g, '$|\\vec{a} \\times \\vec{b}|^{2} + (\\vec{a} \\cdot \\vec{b})^{2} = |\\vec{a}|^{2} |\\vec{b}|^{2}$');
  s = s.replace(/lax\s*or\s*\+\s*\(@\.b\)\s*=\s*I@hbP/g, '$|\\vec{a} \\times \\vec{b}|^{2} + (\\vec{a} \\cdot \\vec{b})^{2} = |\\vec{a}|^{2} |\\vec{b}|^{2}$');
  s = s.replace(/Vy\)\s*<\s*V\(ye\)/g, '$V(\\bar{y}_{st}) < V(\\bar{y}_{sys})$');
  s = s.replace(/\b2\s*a\b(?=\s*-\s*ടെസ്റ്റ്)/g, '$\\chi^{2}$');
  s = s.replace(/\bX\s*~\s*y\s*\(2\)/g, '$X \\sim \\chi^{2}(2)$');

  // Statistical & scientific equation distortions in text (e.g. Q97)
  s = s.replace(/\$?\s*[xX]\s*=\s*=\s*\(\s*0\s*\)\s*=\s*2\s*\)\s*\$?|\$?\s*[xX]\s*[=~—–-]{1,2}\s*[\(=]?\s*[0-9oOnN\(\)\s=]*[,\s=]+[0oO©\s=]*\b2\s*\)\s*\$?/, '$X \\sim N(\\theta, \\sigma^2)$');
  // Match ONLY copyright symbol © / \u00A9 for OCR-misread sigma^2, NEVER plain 0 or O!
  s = s.replace(/\$?[\u00A9©]\s*(?:\^\{?2\}?|2)\$?/g, '$\\sigma^2$');
  s = s.replace(/\\frac\{1\}\{7\}/g, '\\frac{1}{n}');
  s = s.replace(/\\frac\{id\}\{1\}/g, '\\sum');
  s = s.replace(/\\begin\{bmatrix\}\s*1vV\s*\\\\\s*1-1\s*&\s*hn\s*\\end\{bmatrix\}/g, '\\frac{1}{n-1} \\sum');
  s = s.replace(/\(\s*x\s*(?:\\hat\{i\}|[I\]i1])?\s*[-–—=]\s*(?:x[X\s]*|x)\s*\)\s*2/g, '(x_i - \\bar{x})^2');
  s = s.replace(/\$\s*\$/g, ' ');

  // Chemistry & physics text formulas
  s = s.replace(/\bH\s*2\s*O\b/g, '$H_2O$');
  s = s.replace(/\bC\s*O\s*2\b/g, '$CO_2$');
  s = s.replace(/\bH\s*2\s*S\s*O\s*4\b/g, '$H_2SO_4$');
  s = s.replace(/\bK\s*Mn\s*O\s*4\b/g, '$KMnO_4$');
  s = s.replace(/\bCa\s*C\s*O\s*3\b/g, '$CaCO_3$');
  s = s.replace(/\b[eE]\s*=\s*m\s*c\s*[\^]?\s*2\b/g, '$E = mc^2$');
  s = s.replace(/\b[vV]\s*[\^]?\s*2\s*=\s*[uU]\s*[\^]?\s*2\s*\+\s*2\s*a\s*s\b/g, '$v^2 = u^2 + 2as$');
  s = s.replace(/\b[hH]\s*\\?nu\b/g, '$h\\nu$');

  // 0.1 Protect fill-in-the-blanks (3 or more underscores) so they don't break LaTeX parsing
  const blanks: string[] = [];
  s = s.replace(/_{3,}/g, (match) => {
    const id = `%%BLANK_${blanks.length}%%`;
    blanks.push(match);
    return id;
  });

  // Matrix / array pre-cleaning: normalize left[begin{array}... or begin{bmatrix}
  s = s.replace(/\$?\s*(?:\\left\[|left\[)?\s*(?:\\begin|begin)\{array\}\{[a-z]+\}\s*([\s\S]+?)\s*(?:\\end|end)\{array\}\s*(?:\\right\]|right\])?\s*\$?/g, (_, inner) => {
    const fixedRows = inner.trim().replace(/(?<!\\)\\(?!\\)/g, '\\\\');
    return `$\\begin{bmatrix} ${fixedRows} \\end{bmatrix}$`;
  });
  s = s.replace(/\$?\s*(?:\\begin|begin)\{(bmatrix|pmatrix|vmatrix|Vmatrix|matrix)\}\s*([\s\S]+?)\s*(?:\\end|end)\{\1\}\s*\$?/g, (_, env, inner) => {
    const fixedRows = inner.trim().replace(/(?<!\\)\\(?!\\)/g, '\\\\');
    return `$\\begin{${env}} ${fixedRows} \\end{${env}}$`;
  });

  // 1. Convert common Unicode fraction characters to LaTeX
  const unicodeFractions: Record<string, string> = {
    '½': '\\frac{1}{2}', '¼': '\\frac{1}{4}', '¾': '\\frac{3}{4}',
    '⅓': '\\frac{1}{3}', '⅔': '\\frac{2}{3}', '⅕': '\\frac{1}{5}',
    '⅖': '\\frac{2}{5}', '⅗': '\\frac{3}{5}', '⅘': '\\frac{4}{5}',
    '⅙': '\\frac{1}{6}', '⅚': '\\frac{5}{6}', '⅛': '\\frac{1}{8}',
    '⅜': '\\frac{3}{8}', '⅝': '\\frac{5}{8}', '⅞': '\\frac{7}{8}',
  };
  s = s.replace(/[½¼¾⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]/g, (char) => unicodeFractions[char] || char);

  // 2. Process block math: $$...$$ and \[...\]
  s = s.replace(/\$\$([\s\S]+?)\$\$/g, (_, math) => {
    try {
      const sanitized = sanitizeMathForKatex(math);
      const rendered = katex.renderToString(sanitized, { displayMode: true, throwOnError: false, strict: false });
      if (rendered.includes('katex-error')) {
        return `<span class="katex-display">${sanitized}</span>`;
      }
      return rendered;
    } catch {
      return `$$${math}$$`;
    }
  });

  s = s.replace(/\\\[([\s\S]+?)\\\]/g, (_, math) => {
    try {
      const sanitized = sanitizeMathForKatex(math);
      const rendered = katex.renderToString(sanitized, { displayMode: true, throwOnError: false, strict: false });
      if (rendered.includes('katex-error')) {
        return `<span class="katex-display">${sanitized}</span>`;
      }
      return rendered;
    } catch {
      return `\\[${math}\\]`;
    }
  });

  // 3. Process inline math: $...$ and \(...\) (supports multi-line matrix content)
  s = s.replace(/\$([^\$]+?)\$/g, (_, math) => {
    try {
      const sanitized = sanitizeMathForKatex(math);
      const rendered = katex.renderToString(sanitized, { displayMode: false, throwOnError: false, strict: false });
      if (rendered.includes('katex-error')) {
        return `<span class="inline-math">${sanitized}</span>`;
      }
      return rendered;
    } catch {
      return `$${math}$`;
    }
  });

  s = s.replace(/\\\(([\s\S]+?)\\\)/g, (_, math) => {
    try {
      const sanitized = sanitizeMathForKatex(math);
      const rendered = katex.renderToString(sanitized, { displayMode: false, throwOnError: false, strict: false });
      if (rendered.includes('katex-error')) {
        return `<span class="inline-math">${sanitized}</span>`;
      }
      return rendered;
    } catch {
      return `\\(${math}\\)`;
    }
  });

  // 4. Process raw LaTeX fractions (including nested braces): \frac{...}{...}, \dfrac, \tfrac
  s = s.replace(/\\(?:frac|dfrac|tfrac)(\{(?:[^{}]|\{[^{}]*\})*\})(\{(?:[^{}]|\{[^{}]*\})*\})/g, (match) => {
    try {
      return katex.renderToString(match, { displayMode: false, throwOnError: false, strict: false });
    } catch {
      return match;
    }
  });

  // 5. Process raw square roots: \sqrt{...}, \sqrt[n]{...}
  s = s.replace(/\\sqrt(?:\[[^\]]+\])?(\{(?:[^{}]|\{[^{}]*\})*\})/g, (match) => {
    try {
      return katex.renderToString(match, { displayMode: false, throwOnError: false, strict: false });
    } catch {
      return match;
    }
  });

  // 6. Process raw Greek letters & math symbols: \alpha, \beta, \times, \pm, \chi^2, etc.
  s = s.replace(/\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|times|div|pm|mp|neq|ne|approx|leq|geq|le|ge|infty|cdot|circ|degree|sum|prod|int|log|ln|sin|cos|tan)(?:\^([0-9a-zA-Z]+|\{[^\}]+\})|_([0-9a-zA-Z]+|\{[^\}]+\}))?/g, (match) => {
    try {
      return katex.renderToString(match, { displayMode: false, throwOnError: false, strict: false });
    } catch {
      return match;
    }
  });

  // 7. Degree symbol: 45° -> 45^\circ
  s = s.replace(/([0-9]+(?:\.[0-9]+)?)\s*°/g, (_, num) => {
    try {
      return katex.renderToString(`${num}^\\circ`, { displayMode: false, throwOnError: false, strict: false });
    } catch {
      return `${num}°`;
    }
  });

  // 8. Superscript exponents: x², x³, 2⁵, etc.
  s = s.replace(/([a-zA-Z0-9\)])([⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ]+)/g, (_, base, sup) => {
    const map: Record<string, string> = {
      '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
      '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', 'ⁿ': 'n'
    };
    const supStr = Array.from(sup).map((c: any) => map[c] || c).join('');
    try {
      return katex.renderToString(`${base}^{${supStr}}`, { displayMode: false, throwOnError: false, strict: false });
    } catch {
      return `${base}^{${supStr}}`;
    }
  });

  // 9. Inline powers without $...$ e.g. x^2, 10^3, 2^{n+1}
  s = s.replace(/([a-zA-Z0-9\)])\^([0-9a-zA-Z]+|\{[^\}]+\})/g, (match) => {
    try {
      return katex.renderToString(match, { displayMode: false, throwOnError: false, strict: false });
    } catch {
      return match;
    }
  });

  // 10. Restore fill-in-the-blank underscores with an elegant underline span
  blanks.forEach((b, i) => {
    s = s.replaceAll(
      `%%BLANK_${i}%%`,
      '<span style="display:inline-block;border-bottom:1.5px solid currentColor;min-width:48px;margin:0 4px;"></span>'
    );
  });

  // 11. Restore protected HTML tags (use function callback so $$ in tags is not treated as replacement syntax)
  htmlTags.forEach((tag, i) => {
    s = s.replaceAll(`%%HTMLTAG_${i}%%`, () => tag);
  });

  return s;
}

interface AppConfig {
  gemini_enabled: boolean;
  gemini_status: {
    enabled: boolean;
    reason: string;
  };
  groq_enabled?: boolean;
  groq_status?: {
    enabled: boolean;
    reason: string;
  };
  llm_enabled?: boolean;
  active_provider?: string;
  answer_validation_enabled?: boolean;
  validation_enabled?: boolean;
  validation_provider?: string;
  correction_enabled?: boolean;
  correction_provider?: string;
  repair_enabled?: boolean;
  math_ocr_enabled?: boolean;
  math_ocr_engine?: string;
  question_ocr_enabled?: boolean;
  question_ocr_engine?: string;
  auto_convert_math?: boolean;
  available_math_engines?: string[];
  available_question_engines?: string[];
  available_providers?: string[];
}

interface PipelineStepItem {
  name: string;
  label: string;
  status: 'completed' | 'skipped' | 'not_needed' | 'in_progress' | 'error' | string;
  details: string;
}

const PIPELINE_STEP_CONFIGS = [
  { key: 'scanned', shortLabel: 'Scan', label: 'Scanned', icon: ScanLine },
  { key: 'checked_with_caches', shortLabel: 'Cache', label: 'Checked with Caches', icon: Database },
  { key: 'send_errors_to_llms', shortLabel: 'LLM', label: 'Send Errors to LLMs', icon: Bot },
  { key: 'ocr_scanned', shortLabel: 'OCR', label: 'OCR Scanned', icon: Cpu },
];

function getQuestionPipelineSteps(q: any): Record<string, PipelineStepItem> {
  if (q?.pipeline_steps && typeof q.pipeline_steps === 'object') {
    return q.pipeline_steps;
  }

  // Graceful fallback for legacy questions stored before pipeline_steps instrumentation
  const hasStem = Boolean(q?.stem && q.stem.trim().length > 0);
  const hasValidation = Boolean(q?.validated_answer);
  const hasMath = Boolean(
    (q?.stem && (q.stem.includes('<img') || q.stem.includes('$') || q.stem.includes('\\frac'))) ||
    Object.values(q?.options || {}).some((v: any) => typeof v === 'string' && (v.includes('<img') || v.includes('$') || v.includes('\\frac')))
  );

  return {
    scanned: {
      name: 'scanned',
      label: 'Scanned',
      status: hasStem ? 'completed' : 'not_needed',
      details: hasStem ? `Extracted from PDF Page ${(q.page_number || 0) + 1}` : 'Pending scan'
    },
    checked_with_caches: {
      name: 'checked_with_caches',
      label: 'Checked with Caches',
      status: 'completed',
      details: 'Malayalam dictionary checked & font bug cache applied'
    },
    send_errors_to_llms: {
      name: 'send_errors_to_llms',
      label: 'Send Errors to LLMs',
      status: hasValidation ? 'completed' : 'skipped',
      details: hasValidation ? `Validated with AI (${q.validated_answer})` : 'LLM validation skipped or offline'
    },
    ocr_scanned: {
      name: 'ocr_scanned',
      label: 'OCR Scanned',
      status: hasMath ? 'completed' : 'not_needed',
      details: hasMath ? 'Math equations / OCR clips processed' : 'Text-only question (no image OCR required)'
    }
  };
}

export default function Home() {
  const [jobs, setJobs] = useState<any[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [jobResult, setJobResult] = useState<any>(null);
  const [selectedQuestion, setSelectedQuestion] = useState<any>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [rescanStates, setRescanStates] = useState<Record<number, 'rescanning' | 'success' | 'error'>>({});

  // Font Manager State
  const [showFontModal, setShowFontModal] = useState(false);
  const [fontsData, setFontsData] = useState<{ installed_count: number; known_families_count: number; known_families: string[]; fonts: any[] } | null>(null);
  const [isUploadingFont, setIsUploadingFont] = useState(false);
  const [fontUploadMsg, setFontUploadMsg] = useState('');

  const loadFonts = async () => {
    try {
      const res = await fetch(`${API_BASE}/fonts/`);
      if (res.ok) {
        const data = await res.json();
        setFontsData(data);
      }
    } catch (err) {
      console.error('Failed to load fonts:', err);
    }
  };

  const handleFontUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsUploadingFont(true);
    setFontUploadMsg('');
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch(`${API_BASE}/fonts/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok && data.status === 'success') {
        setFontUploadMsg(`Successfully installed ${data.font_name} (${data.glyphs_indexed} glyphs indexed)`);
        loadFonts();
      } else {
        setFontUploadMsg(data.detail || data.message || 'Upload failed');
      }
    } catch {
      setFontUploadMsg('Error uploading font');
    } finally {
      setIsUploadingFont(false);
    }
  };

  // Pipeline Settings Modal State
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [configForm, setConfigForm] = useState<Partial<AppConfig>>({});
  const [isSavingConfig, setIsSavingConfig] = useState(false);
  const [configSaveMsg, setConfigSaveMsg] = useState('');
  const [selectedRescanEngine, setSelectedRescanEngine] = useState<string>('default');

  const openConfigModal = () => {
    if (config) {
      setConfigForm({ ...config });
    }
    setConfigSaveMsg('');
    setShowConfigModal(true);
  };

  const handleSaveConfig = async () => {
    setIsSavingConfig(true);
    setConfigSaveMsg('');
    try {
      const res = await fetch(`${API_BASE}/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(configForm)
      });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || 'Failed to update settings');
      }
      const updated = await res.json();
      setConfig(updated);
      setConfigSaveMsg('Settings saved successfully!');
      setTimeout(() => {
        setConfigSaveMsg('');
        setShowConfigModal(false);
      }, 1000);
    } catch (e: any) {
      console.error('Failed to update config:', e);
      setConfigSaveMsg(e.message || 'Failed to save settings');
    } finally {
      setIsSavingConfig(false);
    }
  };

  const [isRescanning, setIsRescanning] = useState(false);
  const [isRewriting, setIsRewriting] = useState(false);
  const [config, setConfig] = useState<AppConfig | null>(null);

  const handleRescan = async (questionNumber: number, engineOverride?: string) => {
    if (!selectedJobId) return;
    setIsRescanning(true);
    try {
      const engineParam = engineOverride && engineOverride !== 'default' ? `?engine=${engineOverride}` : '';
      const res = await fetch(`${API_BASE}/pdfs/${selectedJobId}/rescan/${questionNumber}${engineParam}`, {
        method: 'POST'
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to rescan question');
      }
      const updatedQuestion = await res.json();
      
      // Update selected question
      setSelectedQuestion(updatedQuestion);
      
      // Update the question in the jobResult list
      setJobResult((prev: any) => {
        if (!prev) return prev;
        const newQuestions = prev.questions.map((q: any) => 
          q.question_number === questionNumber ? updatedQuestion : q
        );
        return { ...prev, questions: newQuestions };
      });
    } catch (err: any) {
      console.error('Failed to rescan', err);
      alert(err.message || 'Failed to rescan question. Please check server logs.');
    } finally {
      setIsRescanning(false);
    }
  };

  const handleAiRewrite = async (qData: any) => {
    setIsRewriting(true);
    try {
      const res = await fetch(`${API_BASE}/fix_question_with_ai`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: selectedJobId,
          question_number: qData.question_number,
          stem: qData.stem,
          options: qData.options,
          marked_answer: qData.marked_answer || qData.marked_correct_answer,
          explanation: qData.explanation || qData.reasoning
        })
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to rewrite question with AI');
      }
      
      const result = await res.json();
      if (!result.success) {
        throw new Error(result.error || 'LLM failed to return a valid correction');
      }
      
      const valAns = result.validated_answer || (result.is_valid ? (qData.marked_answer || qData.marked_correct_answer) : null);
      const markedAns = qData.marked_answer || qData.marked_correct_answer;
      const isValid = result.is_valid !== undefined ? result.is_valid : Boolean(valAns && markedAns && valAns.toUpperCase() === markedAns.toUpperCase());

      const updatedQuestion = {
        ...qData,
        stem: result.corrected_stem,
        options: result.corrected_options,
        is_valid: isValid,
        validated_answer: valAns,
        reasoning: result.reasoning,
        explanation: result.reasoning || qData.explanation,
        token_usage: result.token_usage,
        pipeline_steps: {
          ...(qData.pipeline_steps || {}),
          send_errors_to_llms: {
            name: "send_errors_to_llms",
            label: "Send Errors to LLMs",
            status: "completed",
            details: isValid 
              ? `AI verified answer (Agreed: Option ${valAns || 'Match'}). Text & options rewritten.`
              : `AI disagreed: verified Option ${valAns || 'different'} vs marked ${markedAns || '—'}.`
          }
        }
      };
      
      setSelectedQuestion(updatedQuestion);
      setJobResult((prev: any) => {
        if (!prev) return prev;
        const newQuestions = prev.questions.map((q: any) => 
          q.question_number === qData.question_number ? updatedQuestion : q
        );
        const prevUsage = prev.summary?.token_usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0, estimated_cost_usd: 0 };
        const addedUsage = result.token_usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0, estimated_cost_usd: 0 };
        const newUsage = {
          prompt_tokens: (prevUsage.prompt_tokens || 0) + (addedUsage.prompt_tokens || 0),
          completion_tokens: (prevUsage.completion_tokens || 0) + (addedUsage.completion_tokens || 0),
          total_tokens: (prevUsage.total_tokens || 0) + (addedUsage.total_tokens || 0),
          estimated_cost_usd: Math.round(((prevUsage.estimated_cost_usd || 0) + (addedUsage.estimated_cost_usd || 0)) * 1000000) / 1000000,
        };
        const validatedCount = newQuestions.filter((q: any) => q.is_valid).length;
        const flaggedCount = newQuestions.filter((q: any) => q.validated_answer && !q.is_valid).length;
        return { 
          ...prev, 
          questions: newQuestions,
          summary: {
            ...prev.summary,
            validated: validatedCount,
            flagged: flaggedCount,
            token_usage: newUsage
          }
        };
      });
    } catch (err: any) {
      console.error('Failed to rewrite with AI', err);
      alert(err.message || 'Failed to rewrite question. Please check server logs.');
    } finally {
      setIsRewriting(false);
    }
  };

  const [isConvertingMath, setIsConvertingMath] = useState(false);
  const [mathStatusText, setMathStatusText] = useState<string>('');

  const convertEquations = async (jobId: string) => {
    setIsConvertingMath(true);
    setMathStatusText('Starting conversion...');
    try {
      await fetch(`${API_BASE}/pdfs/${jobId}/convert-equations`, { method: 'POST' });
      // Live-poll math status dynamically until backend marks conversion completed
      const pollTimer = setInterval(async () => {
        try {
          const statusRes = await fetch(`${API_BASE}/pdfs/${jobId}/math-status`);
          if (statusRes.ok) {
            const statusData = await statusRes.json();
            await loadJobResult(jobId);

            if (statusData.is_active) {
              const current = statusData.current_question || 0;
              const total = statusData.total_questions || 0;
              const remaining = statusData.remaining_math_images !== undefined ? statusData.remaining_math_images : 0;
              setMathStatusText(`Converting Q${current}${total ? `/${total}` : ''} (${remaining} left)...`);
            } else {
              // Backend has completely finished scanning every equation!
              clearInterval(pollTimer);
              setIsConvertingMath(false);
              setMathStatusText('');
              await loadJobResult(jobId);
            }
          } else {
            await loadJobResult(jobId);
          }
        } catch (pollErr) {
          console.error('Polling math status error', pollErr);
        }
      }, 1200);
    } catch (e) {
      console.error('Failed to trigger equation conversion', e);
      setIsConvertingMath(false);
      setMathStatusText('');
    }
  };

  useEffect(() => {
    fetchConfig();
    fetchJobs();
    const interval = setInterval(() => {
      fetchJobs();
      fetchConfig();
      if (selectedJobId) {
        loadJobResult(selectedJobId);
        // Sync conversion status in case page was reloaded while converting
        fetch(`${API_BASE}/pdfs/${selectedJobId}/math-status`)
          .then(r => r.ok ? r.json() : null)
          .then(data => {
            if (data?.is_active) {
              setIsConvertingMath(true);
              const current = data.current_question || 0;
              const total = data.total_questions || 0;
              const remaining = data.remaining_math_images !== undefined ? data.remaining_math_images : 0;
              setMathStatusText(`Converting Q${current}${total ? `/${total}` : ''} (${remaining} left)...`);
            }
          })
          .catch(() => {});
      }
    }, 4000);
    return () => clearInterval(interval);
  }, [selectedJobId]);

  const fetchConfig = async () => {
    try {
      const res = await fetch(`${API_BASE}/config`);
      const data = await res.json();
      setConfig(data);
    } catch (e) {
      console.error('Failed to fetch config', e);
    }
  };

  const fetchJobs = async () => {
    try {
      const res = await fetch(`${API_BASE}/pdfs`);
      const data = await res.json();
      setJobs(data);
    } catch (e) {
      console.error('Failed to fetch jobs', e);
    }
  };

  const deleteJob = async (e: React.MouseEvent, jobId: string) => {
    e.stopPropagation();
    try {
      await fetch(`${API_BASE}/pdfs/${jobId}`, {
        method: 'DELETE',
      });
      fetchJobs();
    } catch (err) {
      console.error('Failed to delete job', err);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    setIsUploading(true);
    
    const formData = new FormData();
    formData.append('file', e.target.files[0]);

    try {
      await fetch(`${API_BASE}/pdfs/upload`, {
        method: 'POST',
        body: formData,
      });
      fetchJobs();
    } catch (e) {
      console.error('Upload failed', e);
    } finally {
      setIsUploading(false);
    }
  };

  const loadJobResult = async (jobId: string) => {
    setSelectedJobId(jobId);
    try {
      const res = await fetch(`${API_BASE}/pdfs/${jobId}/result`);
      const data = await res.json();
      setJobResult(data.result);
    } catch (e) {
      console.error('Failed to load result', e);
    }
  };

  return (
    <div className="container">
      <header className="app-header">
        <div className="brand-wrapper">
          <div className="brand-icon">
            <Layers size={22} />
          </div>
          <div>
            <div className="brand-title">PDF QA Agent</div>
            <div className="brand-sub">
              Automated Question Extraction, Malayalam Font Engine &amp; Math LaTeX OCR
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', flexWrap: 'wrap' }}>
          <button
            onClick={openConfigModal}
            className="btn btn-secondary"
            style={{
              padding: '0.4rem 0.85rem',
              borderRadius: '9999px',
              fontSize: '0.8rem',
              color: '#e2e8f0',
              borderColor: 'rgba(56, 189, 248, 0.35)',
              background: 'rgba(56, 189, 248, 0.08)'
            }}
            title="Configure Math OCR, Question OCR, AI models, and validation"
          >
            <Sliders size={14} color="#38bdf8" />
            <span>OCR &amp; Model Settings</span>
          </button>

          <button
            onClick={() => { setShowFontModal(true); loadFonts(); }}
            className="btn btn-secondary"
            style={{
              padding: '0.4rem 0.85rem',
              borderRadius: '9999px',
              fontSize: '0.8rem',
              color: '#c7d2fe',
              borderColor: 'rgba(99, 102, 241, 0.35)'
            }}
            title="View installed Malayalam fonts and install new font files"
          >
            <Type size={14} color="#818cf8" />
            <span>Font Engine {fontsData ? `(${fontsData.installed_count})` : ''}</span>
          </button>

          {config?.groq_enabled ? (
            <div className="pill-badge" style={{ background: 'rgba(249, 115, 22, 0.12)', borderColor: 'rgba(249, 115, 22, 0.35)', color: '#fb923c' }} title="Groq AI is active. Ultra-fast inference.">
              <span className="pill-dot" style={{ background: '#f97316', boxShadow: '0 0 8px #f97316' }} />
              Groq AI Active
            </div>
          ) : config?.gemini_enabled ? (
            <div className="pill-badge valid" title="Gemini AI is active. Questions validated by Gemini.">
              <span className="pill-dot" />
              Gemini AI Active
            </div>
          ) : (
            <div className="pill-badge muted" title={config?.groq_status?.reason || config?.gemini_status?.reason || 'AI is disabled in .env (Offline Extraction Mode)'}>
              <span className="pill-dot" style={{ background: '#94a3b8' }} />
              Offline Mode (AI Disabled)
            </div>
          )}
        </div>
      </header>

      {!selectedJobId ? (
        <>
          <div className="card" style={{ padding: '0.75rem', background: 'rgba(18, 24, 38, 0.4)' }}>
            <label className="upload-zone">
              <div style={{
                width: '60px',
                height: '60px',
                borderRadius: '16px',
                background: 'rgba(99, 102, 241, 0.12)',
                border: '1px solid rgba(99, 102, 241, 0.3)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '1rem',
                color: '#818cf8'
              }}>
                <UploadCloud size={30} className={isUploading ? "spin" : ""} />
              </div>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 600, color: '#f8fafc' }}>
                {isUploading ? 'Uploading & Indexing PDF Document...' : 'Upload Question Paper PDF'}
              </h3>
              <p style={{ color: 'var(--text-muted)', marginTop: '0.35rem', fontSize: '0.86rem' }}>
                Drag and drop your document here, or click to browse files (.pdf)
              </p>
              <input type="file" accept=".pdf" onChange={handleFileUpload} style={{ display: 'none' }} disabled={isUploading} />
            </label>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '2rem', marginBottom: '0.5rem' }}>
            <h3 style={{ fontSize: '0.92rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
              Recent Documents ({jobs.length})
            </h3>
          </div>

          <div className="jobs-grid" style={{ marginTop: '0.5rem' }}>
            {jobs.map(job => (
              <div 
                key={job.job_id} 
                className="card job-card" 
                onClick={() => (job.status === 'done' || job.status === 'validating') && loadJobResult(job.job_id)}
                style={{ padding: '1.25rem' }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.75rem' }}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem', minWidth: 0 }}>
                    <div style={{
                      width: '36px',
                      height: '36px',
                      borderRadius: '8px',
                      background: 'rgba(255, 255, 255, 0.04)',
                      border: '1px solid rgba(255, 255, 255, 0.08)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: '#94a3b8',
                      flexShrink: 0
                    }}>
                      <FileText size={18} />
                    </div>
                    <div style={{ minWidth: 0 }}>
                      <div 
                        style={{ fontWeight: 600, color: '#f1f5f9', fontSize: '0.92rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}
                        title={job.filename}
                      >
                        {job.filename}
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginTop: '0.35rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        <Clock size={12} />
                        <span>{new Date(job.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</span>
                        {job.question_count != null && (
                          <>
                            <span>•</span>
                            <span style={{ fontWeight: 600, color: '#93c5fd' }}>{job.question_count} Qs</span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexShrink: 0 }}>
                    {job.status === 'done' && (
                      <span className="pill-badge valid" style={{ padding: '0.2rem 0.6rem', fontSize: '0.72rem' }}>
                        <span className="pill-dot" /> Ready
                      </span>
                    )}
                    {job.status === 'processing' && (
                      <span className="pill-badge processing" style={{ padding: '0.2rem 0.6rem', fontSize: '0.72rem' }}>
                        <span className="pill-dot" /> Scanning
                      </span>
                    )}
                    {job.status === 'validating' && (
                      <span className="pill-badge flagged" style={{ padding: '0.2rem 0.6rem', fontSize: '0.72rem' }}>
                        <span className="pill-dot" /> Validating
                      </span>
                    )}
                    {job.status === 'failed' && (
                      <span className="pill-badge danger" style={{ padding: '0.2rem 0.6rem', fontSize: '0.72rem' }}>
                        <span className="pill-dot" /> Failed
                      </span>
                    )}

                    {job.has_file && (
                      <a
                        href={`${API_BASE}/pdfs/${job.job_id}/file`}
                        target="_blank"
                        rel="noopener noreferrer"
                        onClick={(e) => e.stopPropagation()}
                        className="btn-ghost"
                        style={{ padding: '4px', borderRadius: '6px' }}
                        title="Open PDF"
                      >
                        <ExternalLink size={15} />
                      </a>
                    )}
                    <button
                      onClick={(e) => deleteJob(e, job.job_id)}
                      className="btn-ghost"
                      style={{ padding: '4px', borderRadius: '6px' }}
                      title="Delete PDF"
                    >
                      <Trash2 size={15} color="#ef4444" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div>
          <button 
            className="btn btn-secondary" 
            onClick={() => { setSelectedJobId(null); setJobResult(null); }}
            style={{ marginBottom: '1rem', borderRadius: '9999px', padding: '0.45rem 1rem' }}
          >
            <ArrowLeft size={15} />
            <span>Back to Documents</span>
          </button>
          
          {jobResult ? (
            <div>
              {/* Document Hero Header */}
              <div className="result-hero">
                <div className="hero-top-row">
                  <div className="hero-file-info">
                    <div className="file-icon-box">
                      <FileText size={22} />
                    </div>
                    <div className="file-title-wrap">
                      <div className="hero-title" title={jobResult.source_file}>
                        {jobResult.source_file}
                      </div>
                      <div className="hero-meta-subtitle">
                        <span>Offline Question Extractor</span>
                        <span>•</span>
                        <span>{jobResult.summary.total || jobResult.questions?.length || 0} Questions</span>
                      </div>
                    </div>
                  </div>

                  <div className="hero-actions">
                    <a
                      href={`${API_BASE}/pdfs/${selectedJobId}/file`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="btn btn-secondary"
                      style={{
                        padding: '0.45rem 0.95rem',
                        borderRadius: '9999px',
                        fontSize: '0.82rem',
                        textDecoration: 'none'
                      }}
                      title="Open source PDF in new browser tab"
                    >
                      <Eye size={14} color="#60a5fa" />
                      <span>View PDF</span>
                    </a>

                    {selectedJobId && jobResult?.questions?.some((q: any) => {
                      const hasMath = (text?: string) => {
                        if (!text || !text.includes('<img')) return false;
                        return text.includes('alt="math"') || text.includes('class="inline-math-img"') || !text.includes('alt="diagram"');
                      };
                      return hasMath(q.stem) || Object.values(q.options || {}).some((v: any) => typeof v === 'string' && hasMath(v));
                    }) && (
                      <button
                        onClick={() => convertEquations(selectedJobId)}
                        disabled={isConvertingMath}
                        className="btn"
                        style={{
                          padding: '0.45rem 0.95rem',
                          borderRadius: '9999px',
                          fontSize: '0.82rem',
                          background: isConvertingMath 
                            ? 'rgba(234, 179, 8, 0.2)' 
                            : 'linear-gradient(135deg, #8b5cf6, #6366f1)',
                          borderColor: isConvertingMath ? 'rgba(234, 179, 8, 0.5)' : 'rgba(139, 92, 246, 0.4)',
                          color: isConvertingMath ? '#facc15' : '#ffffff',
                          cursor: isConvertingMath ? 'wait' : 'pointer',
                          boxShadow: '0 2px 10px rgba(99, 102, 241, 0.3)'
                        }}
                        title="Converts equation image crops into KaTeX LaTeX formulas"
                      >
                        <Sparkles size={14} className={isConvertingMath ? "spin" : ""} />
                        <span>{isConvertingMath ? (mathStatusText || 'Converting to LaTeX...') : 'Convert Math to LaTeX'}</span>
                      </button>
                    )}
                  </div>
                </div>

                {/* Metric / Stat Pills Row */}
                <div className="hero-stats-row">
                  <div className="stat-chip">
                    <span className="stat-chip-label">Total Questions</span>
                    <span className="stat-chip-val">{jobResult.summary.total || jobResult.questions?.length || 0}</span>
                  </div>

                  {jobResult.summary.validated > 0 && (
                    <div className="pill-badge valid">
                      <span className="pill-dot" />
                      <span>Validated: {jobResult.summary.validated}</span>
                    </div>
                  )}

                  {jobResult.summary.flagged > 0 && (
                    <div className="pill-badge flagged">
                      <span className="pill-dot" />
                      <span>Flagged: {jobResult.summary.flagged}</span>
                    </div>
                  )}

                  {/* LLM Token Usage & Cost Chip */}
                  {jobResult.summary?.token_usage && (
                    <div 
                      className="stat-chip" 
                      style={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: '0.65rem',
                        border: '1px solid rgba(56, 189, 248, 0.25)',
                        background: 'linear-gradient(135deg, rgba(56, 189, 248, 0.08) 0%, rgba(99, 102, 241, 0.06) 100%)',
                        padding: '0.45rem 0.85rem'
                      }}
                      title={`Prompt Tokens: ${jobResult.summary.token_usage.prompt_tokens?.toLocaleString() || 0} | Output Tokens: ${jobResult.summary.token_usage.completion_tokens?.toLocaleString() || 0}`}
                    >
                      <div style={{ 
                        width: '26px', 
                        height: '26px', 
                        borderRadius: '6px', 
                        background: 'rgba(56, 189, 248, 0.15)', 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        color: '#38bdf8'
                      }}>
                        <Zap size={14} />
                      </div>
                      <div>
                        <span className="stat-chip-label" style={{ display: 'block', fontSize: '0.68rem', color: '#94a3b8' }}>
                          Token Cost
                        </span>
                        <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
                          <span style={{ fontSize: '0.92rem', fontWeight: 700, color: '#34d399' }}>
                            {formatTokenCost(jobResult.summary.token_usage)}
                          </span>
                          <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>
                            ({(jobResult.summary.token_usage.total_tokens || 0).toLocaleString()} tok)
                          </span>
                        </div>
                      </div>
                    </div>
                  )}

                  {!(jobResult.summary.validated > 0 || jobResult.summary.flagged > 0) && (
                    <div className="pill-badge muted" title="AI Answer Verification is inactive (offline mode)">
                      <span className="pill-dot" />
                      <span>AI Validation: Offline / Inactive</span>
                    </div>
                  )}

                  {jobs.find(j => j.job_id === selectedJobId)?.status === 'validating' && (
                    <div className="pill-badge processing">
                      <span className="pill-dot" />
                      <span>AI Validating in background...</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Modern Question Table */}
              <div className="table-card">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th style={{ width: '64px', textAlign: 'center' }}>Q#</th>
                      <th>Question Stem Preview</th>
                      <th style={{ width: '100px' }}>Marked Key</th>
                      <th style={{ width: '110px' }}>AI Answer</th>
                      <th style={{ width: '250px' }}>Pipeline Steps</th>
                      <th style={{ width: '130px' }}>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {jobResult.questions.map((q: any) => (
                      <tr key={q.question_number} onClick={() => setSelectedQuestion(q)}>
                        <td style={{ textAlign: 'center' }}>
                          <span className="q-num-pill">
                            {q.question_number}
                          </span>
                        </td>
                        <td style={{ maxWidth: '420px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          <span 
                            style={{ color: 'var(--text-secondary)' }}
                            dangerouslySetInnerHTML={{
                              __html: renderLatex(
                                ((q.stem || '')
                                  .replace(/<img[^>]*alt="math"[^>]*\/?>/gi, ' [Equation] ')
                                  .replace(/<img[^>]*alt="diagram"[^>]*\/?>/gi, ' [Diagram] ')
                                  .replace(/<[^>]+>/g, ' ')
                                  .trim()
                                  .slice(0, 110)) || '[Equation]'
                              )
                            }} 
                          />
                        </td>
                        <td>
                          <span style={{ 
                            fontWeight: 700, 
                            color: '#e2e8f0', 
                            background: 'rgba(255, 255, 255, 0.06)',
                            padding: '3px 9px',
                            borderRadius: '6px',
                            fontSize: '0.85rem'
                          }}>
                            {q.marked_correct_answer || q.marked_answer || '—'}
                          </span>
                        </td>
                        <td>
                          {(() => {
                            const valAns = q.validated_answer || (q.is_valid ? (q.marked_correct_answer || q.marked_answer) : null);
                            if (valAns) {
                              return (
                                <span style={{ 
                                  fontWeight: 700, 
                                  color: q.is_valid ? 'var(--success)' : 'var(--warning)',
                                  background: q.is_valid ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)',
                                  padding: '3px 9px',
                                  borderRadius: '6px',
                                  fontSize: '0.85rem'
                                }}>
                                  {valAns} {q.is_valid ? '✓' : '≠'}
                                </span>
                              );
                            }
                            return <span style={{ color: 'var(--text-muted)' }}>—</span>;
                          })()}
                        </td>
                        <td>
                          {(() => {
                            const steps = getQuestionPipelineSteps(q);
                            return (
                              <div style={{ display: 'flex', alignItems: 'center', gap: '4px', flexWrap: 'nowrap' }}>
                                {PIPELINE_STEP_CONFIGS.map(cfg => {
                                  const step = steps[cfg.key];
                                  const status = step?.status || 'completed';
                                  const isDone = status === 'completed';
                                  const isSkipped = status === 'skipped';
                                  const isNotNeeded = status === 'not_needed';
                                  const Icon = cfg.icon;

                                  let color = '#10b981';
                                  let bg = 'rgba(16, 185, 129, 0.12)';
                                  let border = 'rgba(16, 185, 129, 0.25)';

                                  if (isSkipped) {
                                    color = '#f59e0b';
                                    bg = 'rgba(245, 158, 11, 0.12)';
                                    border = 'rgba(245, 158, 11, 0.25)';
                                  } else if (isNotNeeded) {
                                    color = '#94a3b8';
                                    bg = 'rgba(148, 163, 184, 0.08)';
                                    border = 'rgba(148, 163, 184, 0.18)';
                                  } else if (status === 'error') {
                                    color = '#f87171';
                                    bg = 'rgba(248, 113, 113, 0.12)';
                                    border = 'rgba(248, 113, 113, 0.25)';
                                  }

                                  return (
                                    <span
                                      key={cfg.key}
                                      title={`${step?.label || cfg.label}: ${status.toUpperCase()}\n${step?.details || ''}`}
                                      style={{
                                        display: 'inline-flex',
                                        alignItems: 'center',
                                        gap: '3px',
                                        padding: '2px 5px',
                                        borderRadius: '5px',
                                        fontSize: '0.67rem',
                                        fontWeight: 600,
                                        color,
                                        background: bg,
                                        border: `1px solid ${border}`,
                                        cursor: 'help'
                                      }}
                                    >
                                      <Icon size={11} />
                                      {cfg.shortLabel}
                                    </span>
                                  );
                                })}
                              </div>
                            );
                          })()}
                        </td>
                        <td>
                          {q.validated_answer ? (
                            q.is_valid ? (
                              <span className="pill-badge valid" style={{ padding: '0.2rem 0.55rem', fontSize: '0.74rem' }}>
                                <CheckCircle2 size={13} /> Match
                              </span>
                            ) : (
                              <span className="pill-badge flagged" style={{ padding: '0.2rem 0.55rem', fontSize: '0.74rem' }}>
                                <AlertTriangle size={13} /> Mismatch
                              </span>
                            )
                          ) : jobs.find(j => j.job_id === selectedJobId)?.status === 'validating' ? (
                            <span className="pill-badge processing" style={{ padding: '0.2rem 0.55rem', fontSize: '0.74rem' }}>
                              Validating...
                            </span>
                          ) : (
                            <span className="pill-badge muted" style={{ padding: '0.2rem 0.55rem', fontSize: '0.74rem' }}>
                              Extracted
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="card" style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
              <RefreshCw size={24} className="spin" style={{ margin: '0 auto 0.75rem', color: '#6366f1' }} />
              <div>Loading question results...</div>
            </div>
          )}
        </div>
      )}

      {/* Detail Panel */}
      <div className={`overlay ${selectedQuestion ? 'open' : ''}`} onClick={() => setSelectedQuestion(null)} />
      <div className={`detail-panel ${selectedQuestion ? 'open' : ''}`}>
        {selectedQuestion && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.75rem', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
                <div style={{ 
                  display: 'inline-flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  background: 'linear-gradient(135deg, #3b82f6, #6366f1)',
                  color: '#fff',
                  width: '38px',
                  height: '38px',
                  borderRadius: '10px',
                  fontWeight: 800,
                  fontSize: '0.95rem'
                }}>
                  Q{selectedQuestion.question_number}
                </div>
                <div>
                  <h2 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 700 }}>Question Details</h2>
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>OCR Inspection &amp; Answer Verification</span>
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                <button
                  className="btn btn-primary"
                  onClick={() => handleAiRewrite(selectedQuestion)}
                  disabled={isRewriting}
                  style={{
                    padding: '0.35rem 0.75rem',
                    borderRadius: '9999px',
                    fontSize: '0.78rem',
                    background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                    color: 'white',
                    border: 'none',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.4rem',
                    opacity: isRewriting ? 0.7 : 1,
                    cursor: isRewriting ? 'not-allowed' : 'pointer',
                    boxShadow: '0 2px 8px rgba(16, 185, 129, 0.3)'
                  }}
                  title="Sends the current text to AI to completely rewrite it with perfect spelling and formatting"
                >
                  {isRewriting ? (
                    <>
                      <RefreshCw size={12} className="spin" color="white" />
                      <span>Rewriting...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles size={12} color="white" />
                      <span>Rewrite with AI</span>
                    </>
                  )}
                </button>
                {config?.question_ocr_enabled !== false ? (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', background: 'rgba(255,255,255,0.04)', padding: '0.2rem 0.4rem', borderRadius: '9999px', border: '1px solid rgba(255,255,255,0.08)' }}>
                    <select
                      value={selectedRescanEngine}
                      onChange={(e) => setSelectedRescanEngine(e.target.value)}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: '#94a3b8',
                        fontSize: '0.75rem',
                        outline: 'none',
                        cursor: 'pointer',
                        padding: '0.2rem 0.4rem'
                      }}
                      title="Select which OCR model/engine to use for rescanning"
                    >
                      <option value="default" style={{ background: '#18181b', color: '#f8fafc' }}>
                        Default ({config?.question_ocr_engine || 'Configured'})
                      </option>
                      <option value="groq" style={{ background: '#18181b', color: '#f8fafc' }}>Groq Vision</option>
                      <option value="gemini" style={{ background: '#18181b', color: '#f8fafc' }}>Gemini Vision</option>
                      <option value="tesseract" style={{ background: '#18181b', color: '#f8fafc' }}>Tesseract (Local $0)</option>
                      <option value="off" style={{ background: '#18181b', color: '#f87171' }}>Turn Off / Disabled</option>
                    </select>

                    <button 
                      className="btn btn-secondary" 
                      onClick={() => handleRescan(selectedQuestion.question_number, selectedRescanEngine)}
                      disabled={isRescanning}
                      style={{
                        padding: '0.35rem 0.75rem',
                        borderRadius: '9999px',
                        fontSize: '0.78rem',
                        opacity: isRescanning ? 0.7 : 1,
                        cursor: isRescanning ? 'not-allowed' : 'pointer'
                      }}
                      title="Rescans question crop directly from PDF using selected OCR engine"
                    >
                      <RefreshCw size={12} className={isRescanning ? "spin" : ""} color="#60a5fa" />
                      <span>{isRescanning ? 'Scanning...' : 'Rescan Question'}</span>
                    </button>
                  </div>
                ) : (
                  <button 
                    className="btn btn-secondary"
                    onClick={openConfigModal}
                    style={{
                      padding: '0.35rem 0.75rem',
                      borderRadius: '9999px',
                      fontSize: '0.78rem',
                      opacity: 0.7,
                      color: '#94a3b8',
                      cursor: 'pointer'
                    }}
                    title="Question OCR is disabled in Pipeline Settings. Click to enable."
                  >
                    <span>OCR Disabled (Settings)</span>
                  </button>
                )}
                <button 
                  className="btn btn-ghost" 
                  onClick={() => setSelectedQuestion(null)}
                  style={{ borderRadius: '50%', width: '36px', height: '36px', padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                >
                  <X size={18} />
                </button>
              </div>
            </div>
            
            {/* Pipeline Execution Steps */}
            {(() => {
              const steps = getQuestionPipelineSteps(selectedQuestion);
              return (
                <div className="detail-box" style={{ marginBottom: '1.25rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                    <span className="detail-box-title" style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Layers size={14} color="#38bdf8" /> Pipeline Execution Steps
                    </span>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                      Question Processing Lifecycle
                    </span>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.75rem' }}>
                    {PIPELINE_STEP_CONFIGS.map(cfg => {
                      const step = steps[cfg.key];
                      const status = step?.status || 'completed';
                      const isDone = status === 'completed';
                      const isSkipped = status === 'skipped';
                      const isNotNeeded = status === 'not_needed';
                      const Icon = cfg.icon;

                      let badgeBg = 'rgba(16, 185, 129, 0.15)';
                      let badgeColor = '#34d399';
                      let border = 'rgba(16, 185, 129, 0.25)';
                      let statusText = 'Completed';

                      if (isSkipped) {
                        badgeBg = 'rgba(245, 158, 11, 0.15)';
                        badgeColor = '#fbbf24';
                        border = 'rgba(245, 158, 11, 0.25)';
                        statusText = 'Skipped';
                      } else if (isNotNeeded) {
                        badgeBg = 'rgba(148, 163, 184, 0.12)';
                        badgeColor = '#94a3b8';
                        border = 'rgba(148, 163, 184, 0.2)';
                        statusText = 'Not Needed';
                      } else if (status === 'in_progress') {
                        badgeBg = 'rgba(56, 189, 248, 0.15)';
                        badgeColor = '#38bdf8';
                        border = 'rgba(56, 189, 248, 0.25)';
                        statusText = 'In Progress';
                      } else if (status === 'error') {
                        badgeBg = 'rgba(248, 113, 113, 0.15)';
                        badgeColor = '#f87171';
                        border = 'rgba(248, 113, 113, 0.25)';
                        statusText = 'Error';
                      }

                      return (
                        <div
                          key={cfg.key}
                          style={{
                            background: 'rgba(255, 255, 255, 0.02)',
                            border: `1px solid ${border}`,
                            borderRadius: '8px',
                            padding: '0.75rem 0.85rem',
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '0.4rem'
                          }}
                        >
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600, fontSize: '0.82rem', color: '#f1f5f9' }}>
                              <Icon size={14} color={badgeColor} />
                              {step?.label || cfg.label}
                            </div>
                            <span style={{
                              fontSize: '0.65rem',
                              fontWeight: 700,
                              padding: '2px 6px',
                              borderRadius: '4px',
                              textTransform: 'uppercase',
                              letterSpacing: '0.04em',
                              background: badgeBg,
                              color: badgeColor
                            }}>
                              {statusText}
                            </span>
                          </div>
                          <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
                            {step?.details || 'No details available'}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })()}

            {/* Question Stem Box */}
            <div className="detail-box">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.65rem' }}>
                <div className="detail-box-title" style={{ margin: 0 }}>Question Stem</div>
                {selectedQuestion.token_usage && (
                  <span style={{ 
                    fontSize: '0.72rem', 
                    color: '#38bdf8', 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '4px',
                    background: 'rgba(56, 189, 248, 0.08)',
                    padding: '2px 8px',
                    borderRadius: '9999px',
                    border: '1px solid rgba(56, 189, 248, 0.2)'
                  }} title={`Prompt: ${selectedQuestion.token_usage.prompt_tokens || 0} | Completion: ${selectedQuestion.token_usage.completion_tokens || 0}`}>
                    <Zap size={11} />
                    {selectedQuestion.token_usage.total_tokens || 0} tokens • {formatTokenCost(selectedQuestion.token_usage)}
                  </span>
                )}
              </div>
              <div
                style={{
                  lineHeight: '1.8',
                  fontSize: '1rem',
                  color: 'var(--text-main)',
                  whiteSpace: 'pre-line',
                  wordBreak: 'break-word',
                  tabSize: 4
                }}
                dangerouslySetInnerHTML={{
                  __html: renderLatex(formatQuestionStem(selectedQuestion.stem))
                }}
              />
            </div>

            {/* Answer Options Box */}
            <div className="detail-box">
              <div className="detail-box-title">Multiple Choice Options</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
                {Object.entries(selectedQuestion.options).map(([k, v]: [string, any]) => {
                  const valAns = selectedQuestion.validated_answer || (selectedQuestion.is_valid ? (selectedQuestion.marked_correct_answer || selectedQuestion.marked_answer) : null);
                  const markedAns = selectedQuestion.marked_correct_answer || selectedQuestion.marked_answer;
                  const isValidated = Boolean(valAns && valAns.toUpperCase() === k.toUpperCase());
                  const isMarked = Boolean(markedAns && markedAns.toUpperCase() === k.toUpperCase());
                  
                  const rawVal = typeof v === 'string' ? v : String(v || '');
                  // Clean redundant leading option key if already present in text (e.g. "A. option text" -> "option text")
                  const cleanVal = rawVal.replace(new RegExp(`^\\(?[${k}${k.toLowerCase()}]\\)?[\\.\\):-]?\\s*`), '').trim();

                  let border = '1px solid rgba(255, 255, 255, 0.08)';
                  let bg = 'rgba(255, 255, 255, 0.02)';
                  if (isValidated && isMarked) {
                    border = '1.5px solid rgba(16, 185, 129, 0.5)';
                    bg = 'rgba(16, 185, 129, 0.08)';
                  } else if (isValidated) {
                    border = '1.5px solid rgba(59, 130, 246, 0.5)';
                    bg = 'rgba(59, 130, 246, 0.08)';
                  } else if (isMarked) {
                    border = '1.5px solid rgba(245, 158, 11, 0.5)';
                    bg = 'rgba(245, 158, 11, 0.08)';
                  }

                  return (
                    <div
                      key={k}
                      style={{
                        padding: '0.85rem 1rem',
                        background: bg,
                        borderRadius: '10px',
                        border: border,
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'flex-start',
                        gap: '1rem',
                        transition: 'all 0.2s ease'
                      }}
                    >
                      <div style={{ display: 'flex', gap: '0.85rem', alignItems: 'flex-start', flex: 1 }}>
                        <span style={{
                          background: (isValidated && isMarked) ? '#10b981' : isValidated ? '#3b82f6' : isMarked ? '#f59e0b' : 'rgba(255, 255, 255, 0.08)',
                          color: '#fff',
                          fontWeight: 700,
                          fontSize: '0.85rem',
                          padding: '0.2rem 0.55rem',
                          borderRadius: '6px',
                          minWidth: '28px',
                          textAlign: 'center',
                          flexShrink: 0
                        }}>
                          {k}
                        </span>
                        <span
                          style={{ fontSize: '0.96rem', lineHeight: '1.5', color: 'var(--text-secondary)', whiteSpace: 'pre-line', wordBreak: 'break-word' }}
                          dangerouslySetInnerHTML={{ __html: renderLatex(cleanVal) }}
                        />
                      </div>
                      <div style={{ display: 'flex', gap: '0.4rem', flexShrink: 0 }}>
                        {isValidated && isMarked && (
                          <span className="pill-badge valid" style={{ padding: '0.15rem 0.55rem', fontSize: '0.7rem' }}>
                            <Check size={11} /> Verified Match
                          </span>
                        )}
                        {isValidated && !isMarked && (
                          <span className="pill-badge processing" style={{ padding: '0.15rem 0.55rem', fontSize: '0.7rem' }}>
                            AI Choice
                          </span>
                        )}
                        {isMarked && !isValidated && (
                          <span className="pill-badge flagged" style={{ padding: '0.15rem 0.55rem', fontSize: '0.7rem' }}>
                            Marked Key
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Answer Comparison Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.25rem' }}>
              <div className="detail-box" style={{ margin: 0 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                  <span className="detail-box-title" style={{ margin: 0 }}>Marked Exam Key</span>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', background: 'rgba(255,255,255,0.06)', padding: '2px 6px', borderRadius: '4px' }}>Ground Truth</span>
                </div>
                <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#f8fafc', marginTop: '0.35rem' }}>
                  {selectedQuestion.marked_correct_answer || selectedQuestion.marked_answer || '—'}
                </div>
              </div>

              <div className="detail-box" style={{ margin: 0 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                  <span className="detail-box-title" style={{ margin: 0 }}>AI Verified Answer</span>
                  {(() => {
                    const valAns = selectedQuestion.validated_answer || (selectedQuestion.is_valid ? (selectedQuestion.marked_correct_answer || selectedQuestion.marked_answer) : null);
                    if (valAns) {
                      return (
                        <span style={{ fontSize: '0.7rem', color: selectedQuestion.is_valid ? 'var(--success)' : 'var(--warning)', fontWeight: 600 }}>
                          {selectedQuestion.is_valid ? 'Match' : 'Mismatch'}
                        </span>
                      );
                    }
                    return (
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', background: 'rgba(255,255,255,0.04)', padding: '2px 6px', borderRadius: '4px' }}>Offline / None</span>
                    );
                  })()}
                </div>
                <div style={{ 
                  fontSize: '1.75rem', 
                  fontWeight: 800, 
                  color: (() => {
                    const valAns = selectedQuestion.validated_answer || (selectedQuestion.is_valid ? (selectedQuestion.marked_correct_answer || selectedQuestion.marked_answer) : null);
                    if (!valAns) return 'var(--text-muted)';
                    return selectedQuestion.is_valid ? 'var(--success)' : 'var(--warning)';
                  })(),
                  marginTop: '0.35rem'
                }}>
                  {selectedQuestion.validated_answer || (selectedQuestion.is_valid ? (selectedQuestion.marked_correct_answer || selectedQuestion.marked_answer) : null) || '—'}
                </div>
              </div>
            </div>

            {/* Solution / Explanation Box */}
            <div className="detail-box">
              <div className="detail-box-title">Solution &amp; Explanation</div>
              {selectedQuestion.explanation ? (
                <div
                  style={{
                    fontSize: '0.94rem',
                    lineHeight: '1.75',
                    color: 'var(--text-secondary)',
                    whiteSpace: 'pre-wrap'
                  }}
                  dangerouslySetInnerHTML={{ __html: renderLatex(selectedQuestion.explanation) }}
                />
              ) : (
                <div style={{ fontSize: '0.88rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                  No automated explanation available for this offline question.
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Pipeline & OCR Settings Modal */}
      {showConfigModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.75)',
            backdropFilter: 'blur(6px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '1.5rem'
          }}
          onClick={() => setShowConfigModal(false)}
        >
          <div
            style={{
              background: 'var(--panel-bg, #18181b)',
              border: '1px solid var(--panel-border, #27272a)',
              borderRadius: '14px',
              width: '100%',
              maxWidth: '680px',
              maxHeight: '85vh',
              overflowY: 'auto',
              padding: '1.75rem',
              boxShadow: '0 20px 50px rgba(0,0,0,0.6)'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <Sliders size={22} color="#38bdf8" />
                <h2 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 600 }}>Pipeline &amp; OCR Engine Settings</h2>
              </div>
              <button
                onClick={() => setShowConfigModal(false)}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: '4px' }}
              >
                <X size={20} />
              </button>
            </div>

            <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', lineHeight: '1.5', marginTop: 0, marginBottom: '1.5rem' }}>
              Configure models, OCR toggles, and processing behaviors. Changes persist immediately in the server database.
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              {/* SECTION 1: Whole Question OCR */}
              <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '1.25rem', borderRadius: '10px', border: '1px solid rgba(255, 255, 255, 0.07)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                  <div>
                    <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#f8fafc', margin: 0 }}>Whole Question OCR</h3>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '0.2rem 0 0' }}>
                      Controls scanned image fallback &amp; on-demand question rescanning
                    </p>
                  </div>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={configForm.question_ocr_enabled ?? true}
                      onChange={(e) => setConfigForm({ ...configForm, question_ocr_enabled: e.target.checked })}
                      style={{ width: '18px', height: '18px', accentColor: '#38bdf8', cursor: 'pointer' }}
                    />
                    <span style={{ fontSize: '0.85rem', fontWeight: 600, color: (configForm.question_ocr_enabled ?? true) ? '#38bdf8' : '#94a3b8' }}>
                      {(configForm.question_ocr_enabled ?? true) ? 'Enabled' : 'Disabled'}
                    </span>
                  </label>
                </div>

                <div style={{ marginTop: '0.85rem' }}>
                  <label style={{ fontSize: '0.82rem', color: '#cbd5e1', display: 'block', marginBottom: '0.4rem' }}>
                    Question OCR Model / Provider:
                  </label>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.65rem' }}>
                    {[
                      { id: 'groq', name: 'Groq Vision', desc: 'Ultra-fast Cloud', badge: 'Fastest' },
                      { id: 'gemini', name: 'Gemini Vision', desc: 'Google Flash/Pro', badge: 'High Accuracy' },
                      { id: 'tesseract', name: 'Tesseract', desc: 'Local Offline', badge: '$0 Cost' },
                      { id: 'off', name: 'Turn Off', desc: 'Disable Question OCR', badge: 'Off' },
                    ].map((eng) => {
                      const isSelected = eng.id === 'off'
                        ? (configForm.question_ocr_enabled === false || configForm.question_ocr_engine === 'off')
                        : (configForm.question_ocr_enabled !== false && configForm.question_ocr_engine === eng.id);
                      return (
                        <div
                          key={eng.id}
                          onClick={() => {
                            if (eng.id === 'off') {
                              setConfigForm({ ...configForm, question_ocr_enabled: false, question_ocr_engine: 'off' });
                            } else {
                              setConfigForm({ ...configForm, question_ocr_enabled: true, question_ocr_engine: eng.id });
                            }
                          }}
                          style={{
                            border: isSelected ? (eng.id === 'off' ? '2px solid #ef4444' : '2px solid #38bdf8') : '1px solid rgba(255,255,255,0.08)',
                            background: isSelected ? (eng.id === 'off' ? 'rgba(239, 68, 68, 0.12)' : 'rgba(56, 189, 248, 0.1)') : 'rgba(0,0,0,0.2)',
                            padding: '0.75rem',
                            borderRadius: '8px',
                            cursor: 'pointer',
                            position: 'relative'
                          }}
                        >
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <span style={{ fontWeight: 600, fontSize: '0.88rem', color: isSelected && eng.id === 'off' ? '#f87171' : '#f8fafc' }}>
                              {eng.name}
                            </span>
                            <span style={{ fontSize: '0.68rem', padding: '0.1rem 0.4rem', borderRadius: '4px', background: isSelected && eng.id === 'off' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(255,255,255,0.08)', color: isSelected && eng.id === 'off' ? '#f87171' : '#94a3b8' }}>
                              {eng.badge}
                            </span>
                          </div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>{eng.desc}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* SECTION 2: Mathematical Equations OCR */}
              <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '1.25rem', borderRadius: '10px', border: '1px solid rgba(255, 255, 255, 0.07)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                  <div>
                    <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#f8fafc', margin: 0 }}>Math Equations OCR</h3>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '0.2rem 0 0' }}>
                      Transcribes equation &amp; formula image snippets into clean LaTeX
                    </p>
                  </div>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={(configForm.math_ocr_enabled ?? true) && configForm.math_ocr_engine !== 'off'}
                      onChange={(e) => setConfigForm({ 
                        ...configForm, 
                        math_ocr_enabled: e.target.checked,
                        math_ocr_engine: e.target.checked ? (configForm.math_ocr_engine === 'off' ? 'auto' : configForm.math_ocr_engine || 'auto') : 'off'
                      })}
                      style={{ width: '18px', height: '18px', accentColor: '#10b981', cursor: 'pointer' }}
                    />
                    <span style={{ fontSize: '0.85rem', fontWeight: 600, color: ((configForm.math_ocr_enabled ?? true) && configForm.math_ocr_engine !== 'off') ? '#10b981' : '#94a3b8' }}>
                      {((configForm.math_ocr_enabled ?? true) && configForm.math_ocr_engine !== 'off') ? 'Enabled' : 'Disabled'}
                    </span>
                  </label>
                </div>

                <div style={{ marginTop: '0.85rem' }}>
                  <label style={{ fontSize: '0.82rem', color: '#cbd5e1', display: 'block', marginBottom: '0.4rem' }}>
                    Equation OCR Engine:
                  </label>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '0.55rem' }}>
                    {[
                      { id: 'tesseract', name: 'Tesseract', desc: 'Local parser ($0)' },
                      { id: 'groq', name: 'Groq Vision', desc: 'Qwen 3.8B' },
                      { id: 'gemini', name: 'Gemini', desc: 'Google Vision' },
                      { id: 'auto', name: 'Auto Waterfall', desc: 'Tess → Groq → Gem' },
                      { id: 'off', name: 'Turn Off', desc: 'Disable Math OCR' }
                    ].map((eng) => {
                      const isSelected = eng.id === 'off'
                        ? (configForm.math_ocr_enabled === false || configForm.math_ocr_engine === 'off')
                        : (configForm.math_ocr_enabled !== false && configForm.math_ocr_engine === eng.id);
                      return (
                        <div
                          key={eng.id}
                          onClick={() => {
                            if (eng.id === 'off') {
                              setConfigForm({ ...configForm, math_ocr_enabled: false, math_ocr_engine: 'off' });
                            } else {
                              setConfigForm({ ...configForm, math_ocr_enabled: true, math_ocr_engine: eng.id });
                            }
                          }}
                          style={{
                            border: isSelected ? (eng.id === 'off' ? '2px solid #ef4444' : '2px solid #10b981') : '1px solid rgba(255,255,255,0.08)',
                            background: isSelected ? (eng.id === 'off' ? 'rgba(239, 68, 68, 0.12)' : 'rgba(16, 185, 129, 0.1)') : 'rgba(0,0,0,0.2)',
                            padding: '0.65rem',
                            borderRadius: '8px',
                            cursor: 'pointer'
                          }}
                        >
                          <div style={{ fontWeight: 600, fontSize: '0.85rem', color: isSelected && eng.id === 'off' ? '#f87171' : '#f8fafc' }}>
                            {eng.name}
                          </div>
                          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>{eng.desc}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div style={{ marginTop: '0.9rem', paddingTop: '0.8rem', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={configForm.auto_convert_math ?? false}
                      onChange={(e) => setConfigForm({ ...configForm, auto_convert_math: e.target.checked })}
                      style={{ width: '16px', height: '16px', accentColor: '#818cf8', cursor: 'pointer' }}
                    />
                    <span style={{ fontSize: '0.82rem', color: 'var(--text-main)' }}>
                      Automatically convert math equation images to LaTeX in background on upload
                    </span>
                  </label>
                </div>
              </div>

              {/* SECTION 3: Answer Validation & Word Correction */}
              <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '1.25rem', borderRadius: '10px', border: '1px solid rgba(255, 255, 255, 0.07)' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#f8fafc', margin: '0 0 0.75rem' }}>AI Validation &amp; Malayalam Correction</h3>
                
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  {/* Validation */}
                  <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.85rem', borderRadius: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                      <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#f8fafc' }}>Answer Verification</span>
                      <input
                        type="checkbox"
                        checked={configForm.validation_enabled ?? true}
                        onChange={(e) => setConfigForm({ ...configForm, validation_enabled: e.target.checked })}
                        style={{ width: '16px', height: '16px', accentColor: '#f97316' }}
                      />
                    </div>
                    <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.4rem' }}>
                      {[
                        { id: 'groq', label: 'Groq' },
                        { id: 'gemini', label: 'Gemini' },
                        { id: 'off', label: 'Turn Off' }
                      ].map((item) => {
                        const isSelected = item.id === 'off' 
                          ? configForm.validation_enabled === false 
                          : (configForm.validation_enabled !== false && configForm.validation_provider === item.id);
                        return (
                          <button
                            key={item.id}
                            type="button"
                            onClick={() => {
                              if (item.id === 'off') {
                                setConfigForm({ ...configForm, validation_enabled: false });
                              } else {
                                setConfigForm({ ...configForm, validation_enabled: true, validation_provider: item.id });
                              }
                            }}
                            style={{
                              flex: 1,
                              padding: '0.3rem',
                              borderRadius: '6px',
                              border: isSelected ? (item.id === 'off' ? '1px solid #ef4444' : '1px solid #f97316') : '1px solid rgba(255,255,255,0.1)',
                              background: isSelected ? (item.id === 'off' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(249, 115, 22, 0.2)') : 'transparent',
                              color: isSelected ? (item.id === 'off' ? '#f87171' : '#fb923c') : '#94a3b8',
                              fontSize: '0.78rem',
                              cursor: 'pointer'
                            }}
                          >
                            {item.label}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Correction */}
                  <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.85rem', borderRadius: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                      <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#f8fafc' }}>Word Correction</span>
                      <input
                        type="checkbox"
                        checked={configForm.correction_enabled ?? true}
                        onChange={(e) => setConfigForm({ ...configForm, correction_enabled: e.target.checked })}
                        style={{ width: '16px', height: '16px', accentColor: '#818cf8' }}
                      />
                    </div>
                    <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.4rem' }}>
                      {[
                        { id: 'groq', label: 'Groq' },
                        { id: 'gemini', label: 'Gemini' },
                        { id: 'off', label: 'Turn Off' }
                      ].map((item) => {
                        const isSelected = item.id === 'off' 
                          ? configForm.correction_enabled === false 
                          : (configForm.correction_enabled !== false && configForm.correction_provider === item.id);
                        return (
                          <button
                            key={item.id}
                            type="button"
                            onClick={() => {
                              if (item.id === 'off') {
                                setConfigForm({ ...configForm, correction_enabled: false });
                              } else {
                                setConfigForm({ ...configForm, correction_enabled: true, correction_provider: item.id });
                              }
                            }}
                            style={{
                              flex: 1,
                              padding: '0.3rem',
                              borderRadius: '6px',
                              border: isSelected ? (item.id === 'off' ? '1px solid #ef4444' : '1px solid #818cf8') : '1px solid rgba(255,255,255,0.1)',
                              background: isSelected ? (item.id === 'off' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(99, 102, 241, 0.2)') : 'transparent',
                              color: isSelected ? (item.id === 'off' ? '#f87171' : '#c7d2fe') : '#94a3b8',
                              fontSize: '0.78rem',
                              cursor: 'pointer'
                            }}
                          >
                            {item.label}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </div>

              </div>
            </div>

            {/* Action Buttons */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid rgba(255, 255, 255, 0.08)' }}>
              <div style={{ fontSize: '0.85rem', color: configSaveMsg.includes('success') ? '#34d399' : '#f87171' }}>
                {configSaveMsg}
              </div>
              <div style={{ display: 'flex', gap: '0.75rem' }}>
                <button
                  type="button"
                  onClick={() => setShowConfigModal(false)}
                  className="btn btn-secondary"
                  style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleSaveConfig}
                  disabled={isSavingConfig}
                  className="btn btn-primary"
                  style={{
                    padding: '0.5rem 1.25rem',
                    fontSize: '0.85rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.4rem',
                    background: '#38bdf8',
                    color: '#0f172a',
                    fontWeight: 600,
                    border: 'none',
                    borderRadius: '8px',
                    cursor: isSavingConfig ? 'not-allowed' : 'pointer'
                  }}
                >
                  <Save size={16} />
                  <span>{isSavingConfig ? 'Saving...' : 'Save & Apply'}</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Font Manager Modal */}
      {showFontModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.75)',
            backdropFilter: 'blur(6px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '1.5rem'
          }}
          onClick={() => setShowFontModal(false)}
        >
          <div
            style={{
              background: 'var(--panel-bg, #18181b)',
              border: '1px solid var(--panel-border, #27272a)',
              borderRadius: '14px',
              width: '100%',
              maxWidth: '650px',
              maxHeight: '85vh',
              overflowY: 'auto',
              padding: '1.75rem',
              boxShadow: '0 20px 50px rgba(0,0,0,0.5)'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <Type size={22} color="#6366f1" />
                <h2 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 600 }}>Malayalam & Indic Font Engine</h2>
              </div>
              <button
                onClick={() => setShowFontModal(false)}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: '4px' }}
              >
                <X size={20} />
              </button>
            </div>

            <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', lineHeight: '1.5', marginTop: 0 }}>
              The system autonomously identifies, decompiles, and indexes embedded Malayalam fonts into a master glyph contour database for 100% offline decoding.
            </p>

            {/* Upload New Font Box */}
            <div
              style={{
                border: '2px dashed rgba(99, 102, 241, 0.4)',
                borderRadius: '10px',
                padding: '1.25rem',
                textAlign: 'center',
                background: 'rgba(99, 102, 241, 0.04)',
                marginBottom: '1.5rem'
              }}
            >
              <label style={{ cursor: 'pointer', display: 'block' }}>
                <UploadCloud size={32} color="#818cf8" style={{ margin: '0 auto 0.5rem' }} />
                <div style={{ fontWeight: 600, fontSize: '0.92rem', color: '#e0e7ff' }}>
                  {isUploadingFont ? 'Decompiling & Indexing Glyphs...' : 'Install New Font (.ttf / .otf)'}
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                  Upload any Malayalam or Indic font to index its glyph contours permanently
                </div>
                <input
                  type="file"
                  accept=".ttf,.otf"
                  onChange={handleFontUpload}
                  disabled={isUploadingFont}
                  style={{ display: 'none' }}
                />
              </label>
              {fontUploadMsg && (
                <div style={{ marginTop: '0.75rem', fontSize: '0.82rem', color: fontUploadMsg.includes('Successfully') ? '#34d399' : '#f87171' }}>
                  {fontUploadMsg}
                </div>
              )}
            </div>

            {/* Recognized Families */}
            <div style={{ marginBottom: '1.25rem' }}>
              <h4 style={{ margin: '0 0 0.5rem', fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Recognized Font Families ({fontsData?.known_families?.length || 0})
              </h4>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                {fontsData?.known_families?.map((fam) => (
                  <span
                    key={fam}
                    style={{
                      background: 'rgba(255,255,255,0.06)',
                      padding: '0.25rem 0.6rem',
                      borderRadius: '6px',
                      fontSize: '0.78rem',
                      color: 'var(--text-main)'
                    }}
                  >
                    {fam}
                  </span>
                ))}
              </div>
            </div>

            {/* Installed Font Maps */}
            <div>
              <h4 style={{ margin: '0 0 0.5rem', fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Installed Font Maps & Reference Fonts ({fontsData?.fonts?.length || 0})
              </h4>
              <div style={{ maxHeight: '200px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                {fontsData?.fonts?.map((f: any, idx: number) => (
                  <div
                    key={idx}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      background: 'rgba(255,255,255,0.03)',
                      padding: '0.5rem 0.85rem',
                      borderRadius: '6px',
                      fontSize: '0.82rem'
                    }}
                  >
                    <span style={{ fontFamily: 'monospace', color: '#c7d2fe' }}>{f.file_name}</span>
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                      {f.mapped_glyphs ? `${f.mapped_glyphs} glyphs` : f.size_bytes ? `${Math.round(f.size_bytes / 1024)} KB` : ''}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
