
const katex = { renderToString: (str) => '<span class="katex">' + str + '</span>' };
function sanitizeHtmlAndCleanText(text: string): string {
  if (!text) return '';
  let s = text;

  // 1. Decode common HTML entities
  s = s.replace(/&nbsp;/gi, ' ')
       .replace(/&quot;/gi, '"')
       .replace(/&#39;/gi, "'")
       .replace(/&amp;/gi, '&');

  // 2. Extract TeX from KaTeX HTML if full KaTeX markup is present
  s = s.replace(/<span[^>]*class="katex"[^>]*>[\s\S]*?<annotation encoding="application\/x-tex">([\s\S]*?)<\/annotation>[\s\S]*?<\/span>/gi, (_, tex) => `$${tex.trim()}$`);

  // 3. Convert KaTeX display / inline containers to LaTeX math delimiters
  s = s.replace(/<span[^>]*class="katex-display"[^>]*>([\s\S]*?)<\/span>/gi, (_, math) => {
    const trimmed = math.trim();
    return trimmed ? `\n$$${trimmed}$$\n` : '';
  });
  s = s.replace(/<span[^>]*class="inline-math"[^>]*>([\s\S]*?)<\/span>/gi, (_, math) => {
    const trimmed = math.trim();
    return trimmed ? `$${trimmed}$` : '';
  });

  // 4. Strip leftover / dangling KaTeX or span tags (including malformed ones like < spanclass = ... >)
  s = s.replace(/<\/?\s*span\b[^>]*>/gi, '');

  // 5. Convert <br> to newline
  s = s.replace(/<br\s*\/?>/gi, '\n');

  // 6. Convert <p> to paragraphs
  s = s.replace(/<p\b[^>]*>([\s\S]*?)<\/p>/gi, '$1\n\n');
  s = s.replace(/<\/?p\b[^>]*>/gi, '\n');

  // 7. Strip font, center, section tags and markdown-like bold/italic HTML tags (strictly bounded by word boundary)
  s = s.replace(/<\/?(?:font|center|section|article|b|strong|i|em)\b[^>]*>/gi, '');

  // 8. Clean divs that do NOT contain <img>
  s = s.replace(/<div(?![^>]*class="[^"]*diagram[^"]*")[^>]*>([\s\S]*?)<\/div>/gi, (match, inner) => {
    if (/<img/i.test(match)) return match; // keep images/diagrams
    return `\n${inner}\n`;
  });

  // 9. Strip any stray unrecognized tags EXCEPT <img> and diagrams
  s = s.replace(/<(?!img\b|\/?div\b)[^>]+>/gi, '');

  // 10. Heal runaway dollar signs, word splits, and severed/swallowed math
  s = s.replace(/where\\text\{\s*([a-zA-Z])\s*\}[$]*/gi, 'where $$$1$ ');
  s = s.replace(/and\\text\{\s*([a-zA-Z])\s*\}[$]*/gi, 'and $$$1$ ');
  s = s.replace(/[$]{3,}/g, ' ');
  s = healSeveredAndSwallowedMath(s);
  s = s.replace(/[ \t]{2,}/g, ' ');

  return s.trim();
}

function healSeveredAndSwallowedMath(text: string): string {
  if (!text) return '';
  let s = text;

  // Heal word splits caused by older regex passes
  s = s.replace(/\bgiven\s*(?:\$b\$|b)\s*y\b/gi, 'given by');
  s = s.replace(/\bgiven\s*(?:\$t\$|t)\s*hat\b/gi, 'given that');
  s = s.replace(/\bwhere\s*(?:\$t\$|t)\s*he\b/gi, 'where the');
  s = s.replace(/\bwhere\s*(?:\$i\$|i)\s*s\b/gi, 'where is');

  // Heal mangled subscript fractions like \frac{V_}{proton}{V_{alpha} = 1}
  s = s.replace(/\\frac\{([a-zA-Z0-9]+)_\}\{([a-zA-Z0-9]+)\}\{([a-zA-Z0-9]+)_?\{?([a-zA-Z0-9]+)\}?\s*=\s*([0-9a-zA-Z\.\-]+)\}?/g, '\\frac{$1_{$2}}{$3_{$4}} = $5');

  // Heal severed fractions: $\frac{A}$}{B} or $\frac{A}}${B}
  s = s.replace(/\$\\frac\{([^{}]+(?:\{[^{}]+\}[^{}]*)*)\}\$\}\{([^{}]+(?:\{[^{}]+\}[^{}]*)*)\}/g, '$\\frac{$1}{$2}$');
  s = s.replace(/\$\\frac\{([^{}]+(?:\{[^{}]+\}[^{}]*)*)\}\}\{([^{}]+(?:\{[^{}]+\}[^{}]*)*)\}/g, '$\\frac{$1}{$2}$');
  s = s.replace(/\\frac\{([^{}]+(?:\{[^{}]+\}[^{}]*)*)\}\}\{([^{}]+(?:\{[^{}]+\}[^{}]*)*)\}/g, '\\frac{$1}{$2}');

  // Heal specific physics malformed fractions:
  s = s.replace(/\$\\frac\{V_\{?proton\}?\}\$\}\{V_\{?alpha\b.*?[=\s]+([0-9]+)/gi, '$\\frac{V_{proton}}{V_{alpha}} = $1$');
  s = s.replace(/\\frac\{V_\{?proton\}?\}\}\{V_\{?alpha\b.*?[=\s]+([0-9]+)/gi, '\\frac{V_{proton}}{V_{alpha}} = $1');

  // Heal swallowed = RHS inside denominator or radical:
  // e.g. \frac{12.27}{\sqrt{V_{proton} = \frac{12.27}{\sqrt{V_{alpha}}}}} -> \frac{12.27}{\sqrt{V_{proton}}} = \frac{12.27}{\sqrt{V_{alpha}}}
  s = s.replace(/\\frac\{([^{}]+(?:\{[^{}]+\}[^{}]*)*)\}\{\\sqrt\{\s*([^{}=\s]+(?:\s*\{[^{}]+\})*)\s*=\s*(\\frac\{[^{}]+(?:\{[^{}]+\}[^{}]*)*\}\{\\sqrt\{[^{}]+(?:\{[^{}]+\}[^{}]*)*\}\})\s*\}\}/g, '\\frac{$1}{\\sqrt{$2}} = $3');
  s = s.replace(/\\frac\{([^{}]+(?:\{[^{}]+\}[^{}]*)*)\}\{\\sqrt\{\s*([^{}=\s]+(?:\s*\{[^{}]+\})*)\s*=\s*([^{}]+(?:\{[^{}]+\}[^{}]*)*)\s*\}\}/g, '\\frac{$1}{\\sqrt{$2}} = $3');
  s = s.replace(/\\frac\{([^{}]+(?:\{[^{}]+\}[^{}]*)*)\}\{\s*([^{}=\s]+(?:\s*\{[^{}]+\})*)\s*=\s*(\\frac\{[^{}]+(?:\{[^{}]+\}[^{}]*)*\}\{[^{}]+(?:\{[^{}]+\}[^{}]*)*\})\s*\}/g, '\\frac{$1}{$2} = $3');
  s = s.replace(/\\frac\{([^{}]+(?:\{[^{}]+\}[^{}]*)*)\}\{\s*([^{}=\s]+(?:\s*\{[^{}]+\})*)\s*=\s*([0-9a-zA-Z\.\-]+)\s*\}/g, '\\frac{$1}{$2} = $3');

  return s;
}

function healUnitsAndSlashes(text: string): string {
  if (!text) return '';
  let s = text;
  // 1. Redundant slash directly before a negative exponent (e.g. N m^2⁄ C^{-1} -> N m^2 C^{-1})
  s = s.replace(/\s*[\u2044\u2215\/]\s*(?=[A-Za-z]\^\{?-)/g, ' ');

  // 2. Trailing unit slash: (Unit1)\s+(Unit2)[slash](following word)
  // e.g. "N C⁄" -> "N/C", "C m^2⁄" -> "C/m^2", "V m⁄" -> "V/m", "km h⁄" -> "km/h"
  const unitPrefix = '(?:[A-Z][a-zA-Z]*|km|cm|mm|kg|mg|m)';
  s = s.replace(new RegExp(`\\b(${unitPrefix}(?:\\^[0-9\\-]+)?)\\s+([A-Za-z]+(?:\\^[0-9\\-]+|\\d+)?)\\s*[\\u2044\\u2215\\/]([a-zA-Z]*)`, 'g'), (_, u1, u2, post) => {
    const u2Clean = u2.replace(/([a-zA-Z]+)(\d+)/g, '$1^$2');
    const spaceAfter = (post && /^[a-zA-Z]/.test(post)) ? ' ' : '';
    return `${u1}/${u2Clean}${spaceAfter}${post || ''}`;
  });

  // 3. Fraction slash U+2044 and division slash U+2215 conversion to standard '/'
  s = s.replace(/[\u2044\u2215]/g, '/');

  // 4. Collapse spaces around unit slashes (e.g. "N / C" -> "N/C", "km / h" -> "km/h")
  s = s.replace(new RegExp(`\\b(${unitPrefix})\\s*\\/\\s*([A-Za-z]+(?:\\^[0-9\\-]+)?)\\b`, 'g'), '$1/$2');

  return s;
}

function healStudentReadability(text: string): string {
  if (!text) return '';
  let s = text;
  // 1. Un-mash prepositions and single-letter variables in running text
  // e.g. whereh -> where $h$, letx -> let $x$, withv -> with $v$ (contiguous only)
  s = s.replace(/\b(where|when|let|if|with|here)([a-zA-Z])\b/gi, '$1 $$$2$');
  // e.g. ande -> and $e$, andh -> and $h$ (exclude 'and a' or 'and I')
  s = s.replace(/\b(and)([b-hj-zB-HJ-Z])\b/gi, '$1 $$$2$');
  // Isolated single-letter variable following where/let/if/when with space (e.g. "where V is", "let x be")
  s = s.replace(/\b(where|when|let|if|with|here)\s+([b-hj-zB-HJ-Z])\b(?=\s+[a-zA-Z]{2,})/gi, '$1 $$$2$');

  const knownMathCommands = new Set([
    'frac', 'dfrac', 'tfrac', 'sqrt', 'times', 'div', 'pm', 'mp', 'cdot', 'circ', 'degree',
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon', 'zeta', 'eta', 'theta', 'vartheta',
    'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'varpi', 'rho', 'varrho', 'sigma', 'varsigma',
    'tau', 'upsilon', 'phi', 'varphi', 'chi', 'psi', 'omega',
    'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon', 'Phi', 'Psi', 'Omega',
    'neq', 'ne', 'approx', 'leq', 'geq', 'le', 'ge', 'infty', 'sum', 'prod', 'int', 'oint',
    'log', 'ln', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan',
    'in', 'notin', 'subset', 'subseteq', 'supset', 'supseteq', 'cup', 'cap', 'setminus',
    'partial', 'nabla', 'vec', 'hat', 'dot', 'ddot', 'bar', 'tilde', 'overline', 'underline',
    'begin', 'end', 'left', 'right', 'bmatrix', 'pmatrix', 'vmatrix', 'Vmatrix', 'Bmatrix', 'matrix', 'array',
    'sim', 'hbar', 'rightarrow', 'to', 'leftarrow', 'leftrightarrow', 'Rightarrow', 'Leftarrow', 'Leftrightarrow',
    'text', 'mathrm', 'mathbf', 'mathit', 'mathbb', 'mathcal', 'quad', 'qquad',
    'equiv', 'propto', 'perp', 'parallel', 'therefore', 'because', 'angle', 'triangle',
    'bullet', 'dots', 'cdots', 'ldots', 'over', 'choose', 'cases', 'hline'
  ]);

  const commonProseWords = [
    'is', 'are', 'was', 'were', 'the', 'of', 'and', 'constant', 'charge',
    'electron', 'mass', 'velocity', 'potential', 'frequency', 'energy',
    'force', 'field', 'value', 'between', 'with', 'from', 'where', 'planck',
    'which', 'that', 'this', 'given', 'equal', 'equals', 'zero', 'ratio',
    'graph', 'slope', 'stopping', 'threshold', 'work', 'function', 'kinetic',
    'state', 'ground', 'excited', 'particle', 'wavelength', 'momentum'
  ];

  const fixMathProseBlock = (match: string, inner: string) => {
    const raw = inner.trim();
    if (!raw) return match;
    if (raw.includes('\uE000HTMLTAG')) return match;

    // Check if it has genuine math commands and no bare Indic characters
    const cmds = raw.match(/\\([a-zA-Z]+)/g)?.map(c => c.slice(1)) || [];
    const hasIndic = /[\u0900-\u0DFF]/.test(raw);
    if (cmds.length > 0 && cmds.some(c => knownMathCommands.has(c)) && !hasIndic) {
      return match;
    }

function sanitizeMathForKatex(math: string): string {
  if (!math) return '';
  let m = math.trim();

  // Replace intra-math newlines so equations never stack vertically across multiple lines
  m = m.replace(/[\r\n]+/g, ' ');

  // Normalize Unicode math & Greek symbols
  m = m.replace(/×/g, '\\times ').replace(/÷/g, '\\div ').replace(/±/g, '\\pm ');
  m = m.replace(/≤/g, '\\le ').replace(/≥/g, '\\ge ').replace(/≠/g, '\\ne ');
  m = m.replace(/°/g, '^\\circ ');
  m = m.replace(/π/g, '\\pi ').replace(/σ/g, '\\sigma ').replace(/ε|ϵ/g, '\\epsilon ');
  m = m.replace(/τ/g, '\\tau ').replace(/ϕ|φ/g, '\\phi ').replace(/θ/g, '\\theta ');
  m = m.replace(/∆|Δ/g, '\\Delta ').replace(/∴/g, '\\therefore ');
  m = m.replace(/μ|µ/g, '\\mu ');
  // Arrow and logical symbols
  m = m.replace(/⇒/g, '\\Rightarrow ').replace(/⇐/g, '\\Leftarrow ');
  m = m.replace(/→/g, '\\rightarrow ').replace(/←/g, '\\leftarrow ');
  m = m.replace(/≈/g, '\\approx ').replace(/∝/g, '\\propto ');
  m = m.replace(/∞/g, '\\infty ').replace(/≡/g, '\\equiv ');
  // Vector hat characters
  m = m.replace(/î/g, '\\hat{i}').replace(/ĵ/g, '\\hat{j}');
  m = m.replace(/k\u0302/g, '\\hat{k}').replace(/x\u0302/g, '\\hat{x}');
  m = m.replace(/n\u0302/g, '\\hat{n}').replace(/r\u0302/g, '\\hat{r}');
  // Unicode subscript digits inside math
  m = m.replace(/₀/g, '_0').replace(/₁/g, '_1').replace(/₂/g, '_2').replace(/₃/g, '_3').replace(/₄/g, '_4');
  m = m.replace(/₅/g, '_5').replace(/₆/g, '_6').replace(/₇/g, '_7').replace(/₈/g, '_8').replace(/₉/g, '_9');

  // Clean trailing backslashes
  m = m.replace(/\\+$/, '');

  // Matrix healing: restore missing backslashes for begin/end/left/right
  m = m.replace(/(?<!\\)\b(begin|end|left|right)\b/g, '\\$1');
  m = m.replace(/\\left\[\s*\\begin\{array\}\{[a-z]+\}\s*([\s\S]+?)\s*\\end\{array\}\s*\\right\]/g, '\\begin{bmatrix} $1 \\end{bmatrix}');

  // Comprehensive Precision Fraction Healing:
  // (Removed regex-based fraction healing because it corrupts valid nested LaTeX like \frac{V_{rms}}{R})

  // Clean unknown single-letter backslash commands: e.g. \Y -> Y, \A -> A
  m = m.replace(/\\([a-zA-Z]+)/g, (full, cmd) => {
    if (KNOWN_KATEX_COMMANDS.has(cmd)) return full;
    return cmd;
  });

  // Bidirectional auto-balance curly braces (drop stray unmatched '}' and close unclosed '{')
  let depth = 0;
  const chars: string[] = [];
  for (let i = 0; i < m.length; i++) {
    const ch = m[i];
    if (ch === '{' && (i === 0 || m[i - 1] !== '\\')) {
      depth++;
      chars.push(ch);
    } else if (ch === '}' && (i === 0 || m[i - 1] !== '\\')) {
      if (depth > 0) {
        depth--;
        chars.push(ch);
      }
      // Else drop stray unmatched closing brace
    } else {
      chars.push(ch);
    }
  }
  if (depth > 0) {
    chars.push('}'.repeat(depth));
  }

function renderLatex(text: string): string {
  if (!text) return '';

  let s = text;

  // Convert markdown images to HTML image containers if present
  s = s.replace(/!\[(.*?)\]\((https?:\/\/[^\s\)]+)\)/g, '<div class="my-3 flex flex-col items-center"><img src="$2" alt="$1" class="max-h-64 rounded shadow-sm border border-slate-200" /></div>');

  // Ensure Cloudflare CDN domain always points to examcdn.trogon.info
  s = s.replaceAll('exam.trogon.info', 'examcdn.trogon.info');

  // Strip any text link anchor showing below images
  s = s.replace(/<a\s+[^>]*>[\s\S]*?<\/a>/gi, '');

  // 0. Protect HTML tags (inline equation images & diagrams) FIRST before ANY text/math alterations
  const htmlTags: string[] = [];
  const HTML_TAG_PREFIX = '\uE000HTMLTAG';
  const HTML_TAG_SUFFIX = '\uE001';
  s = s.replace(/<div[^>]*>[\s\S]*?<\/div>|<img[^>]*\/?>/gi, (match) => {
    const id = `${HTML_TAG_PREFIX}${htmlTags.length}${HTML_TAG_SUFFIX}`;
    htmlTags.push(match);
    return id;
  });

  s = sanitizeHtmlAndCleanText(s);

  // Convert markdown bold **...** into styled HTML strong tags
  s = s.replace(/\*\*([^*\n]+?)\*\*/g, '<strong style="font-weight: 700; color: inherit;">$1</strong>');

  // Auto-heal truncated words caused by faulty stripping or OCR
  s = s.replace(/\bOth\b/g, 'Both').replace(/\both\b/g, (match, offset) => {
    return (offset === 0 || /(?:^|[.!?\n]\s*)$/.test(text.slice(0, offset))) ? 'Both' : 'both';
  });
  s = s.replace(/\bll\s+of\s+the\s+above\b/gi, 'All of the above');

  // Gracefully replace any raw leftover diagram placeholder with a styled badge if no image was provided
  s = s.replace(/\[DIAGRAM\/IMAGE\s+PLACEHOLDER\]/gi, '<div style="display:flex;justify-content:center;margin:12px 0;"><span style="display:inline-flex;align-items:center;gap:6px;padding:6px 14px;background:rgba(56,189,248,0.08);border:1px dashed rgba(56,189,248,0.3);border-radius:6px;font-size:0.82rem;color:#38bdf8;font-weight:600;">📊 Diagram / Figure</span></div>');
  s = s.replace(/\[\[\s*QAI_IMAGE:[^\]]+\]\]/gi, '');

  // Pre-heal fraction slashes and scientific units before LaTeX processing
  s = healUnitsAndSlashes(s);
  // Heal student readability (unwrap mistakenly wrapped prose, un-mash words and variables)
  s = healStudentReadability(s);
  // Restore exponents flattened by PDF text extraction, e.g. 4.0 × 1016.
  s = s.replace(
    /(\b\d+(?:\.\d+)?)\s*(?:×|x|\\times)\s*10\s*(\d{2,3})(?!\d)/gi,
    '$1 \\times 10^{$2}'
  );

  // 0a. Normalize vector hat characters in running text
  s = s.replace(/î/g, '$\\hat{i}$').replace(/ĵ/g, '$\\hat{j}$');
  s = s.replace(/k\u0302/g, '$\\hat{k}$').replace(/x\u0302/g, '$\\hat{x}$');
  s = s.replace(/n\u0302/g, '$\\hat{n}$').replace(/r\u0302/g, '$\\hat{r}$');

  // 0b. Normalize Unicode subscript digits: q₀ → q$_0$, r₁ → r$_1$, ε₀ → ε$_0$
  const subscriptMap: Record<string, string> = {
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
    '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
  };
  s = s.replace(/([a-zA-Z\u03B1-\u03C9\u03A1-\u03A9])([\u2080-\u2089]+)/g, (_, base, subs) => {
    const subStr = Array.from(subs).map((c: any) => subscriptMap[c] || c).join('');
    try {
      return katex.renderToString(`${base}_{${subStr}}`, { displayMode: false, throwOnError: false, strict: false });
    } catch {
      return `${base}_{${subStr}}`;
    }
  });
  // Handle standalone subscript digits (not preceded by a letter)
  s = s.replace(/[\u2080-\u2089]+/g, (subs) => {
    const subStr = Array.from(subs).map((c: any) => subscriptMap[c] || c).join('');
    return `$_{${subStr}}$`;
  });

  // 0c. Normalize arrow and logical symbols
  s = s.replace(/⇒/g, '$\\Rightarrow$').replace(/∴/g, '$\\therefore$');
  s = s.replace(/⇐/g, '$\\Leftarrow$').replace(/≈/g, '$\\approx$');
  s = s.replace(/∝/g, '$\\propto$').replace(/∞/g, '$\\infty$');

  // 0.1 Protect fill-in-the-blanks (3 or more underscores) so they don't break LaTeX parsing
  const blanks: string[] = [];
  const BLANK_PREFIX = '\uE002BLANK';
  const BLANK_SUFFIX = '\uE003';
  s = s.replace(/_{3,}/g, (match) => {
    const id = `${BLANK_PREFIX}${blanks.length}${BLANK_SUFFIX}`;
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
  const renderedMath: string[] = [];
  const MATH_PREFIX = '\uE004KATEX';
  const MATH_SUFFIX = '\uE005';

  s = s.replace(/\$\$([\s\S]+?)\$\$/g, (_, math) => {
    if (math.includes('\uE000HTMLTAG')) return `$$${math}$$`;
    try {
      const sanitized = sanitizeMathForKatex(math);
      const rendered = katex.renderToString(sanitized, { displayMode: true, throwOnError: false, strict: false });
      const id = `${MATH_PREFIX}${renderedMath.length}${MATH_SUFFIX}`;
      renderedMath.push(rendered.includes('katex-error') ? `<span class="inline-math">${sanitized}</span>` : rendered);
      return id;
    } catch {
      return `$$${math}$$`;
    }
  });

  s = s.replace(/\\\[([\s\S]+?)\\\]/g, (_, math) => {
    if (math.includes('\uE000HTMLTAG')) return `\\[${math}\\]`;
    try {
      const sanitized = sanitizeMathForKatex(math);
      const rendered = katex.renderToString(sanitized, { displayMode: true, throwOnError: false, strict: false });
      const id = `${MATH_PREFIX}${renderedMath.length}${MATH_SUFFIX}`;
      renderedMath.push(rendered.includes('katex-error') ? `<span class="inline-math">${sanitized}</span>` : rendered);
      return id;
    } catch {
      return `\\[${math}\\]`;
    }
  });

  // 3. Process inline math: $...$ and \(...\) (supports multi-line matrix content)
  s = s.replace(/\$([^\$]+?)\$/g, (_, math) => {
    if (math.includes('\uE000HTMLTAG')) return `$${math}$`;
    try {
      const sanitized = sanitizeMathForKatex(math);
      const rendered = katex.renderToString(sanitized, { displayMode: false, throwOnError: false, strict: false });
      const id = `${MATH_PREFIX}${renderedMath.length}${MATH_SUFFIX}`;
      renderedMath.push(rendered.includes('katex-error') ? `<span class="inline-math">${sanitized}</span>` : rendered);
      return id;
    } catch {
      return `$${math}$`;
    }
  });

  s = s.replace(/\\\(([\s\S]+?)\\\)/g, (_, math) => {
    if (math.includes('\uE000HTMLTAG')) return `\\(${math}\\)`;
    try {
      const sanitized = sanitizeMathForKatex(math);
      const rendered = katex.renderToString(sanitized, { displayMode: false, throwOnError: false, strict: false });
      const id = `${MATH_PREFIX}${renderedMath.length}${MATH_SUFFIX}`;
      renderedMath.push(rendered.includes('katex-error') ? `<span class="inline-math">${sanitized}</span>` : rendered);
      return id;
    } catch {
      return `\\(${math}\\)`;
    }
  });

  // 4. Process raw LaTeX fractions (including nested braces and pre-healing broken fractions)
  s = healSeveredAndSwallowedMath(s);
  s = s.replace(/\\(?:frac|dfrac|tfrac)(\{(?:[^{}]|\{[^{}]*\})*\})(\{(?:[^{}]|\{[^{}]*\})*\})/g, (match) => {
    try {
      const sanitized = sanitizeMathForKatex(match);
      const rendered = katex.renderToString(sanitized, { displayMode: false, throwOnError: false, strict: false });
      return rendered.includes('katex-error') ? match : rendered;
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

  // 5.5 Process raw scientific notation powers before standalone \times:
  // e.g. 26.4 \times 10^{-12}, 1.5 \times 10^{-16}, 3 \times 10^{-10}, 10^{-12}
  s = s.replace(/(?<=^|[\s\(\[\{=+\-*/,~])(?:\d+(?:\.\d+)?\s*(?:\\times|×)\s*)?([a-zA-Z0-9]+|\))\^([0-9a-zA-Z\-]+|\{[^}]+\})(?=$|[\s\)\]},.;:!?=+\-*/~])/g, (match) => {
    try {
      const sanitized = sanitizeMathForKatex(match);
      return katex.renderToString(sanitized, { displayMode: false, throwOnError: false, strict: false });
    } catch {
      return match;
    }
  });

  // 6. Process raw Greek letters & math symbols: \alpha, \beta, \times, \pm, \chi^2, etc.
  s = s.replace(/\\(alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|iota|kappa|lambda|mu|nu|xi|pi|rho|varrho|sigma|tau|upsilon|phi|varphi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|times|div|pm|mp|neq|ne|approx|leq|geq|le|ge|infty|cdot|circ|degree|sum|prod|int|log|ln|sin|cos|tan)(?:\^([0-9a-zA-Z]+|\{[^\}]+\})|_([0-9a-zA-Z]+|\{[^\}]+\}))?/g, (match) => {
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

  // 9. Inline powers without $...$ e.g. x^2, 10^3, 2^{n+1}, m^2
  s = s.replace(/(?<=^|[\s\(\[\{=+\-*/,~])([a-zA-Z0-9]+|\))\^([0-9a-zA-Z\-]+|\{[^}]+\})(?=$|[\s\)\]},.;:!?=+\-*/~])/g, (match) => {
    try {
      const sanitized = sanitizeMathForKatex(match);
      return katex.renderToString(sanitized, { displayMode: false, throwOnError: false, strict: false });
    } catch {
      return match;
    }
  });

  // 9b. Inline subscripts without $...$ e.g. q_0, q_{0}, a_1, x_i
  s = s.replace(/(?<=^|[\s\(\[\{=+\-*/,~])([a-zA-Z\u03B1-\u03C9\u0391-\u03A9]{1,2})_([0-9a-zA-Z\u03B1-\u03C9]{1,3}|\{[^}]+\})(?=$|[\s\)\]},.;:!?=+\-*/~])/g, (match) => {
    try {
      return katex.renderToString(match, { displayMode: false, throwOnError: false, strict: false });
    } catch {
      return match;
    }
  });

  // (Removed bare \text{} stripping)

  // 10. Restore fill-in-the-blank underscores with an elegant underline span
  blanks.forEach((b, i) => {
    const blankHtml = '<span style="display:inline-block;border-bottom:1.5px solid currentColor;min-width:48px;margin:0 4px;"></span>';
    s = s.replaceAll(`${BLANK_PREFIX}${i}${BLANK_SUFFIX}`, blankHtml);
    s = s.replaceAll(`%%BLANK_${i}%%`, blankHtml);
  });

  // 11. Restore protected HTML tags (use function callback so $$ in tags is not treated as replacement syntax)
  htmlTags.forEach((tag, i) => {
    const cleanReplacer = () => {
      let normalizedTag = tag.replaceAll('exam.trogon.info', 'examcdn.trogon.info');
      // Strip any text link anchor below or inside the image container
      normalizedTag = normalizedTag.replace(/<a\s+[^>]*>[\s\S]*?<\/a>/gi, '');
      return normalizedTag;
    };

    s = s.replaceAll(`${HTML_TAG_PREFIX}${i}${HTML_TAG_SUFFIX}`, cleanReplacer);
    s = s.replaceAll(`%%HTMLTAG_${i}%%`, cleanReplacer);
    s = s.replaceAll(`%%HTMLTAG${i}%%`, cleanReplacer);
    s = s.replaceAll(`___HTML_TAG_${i}___`, cleanReplacer);
    s = s.replaceAll(`\x02HTMLTAG${i}\x03`, cleanReplacer);
    s = s.replaceAll(`\u0002HTMLTAG${i}\u0003`, cleanReplacer);
    s = s.replaceAll(`HTMLTAG${i}`, cleanReplacer);
    // Also heal any legacy/KaTeX-corrupted tag tokens like %%HTMLTA<span...G_0...%%
    s = s.replace(new RegExp(`\uE000HTMLTAG(?:<[^>]+>|[^\uE001])*?${i}(?:<[^>]+>|[^\uE001])*?\uE001`, 'g'), cleanReplacer);
  });

  // 12. Restore protected KaTeX rendered blocks so subsequent regexes never mutated them
  renderedMath.forEach((mathHtml, i) => {
    s = s.replaceAll(`${MATH_PREFIX}${i}${MATH_SUFFIX}`, () => mathHtml);
  });

  // 13. Strip any stray raw HTMLTAG markers that leaked from the backend database (for older processed PDFs)
  s = s.replace(/[\u0000-\u001F\uE000-\uE007]*HTMLTAG\d+[\u0000-\u001F\uE000-\uE007]*/gi, '');

  return s;
}


const KNOWN_KATEX_COMMANDS = new Set([
  'frac', 'dfrac', 'tfrac', 'sqrt', 'times', 'div', 'pm', 'mp', 'cdot', 'circ', 'degree',
  'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon', 'zeta', 'eta', 'theta', 'vartheta', 'iota', 'kappa',
  'lambda', 'mu', 'nu', 'xi', 'pi', 'varpi', 'rho', 'varrho', 'sigma', 'varsigma', 'tau', 'upsilon', 'phi', 'varphi',
  'chi', 'psi', 'omega', 'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon', 'Phi', 'Psi', 'Omega',
  'neq', 'ne', 'approx', 'leq', 'geq', 'le', 'ge', 'infty', 'sum', 'prod', 'int', 'oint', 'log', 'ln', 'sin', 'cos', 'tan',
  'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan', 'in', 'notin', 'subset', 'subseteq', 'supset', 'supseteq', 'cup', 'cap',
  'setminus', 'partial', 'nabla', 'vec', 'hat', 'dot', 'ddot', 'bar', 'tilde', 'overline', 'underline', 'begin', 'end',
  'left', 'right', 'bmatrix', 'pmatrix', 'vmatrix', 'Vmatrix', 'Bmatrix', 'matrix', 'array', 'sim', 'hbar', 'rightarrow',
  'to', 'leftarrow', 'leftrightarrow', 'Rightarrow', 'Leftarrow', 'Leftrightarrow', 'text', 'mathrm', 'mathbf', 'mathit',
  'mathcal', 'quad', 'qquad', 'equiv', 'propto', 'perp', 'parallel', 'therefore', 'because', 'angle', 'triangle', 'bullet',
  'dots', 'cdots', 'ldots', 'over', 'choose', 'cases', 'hline', 'rm', 'bf', 'it', 'cal', 'mathbb', 'oint', 'lim', 'det',
  'sec', 'csc', 'cot', 'arcsin', 'arccos', 'arctan', 'forall', 'exists', 'implies', 'iff'
]);

console.log("TEST 1:");
console.log(renderLatex("\\vec{Q} \\Phi_{E}8 = 2l"));

console.log("\nTEST 2:");
console.log(renderLatex("\\vec{Q} $\\Phi$ _{E}8 = 2l"));

console.log("\nTEST 3:");
console.log(renderLatex("$\\vec{Q}$_{E}8 = 2l"));

console.log("\nTEST 4:");
console.log(renderLatex("$\\vec{Q} \\Phi_{E}$8 = 2l"));

console.log("\nTEST 5: Using OCR glitch");
console.log(renderLatex("Here._1 E = 6 V"));

console.log("\nTEST 6: Option B glitch");
console.log(renderLatex("1.5 N / 10^{-16} C"));
