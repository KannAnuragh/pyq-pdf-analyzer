export function renderLatex(text: string): string {
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
