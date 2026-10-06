'use client';

import { useState, useEffect } from 'react';
import { 
  UploadCloud, FileText, CheckCircle2, AlertTriangle, Trash2, ExternalLink, 
  RefreshCw, Type, X, Sparkles, ArrowLeft, Eye, Check, Clock, Cpu, ShieldCheck, 
  ChevronRight, BarChart3, HelpCircle, Layers, Sliders, Settings, Save, CheckCircle,
  ScanLine, Database, Bot, Zap, Cloud, Copy
} from 'lucide-react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || (
  typeof window === 'undefined'
    ? 'http://localhost:8000'
    : `${window.location.protocol}//${window.location.hostname}:8000`
);

const KNOWN_KATEX_COMMANDS = new Set([
  'frac', 'dfrac', 'tfrac', 'sqrt', 'times', 'div', 'pm', 'mp', 'cdot', 'circ', 'degree',
  'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon', 'zeta', 'eta', 'theta', 'vartheta', 'iota', 'kappa',
  'lambda', 'mu', 'nu', 'xi', 'pi', 'varpi', 'rho', 'varrho', 'sigma', 'varsigma', 'tau', 'upsilon', 'phi', 'varphi', 'chi', 'psi',
  'omega', 'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon', 'Phi', 'Psi',
  'Omega', 'neq', 'ne', 'approx', 'leq', 'geq', 'le', 'ge', 'infty', 'sum', 'prod', 'int',
  'log', 'ln', 'sin', 'cos', 'tan', 'in', 'partial', 'nabla', 'vec', 'hat', 'dot', 'ddot', 'overline', 'underline',
  'begin', 'end', 'left', 'right', 'bmatrix', 'pmatrix', 'vmatrix', 'Vmatrix', 'Bmatrix', 'matrix', 'array',
  'sim', 'bar', 'tilde', 'hbar', 'rightarrow', 'to', 'leftrightharpoons', 'rightleftharpoons', 'leftrightarrow',
  'text', 'mathrm', 'mathbf', 'quad', 'qquad', 'equiv', 'propto', 'perp', 'parallel', 'cup', 'cap', 'subset', 'subseteq',
  'therefore', 'because', 'Rightarrow', 'Leftarrow', 'Leftrightarrow', 'leftarrow', 'rightarrow',
  'angle', 'triangle', 'bullet', 'dots', 'cdots', 'ldots', 'over', 'choose', 'cases', 'hline', 'rm', 'bf', 'it', 'cal', 'mathbb',
  'oint', 'lim', 'det', 'sec', 'csc', 'cot', 'arcsin', 'arccos', 'arctan', 'forall', 'exists', 'implies', 'iff'
]);

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

  return chars.join('');
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
  s = s.replace(/^(?:Question\s*|Q\.?\s*|\u0d1a\u0d4b\u0d26\u0d4d\u0d2f\u0d02\s*)\d+\s*(?::\-?|[\.\):-])\s*/i, '');
  s = s.replace(/^\d+\s*[\.\):-]\s+/, '');

  // Strip trailing marks indicators like [1 mark], (2 marks), [5 marks]
  s = s.replace(/\s*\[\s*\d+(?:\.\d+)?(?:\s*\/\s*\d+)?\s*(?:marks?|mark|m)\s*\]/gi, '');
  s = s.replace(/\s*\(\s*\d+(?:\.\d+)?(?:\s*\/\s*\d+)?\s*(?:marks?|mark|m)\s*\)/gi, '');

  // Separate Reason onto a newline if it was merged right after Assertion
  s = s.replace(/(?<=[^\n])\s*(?=\b(?:Reason|\u0d15\u0d3e\u0d30\u0d23\u0d02)\s*(?:\([rR]\)|[\:\.\-]+))/gi, '\n');

  // Bold Assertion and Reason labels if not already wrapped in bold
  s = s.replace(/(?<!\*\*)\b(Assertion(?:\s*\([aA]\))?\s*[:\.\-])(?!\*\*)/gi, '**$1**');
  s = s.replace(/(?<!\*\*)\b(Reason(?:\s*\([rR]\))?\s*[:\.\-])(?!\*\*)/gi, '**$1**');
  s = s.replace(/(?<!\*\*)\b(\u0d2a\u0d4d\u0d30\u0d38\u0d4d\u0d24\u0d3e\u0d35\u0d28(?:\s*\([aA]\))?\s*[:\.\-])(?!\*\*)/gi, '**$1**');
  s = s.replace(/(?<!\*\*)\b(\u0d15\u0d3e\u0d30\u0d23\u0d02(?:\s*\([rR]\))?\s*[:\.\-])(?!\*\*)/gi, '**$1**');

  // Ensure Roman numerals like (i), (ii), (iii), (iv), (v) or i), ii) have a newline before them
  s = s.replace(/(?<=[^\n])\s+(\([ivxlcdmIVXLCDM]+\))\s*/g, '\n$1 ');
  s = s.replace(/(?<=[?.!:;])\s*([ivxlcdmIVXLCDM]+\))\s*/g, '\n$1 ');

  // Ensure numbered sub-statements like (1), (2), (3) or (a), (b), (c) after sentence endings have newlines
  s = s.replace(/(?<=[?.!:;])\s*(\([1-9a-dA-D]\))\s*/g, '\n$1 ');

  // Trim extra blank lines (maximum 2 consecutive newlines)
  s = s.replace(/\n{3,}/g, '\n\n');

  return s.trim();
}

function plainClassificationText(value: unknown): string {
  return String(value ?? '')
    .replace(/<(?:strong|b|em|i|mark)\b[^>]*>([\s\S]*?)<\/(?:strong|b|em|i|mark)>/gi, '$1')
    .replace(/<[^>]+>/g, '')
    .replace(/(?<!\w)(?:\*{1,3}|_{1,3})([^\n]*?)(?:\*{1,3}|_{1,3})(?!\w)/g, '$1')
    .trim();
}

export function isMcq(q: any): boolean {
  if (!q) return false;
  
  const healed = healAssertionReasonQuestion(q);
  const opts: Record<string, any> = (healed && healed.options && Object.keys(healed.options).length > 0) ? healed.options : (q.options || {});
  const hasOptions = Object.keys(opts).length >= 2;

  // 1. Strict priority to backend / LLM classification if explicitly confirmed
  // BUT only trust it when it makes sense with the actual data:
  // - Trust 'mcq' ONLY if options actually exist (otherwise show nothing = bug)  
  // - Trust 'descriptive_*' ONLY if no mcq-like options with marked answer exist
  if (q.question_type) {
    const qt = String(q.question_type).toLowerCase().trim();
    if (qt === 'mcq') {
      // A persisted MCQ label can be stale when A/B are actually directive
      // subquestions. Validate the content before trusting the label.
      if (q.sub_parts && isValidSubparts(q.sub_parts, q.stem)) return false;
      const candidateParts = q.options && isValidSubparts(q.options, q.stem) ? q.options : null;
      if (candidateParts) {
        const values = Object.values(candidateParts).map(v => plainClassificationText(v).toLowerCase());
        const hasDirective = values.some(v => /^(?:define|explain|calculate|draw|state|derive|determine|find|show|prove|write|what|why|how|which|describe|compare|evaluate|obtain)\b/i.test(v));
        const hasMarks = values.some(v => /\b\d+\s*marks?\b|\[\s*\d+\s*marks?/i.test(v));
        if (hasDirective || hasMarks) return false;
      }
      // Only trust MCQ classification if options actually exist
      if (hasOptions) return true;
      // If options are empty despite mcq classification, fall through to check
    }
    if (qt === 'descriptive_subquestions') return false;
    if (qt === 'descriptive_single') {
      // Only trust if no options with a marked answer (which would be an MCQ)
      const marked = (q.marked_correct_answer || q.marked_answer || '').toString().toUpperCase().trim();
      if (!hasOptions || !marked) return false;
      // Has options + marked answer -> might be a miscategorized MCQ, fall through
    }
  }

  // 2. If sub_parts exist and are populated with genuine subquestions, it is descriptive
  if (q.sub_parts && isValidSubparts(q.sub_parts, q.stem)) return false;

  if (!hasOptions) return false;

  // If marked answer matches an option, it's an MCQ!
  const marked = (q.marked_correct_answer || q.marked_answer || '').toString().toUpperCase().trim();
  const keys = Object.keys(opts).map(k => k.toUpperCase().trim()).sort();
  if (marked && keys.includes(marked)) return true;


  // Check if stem is a directive command asking subtasks (e.g. "Explain how ... when:") without an answer key
  const stemClean = plainClassificationText(q.stem);
  const hasDirectiveStem = /^(?:explain|state|describe|calculate|find|define|show|prove|derive|write|determine|what|why|how|distinguish|differentiate|obtain|evaluate|name|give|list|mention|compare|വിശദീകരിക്കുക|കണക്കാക്കുക|എഴുതുക|നിർവചിക്കുക|കണ്ടെത്തുക|തെളിയിക്കുക)\b/i.test(stemClean) || stemClean.endsWith(':');
  if (hasDirectiveStem && keys.length <= 3 && isValidSubparts(opts, stemClean)) return false;

  const isStandardMcqKeys = (
    (keys.length === 4 && keys.join(',') === 'A,B,C,D') ||
    (keys.length === 3 && keys.join(',') === 'A,B,C') ||
    (keys.length === 5 && keys.join(',') === 'A,B,C,D,E') ||
    (keys.length === 2 && keys.join(',') === 'A,B') ||
    (keys.length === 4 && keys.join(',') === '1,2,3,4') ||
    (keys.length === 3 && keys.join(',') === '1,2,3') ||
    (keys.length === 2 && keys.join(',') === '1,2') ||
    (keys.length === 4 && keys.join(',') === 'I,II,III,IV') ||
    (keys.length === 3 && keys.join(',') === 'I,II,III')
  );

  const isDigitMcq = keys.every(k => ['1', '2', '3', '4', '5'].includes(k)) && keys.includes('1') && keys.length >= 2;
  const isLetterMcq = keys.every(k => ['A', 'B', 'C', 'D', 'E'].includes(k)) && (keys.includes('A') || keys.includes('B')) && keys.length >= 2;
  const isRomanMcq = keys.every(k => ['I', 'II', 'III', 'IV', 'V'].includes(k)) && keys.includes('I') && keys.length >= 2;

  const isImperfectMcqKeys = isDigitMcq || isLetterMcq || isRomanMcq;

  if (!isStandardMcqKeys && !isImperfectMcqKeys) return false;

  // 3. Inspect option content for genuine sub-question signals (imperative command starters, marks, question marks)
  let subQCount = 0;
  let hasQuestionMark = false;
  let hasMarksIndicator = false;

  for (const v of Object.values(opts)) {
    const text = String(v ?? '').trim();
    const textLower = text.toLowerCase();
    if (text.includes('?')) hasQuestionMark = true;
    if (/\[\s*\d+\s*marks?\s*\]|\(\s*\d+\s*marks?\s*\)|\b\d+\s*marks?\b/i.test(text)) hasMarksIndicator = true;
    if (/^\s*(?:draw|calculate|find|explain|state|prove|derive|show\s+that|determine|define|describe|why|what|how|when|where|which|mention|list|compare|differentiate|distinguish|obtain|verify|establish|evaluate|solve|write|name|give)\b/i.test(textLower)) {
      subQCount++;
    }
  }

  // If options have question marks, marks brackets, or multiple directive prompts, treat as descriptive subquestions
  if ((hasQuestionMark || hasMarksIndicator) && subQCount >= 1) return false;
  if (subQCount >= 2) return false;

  // If options consist only of equations (e.g. no prose/commands), they are likely MCQ options or parts, but since we verified MCQ keys, treat as MCQ.
  return true;
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

/**
 * Strictly verifies whether a dictionary of sub-parts contains genuine sub-questions.
 * Rejects empty sub-questions, marks-only sub-questions, non-consecutive letters,
 * or false splits from numbers/units (e.g. "0.2 A." or "points A and B.").
 */
function isValidSubparts(subparts: Record<string, any> | null | undefined, stem?: string): boolean {
  if (!subparts || typeof subparts !== 'object') return false;
  const entries = Object.entries(subparts);
  if (entries.length < 2) return false;

  const keys = Object.keys(subparts).map(k => k.toLowerCase().trim());
  const firstK = keys[0];
  const secondK = keys[1];

  // Must start with valid sub-question marker ('a', '1', or 'i')
  if (!['a', '1', 'i'].includes(firstK)) return false;

  // Second key must be strictly consecutive
  if (firstK === 'a' && secondK !== 'b') return false;
  if (firstK === '1' && secondK !== '2') return false;
  if (firstK === 'i' && secondK !== 'ii') return false;

  const engDirectives = [
    'define', 'explain', 'calculate', 'draw', 'state', 'derive', 'determine',
    'find', 'show', 'prove', 'distinguish', 'differentiate', 'obtain',
    'evaluate', 'name', 'give', 'write', 'what', 'why', 'how', 'which',
    'where', 'when', 'who', 'compare', 'list', 'mention', 'describe',
    'illustrate', 'sketch', 'identify', 'plot', 'compute', 'verify'
  ];
  const malDirectives = [
    'എഴുതുക', 'വിശദീകരിക്കുക', 'കണക്കാക്കുക', 'വരയ്ക്കുക', 'നിർവചിക്കുക',
    'കണ്ടെത്തുക', 'തെളിയിക്കുക', 'വ്യത്യാസ', 'രൂപീകരിക്കുക', 'പ്രസ്താവിക്കുക',
    'താരതമ്യം', 'പട്ടികപ്പെടുത്തുക', 'സൂചിപ്പിക്കുക', 'എന്താണ്', 'എന്ത്',
    'എങ്ങനെ', 'എവിടെ', 'എപ്പോൾ', 'ഏതാണ്', 'വിവരിക്കുക', 'തരംതിരിക്കുക'
  ];

  let stemHasDirective = false;
  if (stem && typeof stem === 'string') {
    const s = stem.trim().toLowerCase();
    stemHasDirective = /^(?:explain|state|describe|calculate|find|define|show|prove|derive|write|determine|what|why|how|distinguish|differentiate|obtain|evaluate|name|give|list|mention|compare|വിശദീകരിക്കുക|കണക്കാക്കുക|എഴുതുക|നിർവചിക്കുക|കണ്ടെത്തുക|തെളിയിക്കുക)\b/i.test(s) || s.endsWith(':') || s.endsWith('when:');
  }

  let hasQuestionFeature = stemHasDirective;

  for (const [_, v] of entries) {
    const val = plainClassificationText(v);
    // Any empty sub-part means it's NOT a valid set of sub-questions
    if (!val) return false;

    const valLower = val.toLowerCase();

    // If an item contains explicit answer markers or comparative states without any directive, it is an answer step!
    const isPureAnswerVal = /^(?:higher|lower|increases?|decreases?|remains constant|constant|zero|positive|negative|forward bias|reverse bias|same|remains same|no change)[\.\s]*$/i.test(valLower);
    const hasAnswerHeader = /answer:|ans:|solution:|sol:|hence|therefore|we get/i.test(valLower);
    const isAnswerStep = isPureAnswerVal || hasAnswerHeader;

    // Check for question features: question mark, marks allocation, or directive verbs
    const hasQMark = val.includes('?');
    const hasMarks = /\[\s*\d+\s*marks?\s*\]|\(\s*\d+\s*marks?\s*\)|\b\d+\s*marks?\b/i.test(val);
    const hasDirective = engDirectives.some(d => valLower.includes(d)) || malDirectives.some(d => valLower.includes(d));

    if (hasQMark || hasMarks || hasDirective) {
      hasQuestionFeature = true;
    } else if (isAnswerStep) {
      // Answer steps are NEVER subquestions
      return false;
    }

    // Strip marks indicators, images, and whitespace/punctuation to test for actual question text
    let substantive = val
      .replace(/\[\s*\d+\s*marks?\s*\]|\(\s*\d+\s*marks?\s*\)|\b\d+\s*marks?\b/gi, '')
      .replace(/<img[^>]*>/gi, '')
      .replace(/[\s\.\:\-\,\;]/g, '');

    // Check if it's a math equation or formula
    const isMath = val.includes('$') || val.includes('\\') || /[\=\+\-\*\/\^\_\±\×\÷\√]/.test(val);

    // Every subpart MUST have actual question text or formula (at least 2 letters/symbols unless it's math)
    if (substantive.length < 2 && !isMath) return false;
  }
  return hasQuestionFeature;
}

export function extractSubpartsFromText(text?: string): Record<string, string> | null {
  if (!text || typeof text !== 'string') return null;

  // Truncate at answer/solution boundary so answers are NEVER extracted as subquestions!
  const ansIdx = text.search(/(?:\n|\s{2,}|\t)\*?\*?(?:Correct(?:\s*Option|\s*Answer)?|Answer|Ans(?:wer)?|Solution|Explanation|Description|Hint|Hints|ഉത്തരം|ശരിയായ\s*ഉത്തരം|വിശദീകരണം|പരിഹാരം)\s*(?:[\*\:\-]+|\s*$)/i);
  const stemOnly = plainClassificationText(ansIdx !== -1 ? text.substring(0, ansIdx) : text);

  // Matches markers like (a), (b) or (1), (2) or (i), (ii), or bare letters a., b., a), b)
  const subqPattern = /(?:(?:^|\n|\s{2,}|\.\s+)\(([a-eA-E1-4]|[ivxlcdmIVXLCDM]+)\)\s*)|(?:(?:^|\n|[ \t]{2,}|\.\s+)([a-eA-E]|[ivxlcdmIVXLCDM]+)[\.\)\:-]\s*(?!\d))/g;
  const matches = [...stemOnly.matchAll(subqPattern)];
  if (matches.length < 2) return null;

  // Filter candidate matches: reject if preceded by digits (e.g. scientific unit "0.2 A.") or point references
  const validMatches = matches.filter(m => {
    const idx = m.index ?? 0;
    const prefix = stemOnly.substring(Math.max(0, idx - 35), idx).toLowerCase();
    const hasParen = Boolean(m[1]);
    const postSnippet = stemOnly.substring(idx + m[0].length, idx + m[0].length + 10).trim();

    // Reject if followed by equation operator or equal sign (e.g. A = 240, A = 120, A + B)
    if (/^[\=\+\-\*\/\^\_\±\×\÷\√]/.test(postSnippet)) return false;

    if (/(?:assertion|reason|statement|പ്രസ്താവന|കാരണം)[\s\:\-\*\_\(]*$/i.test(prefix)) return false;
    if (/(?:both|between|from|to|and|or|is|are|was|were|if|whether|that|when|where|of|in|for|with|junction|terminal|point|points|cell|cells|charge|charges|plate|plates|number|mass|each|fragment|fragments|side|radius|amplitude|wavelength|frequency|constant|variable)\s*(?:and\s*)?$/i.test(prefix)) return false;
    if (!hasParen && /\d+(?:\.\d+)?\s*$/.test(prefix)) return false;
    return true;
  });
  if (validMatches.length < 2) return null;

  const keys = validMatches.map(m => (m[1] || m[2]).toLowerCase());
  const firstK = keys[0];
  if (!['a', '1', 'i'].includes(firstK)) return null;

  const isSeq = (firstK === 'a' && keys[1] === 'b') || 
                (firstK === 'i' && keys[1] === 'ii') || 
                (firstK === '1' && keys[1] === '2');
  if (!isSeq) return null;

  const subparts: Record<string, string> = {};
  for (let i = 0; i < validMatches.length; i++) {
    const k = (validMatches[i][1] || validMatches[i][2]).toLowerCase();
    const start = validMatches[i].index! + validMatches[i][0].length;
    const end = i + 1 < validMatches.length ? validMatches[i + 1].index! : stemOnly.length;
    let partText = stemOnly.substring(start, end).trim();
    if (partText) {
      subparts[k] = partText;
    }
  }
  return isValidSubparts(subparts) ? subparts : null;
}

export function getIntroductoryStem(stem: string, hasSubparts: boolean, firstKey?: string): string {
  if (!stem) return '';
  const clean = formatQuestionStem(stem);
  if (!hasSubparts) return clean;

  let m: RegExpMatchArray | null = null;
  if (firstKey) {
    const k = firstKey.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const pat = new RegExp(`(?:(?:^|\\n|\\s{2,}|\\.\\s+)\\([${k}${k.toUpperCase()}]\\)\\s*)|(?:(?:^|\\n|[ \\t]{2,}|\\.\\s+)[${k}${k.toUpperCase()}][\\.\\)\\:-]\\s*)`);
    m = clean.match(pat);
  } else {
    m = clean.match(/(?:(?:^|\n|\s{2,}|\.\s+)\(([aA1iI])\)\s*)|(?:(?:^|\n|[ \t]{2,}|\.\s+)([aA1iI])[\.\)\:-]\s*)/);
  }

  if (!m) return clean;

  const idx = m.index ?? 0;
  const prefix = clean.substring(Math.max(0, idx - 35), idx).toLowerCase();
  const hasParen = Boolean(m[1]);
  if (!hasParen && (/\d+(?:\.\d+)?\s*$/.test(prefix) || /(?:points?|between|junction|terminal|from|cells?|charges?|plates?)\s*(?:and\s*)?$/i.test(prefix))) {
    return clean;
  }

  const intro = clean.substring(0, m.index).trim();
  // If the intro is empty or only question prefix like "Question 9:" or punctuation, return empty
  if (!intro || /^(?:question\s*\d*|q\d*|ചോദ്യം\s*\d*|[\d\.\:\-\s])*$/i.test(intro)) {
    return '';
  }
  return intro;
}

export function splitMergedOptionsFrontend(options: Record<string, string>): Record<string, string> {
  if (!options || typeof options !== 'object') return options || {};
  const out: Record<string, string> = {};
  const keys = Object.keys(options);

  for (const k of keys) {
    let currentK = k;
    let currentText = String(options[k] || '').trim();

    while (true) {
      let candChars: string[] = [];
      let bareChars: string[] = [];
      if (/^\d+$/.test(currentK)) {
        const nextVal = parseInt(currentK, 10) + 1;
        if (nextVal <= 8) {
          candChars = [String(nextVal)];
          bareChars = candChars;
        }
      } else if (currentK.length === 1 && /[a-zA-Z]/.test(currentK)) {
        const nextCharCode = currentK.toUpperCase().charCodeAt(0) + 1;
        if (nextCharCode <= 72) { // up to 'H'
          const nextChar = String.fromCharCode(nextCharCode);
          candChars = [nextChar, nextChar.toLowerCase()];
          bareChars = /^[A-Z]/.test(currentK) ? [nextChar] : [nextChar.toLowerCase()];
        }
      }

      if (candChars.length === 0) {
        out[currentK] = currentText;
        break;
      }

      const parenGroup = candChars.map(c => c.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
      const bareGroup = bareChars.map(c => c.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
      const bareBranch = bareGroup ? `|([${bareGroup}])[\\.\\)\\:\\-]\\s+` : '';
      const pat = new RegExp(`(?:(?<=\\s)|(?<=\\n)|^)(?:\\(([${parenGroup}])\\)[\\.\\:\\-]?\\s*|\\[([${parenGroup}])\\][\\.\\:\\-]?\\s*${bareBranch})`);
      const m = currentText.match(pat);
      if (m && m.index !== undefined && m.index > 0) {
        const splitIdx = m.index;
        const firstPart = currentText.substring(0, splitIdx).trim();
        if (!firstPart) {
          out[currentK] = currentText;
          break;
        }
        const matchedChar = m[1] || m[2] || m[3];
        const nextK = /^[A-Z]/.test(currentK) ? matchedChar.toUpperCase() : matchedChar.toLowerCase();
        const secondPart = currentText.substring(m.index + m[0].length).trim();

        out[currentK] = firstPart;
        currentK = nextK;
        currentText = secondPart;
      } else {
        out[currentK] = currentText;
        break;
      }
    }
  }
  return out;
}

export function extractOptionsFromStemFrontend(content: string): { stem: string; options: Record<string, string> } | null {
  if (!content || typeof content !== 'string') return null;

  const optPattern = /(?:(?:^|\n|\s+)\(([A-Ea-e1-4])\)[\.\:\-]?\s*)|(?:(?:^|\n|[ \t]+|\.\s+)([A-Ea-e1-4])[\.\)\:-]\s*(?!\d))/g;
  const matches = [...content.matchAll(optPattern)];
  if (matches.length < 2) return null;

  const validMatches = matches.filter(m => {
    const startIdx = m.index ?? 0;
    const prefix = content.substring(Math.max(0, startIdx - 35), startIdx).toLowerCase();
    const fullMatch = m[0].trim();
    const hasParen = fullMatch.startsWith('(');
    const postSnippet = content.substring(startIdx + m[0].length, startIdx + m[0].length + 10).trim();

    // Reject if followed by equation operator or equal sign (e.g. A = 240, A = 120, A + B)
    // CRITICAL: Parenthesized options like (d) \sqrt{c}: 1 or (b) +5 V are completely valid!
    if (!hasParen && /^[\=\+\-\*\/\^\_\±\×\÷\√]/.test(postSnippet)) return false;
    if (hasParen && postSnippet.startsWith('=')) return false;

    // Reject Assertion/Reason prefixes
    if (/(?:assertion|reason|statement|പ്രസ്താവന|കാരണം)[\s\:\-\*\_\(]*$/i.test(prefix)) return false;

    // Reject prepositions, linkers, and math variables
    const prevWordMatch = prefix.match(/([a-zA-Z]+)\s*(?:and\s*)?$/);
    if (prevWordMatch) {
      const prevWord = prevWordMatch[1].toLowerCase();
      if (!hasParen) {
        const rejectWords = new Set(['both', 'between', 'from', 'and', 'or', 'is', 'are', 'was', 'were',
          'if', 'whether', 'that', 'when', 'where', 'of', 'in', 'for', 'with',
          'junction', 'terminal', 'point', 'points', 'cell', 'cells',
          'charge', 'charges', 'plate', 'plates', 'to', 'number', 'mass',
          'each', 'fragment', 'fragments', 'side', 'radius', 'amplitude']);
        if (rejectWords.has(prevWord)) return false;
      } else {
        const structuralReject = new Set(['point', 'points', 'junction', 'terminal', 'cell', 'cells', 'number', 'mass']);
        if (structuralReject.has(prevWord)) return false;
      }
    }

    if (!hasParen && /\d+(?:\.\d+)?\s*$/.test(prefix)) return false;
    return true;
  });

  if (validMatches.length < 2) return null;

  let bestSeq: RegExpMatchArray[] = [];
  for (let i = 0; i < validMatches.length; i++) {
    const letter = (validMatches[i][1] || validMatches[i][2]).toUpperCase();
    if (letter !== 'A' && letter !== '1') continue;
    const currSeq = [validMatches[i]];
    let currLetter = letter;

    for (let j = i + 1; j < validMatches.length; j++) {
      const nextLetter = (validMatches[j][1] || validMatches[j][2]).toUpperCase();
      if (nextLetter > currLetter) {
        const letterGap = nextLetter.charCodeAt(0) - currLetter.charCodeAt(0);
        if (letterGap <= 2) {
          currSeq.push(validMatches[j]);
          currLetter = nextLetter;
        }
      }
    }

    if (currSeq.length >= 2 && currSeq.length > bestSeq.length) {
      bestSeq = currSeq;
    }
  }

  if (bestSeq.length < 2) return null;

  const stem = content.substring(0, bestSeq[0].index).trim();
  const options: Record<string, string> = {};

  for (let idx = 0; idx < bestSeq.length; idx++) {
    const mInfo = bestSeq[idx];
    const letter = (mInfo[1] || mInfo[2]).toUpperCase();
    const startPos = (mInfo.index ?? 0) + mInfo[0].length;
    const endPos = idx < bestSeq.length - 1 ? (bestSeq[idx + 1].index ?? content.length) : content.length;
    let optText = content.substring(startPos, endPos).trim();

    if (idx === bestSeq.length - 1) {
      const ansIdx = optText.search(/(?:\n|\s{2,}|\t)\*?\*?(?:Correct(?:\s*Option|\s*Answer)?|Answer|Ans(?:wer)?|Solution|Explanation|Description|Hint|Hints|ഉത്തരം|ശരിയായ\s*ഉത്തരം|വിശദീകരണം)\s*(?:[\*\:\-]+|\s*$)/i);
      if (ansIdx !== -1) {
        optText = optText.substring(0, ansIdx).trim();
      }
    }
    options[letter] = optText;
  }

  return { stem, options };
}

export function healAssertionReasonQuestion(q: any): { stem: string; options: Record<string, string> } {
  let stem = q?.stem || '';
  let options = q?.options ? { ...q.options } : {};

  // 0. Auto-heal: If existing options contain invalid keys like '7', '8', or numbers > 5, strip them and restore stem
  const currentKeys = Object.keys(options);
  const hasInvalidKeys = currentKeys.some(k => /^[6-9]$/.test(k) || (!['A', 'B', 'C', 'D', 'E', '1', '2', '3', '4', '5'].includes(k.toUpperCase()) && !/^[IVX]+$/i.test(k)));
  if (hasInvalidKeys) {
    for (const k of Object.keys(options).sort((a, b) => (parseInt(a) || 0) - (parseInt(b) || 0))) {
      stem = `${stem} ${k}.${options[k]}`.trim();
    }
    options = {};
  }

  // 1. Split any merged options in existing options
  if (options && Object.keys(options).length > 0) {
    options = splitMergedOptionsFrontend(options);
  }

  // Strip trailing marks bracket from stem first
  stem = stem.replace(/\s*(?:\[\s*\d+(?:\.\d+)?(?:\s*\/\s*\d+)?\s*(?:marks?|mark|m)\s*\]|\(\s*\d+(?:\.\d+)?(?:\s*\/\s*\d+)?\s*(?:marks?|mark|m)\s*\))\s*$/i, '').trim();

  // 1.5. Trailing option extraction from stem (ONLY if options are non-empty and incomplete)
  // If options is empty, a descriptive question with math equations must NEVER have pieces sliced off!
  if (Object.keys(options).length > 0 && Object.keys(options).length < 4) {
    const trailingOptPat = /(?:(?:\n|\s+|\.\s+)\(([A-Ea-e1-4])\)[\.\:\-]?\s*|(?:\n|\s{2,}|\.\s+)([A-Ea-e])[\.\)\:-]\s*(?!\d))([^ \n\r\t][^\n\r]*)$/i;
    while (true) {
      const m = stem.match(trailingOptPat);
      if (!m || m.index === undefined) break;
      const letter = (m[1] || m[2]).toUpperCase();
      const val = m[3].trim();
      const prefix = stem.substring(Math.max(0, m.index - 35), m.index).toLowerCase();

      const hasParen = Boolean(m[1]);
      // Reject if val starts with equation operator or equal sign (e.g. = 240, = 120)
      // Parenthesized options like (d) \sqrt{c}: 1 or (b) +5 V are completely valid!
      if (!hasParen && /^[\=\+\-\*\/\^\_\±\×\÷\√]/.test(val)) break;
      if (hasParen && val.startsWith('=')) break;

      // Reject if preceded by Assertion / Reason / Statement labels or math variable words
      if (/(?:assertion|reason|statement|പ്രസ്താവന|കാരണം|point|points|junction|terminal|cell|cells|charge|charges|plate|plates|number|mass|each|fragment|fragments|side|radius|amplitude)[\s\:\-\*\_\(]*$/i.test(prefix)) break;
      // Reject if val contains Reason / Assertion label
      if (/\b(?:reason|കാരണം|assertion|പ്രസ്താവന)\b/i.test(val)) break;
      // If option already exists and is non-empty, never slice or overwrite
      if (options[letter] && options[letter].trim()) break;

      stem = stem.substring(0, m.index).trim();
      stem = stem.replace(/\s*(?:\[\s*\d+(?:\.\d+)?(?:\s*\/\s*\d+)?\s*(?:marks?|mark|m)\s*\]|\(\s*\d+(?:\.\d+)?(?:\s*\/\s*\d+)?\s*(?:marks?|mark|m)\s*\))\s*$/i, '').trim();
      options[letter] = val;
    }
  }

  // 1.6. Recover answer choice from explanation / answer_text if missing from options
  const exp = q?.explanation || q?.answer_text;
  if (exp && typeof exp === 'string') {
    const ansMatch = exp.match(/(?:^|\n|\s{2,}|\b)(?:\*\*\s*(?:Answer|Ans(?:wer)?|Correct(?:\s*Option|\s*Answer)?|ഉത്തരം)\s*\*\*|\*?\*?(?:Answer|Ans(?:wer)?|Correct(?:\s*Option|\s*Answer)?|ഉത്തരം)\*?\*?\s*[\:\.\-]+)\s*[\(\[]?\s*([A-Da-d1-4])\s*[\)\]]?[\s\:\.\-]*([^\n]*)/i);
    if (ansMatch) {
      const ansLetter = ansMatch[1].toUpperCase();
      const ansText = ansMatch[2].trim();
      if (ansText && !options[ansLetter]) {
        options[ansLetter] = ansText;
      }
    } else {
      const leadingM = exp.match(/^\s*[\(\[]\s*([A-Da-d1-4])\s*[\)\]][\s\:\.\-]*([^\n]*)/i);
      if (leadingM) {
        const ansLetter = leadingM[1].toUpperCase();
        const ansText = leadingM[2].trim();
        if (ansText && !options[ansLetter]) {
          options[ansLetter] = ansText;
        }
      }
    }
  }

  // 2. Check for Assertion/Reason format where Reason is bundled inside Option A
  const optA = options.A || options.a || '';
  if (optA && typeof optA === 'string') {
    const hasReason = /\b(?:reason|കാരണം)\s*(?:\([rR]\)|[\:\.\-]+|\s*$)/i.test(optA);
    const stemClean = stem.trim();
    const stemIsShortAssertion = /\b(?:assertion|പ്രസ്താവന)\b/i.test(stemClean) && stemClean.replace(/<[^>]+>/g, '').trim().length < 120;

    if (hasReason || stemIsShortAssertion) {
      const splitMatch = optA.match(/(?:(?:^|\n|\s{2,}|\.\s+)\(?([aA1])\)?[\.\:\-]?\s+(?=(?:if\s+both|both|assertion|statement|[aA]\s+is\b)))|(?:(?:^|\n)\s*(?=(?:if\s+both|both\s+(?:assertion|a\b|r\b)|assertion\s+is\s+true|[aA]\s+is\s+true)))/i);
      if (splitMatch && splitMatch.index !== undefined) {
        const splitIdx = splitMatch.index;
        let statementBlock = optA.substring(0, splitIdx).trim();
        let choiceText = optA.substring(splitIdx).trim().replace(/^\s*\(?[aA1]\)?[\.\:\-]?\s*/, '').trim();

        statementBlock = statementBlock.replace(/\s*\[\s*\d+(?:\.\d+)?(?:\s*\/\s*\d+)?\s*(?:marks?|mark|m)\s*\]/gi, '');
        statementBlock = statementBlock.replace(/\s*\(\s*\d+(?:\.\d+)?(?:\s*\/\s*\d+)?\s*(?:marks?|mark|m)\s*\)/gi, '').trim();

        if (statementBlock) {
          let newStem = stemClean;
          if (!stemClean || /^(?:assertion|പ്രസ്താവന)\s*[\:\.\-]?$/i.test(stemClean)) {
            newStem = `Assertion (A): ${statementBlock}`;
          } else if (!/\b(?:reason|കാരണം)\b/i.test(stemClean)) {
            newStem = `${stemClean}\n${statementBlock}`;
          }
          const keyA = options.A !== undefined ? 'A' : 'a';
          options[keyA] = choiceText;
          return { stem: newStem, options: splitMergedOptionsFrontend(options) };
        }
      }
    }
  }

  // 3. If options are empty or incomplete (< 4), attempt to recover options from stem ONLY if stem is an MCQ
  const optCount = Object.keys(options).length;
  if (optCount === 0 && stem) {
    const recovered = extractOptionsFromStemFrontend(stem);
    if (recovered && Object.keys(recovered.options).length >= 3) {
      const recKeys = Object.keys(recovered.options).map(k => k.toUpperCase());
      const isCleanSeq = recKeys.join(',') === 'A,B,C,D' || recKeys.join(',') === 'A,B,C' || recKeys.join(',') === '1,2,3,4' || recKeys.join(',') === '1,2,3';
      if (isCleanSeq) {
        return { stem: recovered.stem, options: splitMergedOptionsFrontend(recovered.options) };
      }
    }
  } else if (optCount > 0 && optCount < 4 && stem) {
    const recovered = extractOptionsFromStemFrontend(stem);
    if (recovered && Object.keys(recovered.options).length >= 2 && Object.keys(recovered.options).length > optCount) {
      return { stem: recovered.stem, options: splitMergedOptionsFrontend(recovered.options) };
    }
  }

  return { stem, options };
}

function cleanOptionDisplay(text: string, optionKey?: string): string {
  if (!text) return '';
  let s = text.trim();
  // Strip leading option prefix like (a), a., a:
  // CRITICAL: Must be in parens/brackets or followed by delimiter/space.
  // NEVER strip 'B' from 'Both' or 'A' from 'All'!
  if (optionKey) {
    const k = optionKey.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const pat = new RegExp(`^\\s*(?:\\([\\s]*[${k}${k.toLowerCase()}][\\s]*\\)[\\.\\:\\-]?|\\[[\\s]*[${k}${k.toLowerCase()}][\\s]*\\][\\.\\:\\-]?|[${k}${k.toLowerCase()}]\\)[\\.\\:\\-]?\\s*)\\s*`);
    s = s.replace(pat, '').trim();
  } else {
    s = s.replace(/^\s*(?:\([\s]*[a-dA-D][\s]*\)[\.\:\-]?|\[[\s]*[a-dA-D][\s]*\][\.\:\-]?|[a-dA-D]\)[\.\:\-]?\s*)\s*/, '').trim();
  }
  // Auto-heal truncated words anywhere in options
  s = s.replace(/\bOth\b/g, 'Both').replace(/\both\b/g, (match, offset) => {
    return (offset === 0 || /(?:^|[.!?\n]\s*)$/.test(s.slice(0, offset))) ? 'Both' : 'both';
  });
  s = s.replace(/\bll\s+of\s+the\s+above\b/gi, 'All of the above');

  // Strip marks indicators from option text (e.g. '[1 mark]', '(2 marks)', '[5 marks]')
  s = s.replace(/\s*\[\s*\d+(?:\.\d+)?(?:\s*\/\s*\d+)?\s*(?:marks?|mark|m)\s*\]/gi, '');
  s = s.replace(/\s*\(\s*\d+(?:\.\d+)?(?:\s*\/\s*\d+)?\s*(?:marks?|mark|m)\s*\)/gi, '');
  s = s.trim();

  // Strip trailing answer or solution markers if accidentally present (requires delimiter or end-of-string)
  const ansIdx = s.search(/(?:\n|\s{2,}|\t)\*?\*?(?:Correct(?:\s*Option|\s*Answer)?|Answer|Ans(?:wer)?|Solution|Explanation|Description|Hint|Hints|ഉത്തരം|ശരിയായ\s*ഉത്തരം|വിശദീകരണം)\s*(?:[\*\:\-]+|\s*$)/i);
  if (ansIdx !== -1) {
    s = s.substring(0, ansIdx).trim();
  }
  // (Removed bare \text{} stripping)
  // Heal scientific units and fraction slashes (e.g. "1.5 N C⁄" -> "1.5 N/C")
  s = healUnitsAndSlashes(s);
  return s;
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

    const words = (raw.match(/[a-zA-Z]{2,}/g) || []).map(w => w.toLowerCase());
    const proseMatches = words.filter(w => commonProseWords.some(p => w.includes(p)));

    let isProse = false;
    if (hasIndic && !raw.includes('\\text')) {
      isProse = true;
    } else if (proseMatches.length >= 2 || raw.toLowerCase().includes('planck') || raw.includes("'s")) {
      isProse = true;
    } else if (proseMatches.length === 1 && !/[=+\\_^]/.test(raw)) {
      isProse = true;
    }

    if (!isProse) return match;

    // Un-mash prose
    let unmashed = raw;
    if (!unmashed.includes(' ') || (unmashed.match(/\s+/g) || []).length < proseMatches.length) {
      unmashed = unmashed.replace(/^is(?=[A-Z])/g, 'is ');
      unmashed = unmashed.replace(/([a-z])([A-Z])/g, '$1 $2');
      unmashed = unmashed.replace(/([a-zA-Z]+'s)([a-zA-Z]+)/g, '$1 $2');
      unmashed = unmashed.replace(/([a-zA-Z]+)(and)([a-zA-Z]+)/g, '$1 and $3');
      unmashed = unmashed.replace(/\b(constant)([a-zA-Z]+)/g, '$1 $2');
      unmashed = unmashed.replace(/\b(charge)([a-zA-Z]+)/g, '$1 $2');
      unmashed = unmashed.replace(/\s{2,}/g, ' ').trim();
    }

    // Wrap single letter variables
    unmashed = unmashed.replace(/(?<!['\w\\])\b([b-hj-zB-HJ-Z])\b/g, '$$$1$');
    return unmashed;
  };

  s = s.replace(/\$\$([\s\S]+?)\$\$/g, fixMathProseBlock);
  s = s.replace(/\\\[([\s\S]+?)\\\]/g, fixMathProseBlock);
  s = s.replace(/(?<!\$)\$(?!\$)([^\$\r\n]+?)(?<!\$)\$(?!\$)/g, fixMathProseBlock);
  s = s.replace(/\\\(([\s\S]+?)\\\)/g, fixMathProseBlock);

  // Clean up mid-sentence line breaks created by unwrapped display blocks
  s = s.replace(/(\$[a-zA-Z]\$|[a-zA-Z,])\s*\n\s*([a-z])/g, '$1 $2');

  s = healSeveredAndSwallowedMath(s);

  return s;
}

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

function cleanExplanationDisplay(text: string, markedAnswer?: string, isMcqQuestion?: boolean): string {
  if (!text) return '';
  let s = text.trim();
  s = sanitizeHtmlAndCleanText(s);

  // 1. Strip redundant leading "Solution:" or "Explanation:" label FIRST so subpart detection works on actual content
  s = s.replace(/^(?:Solution|Explanation|Description|Sol|Exp|Hint|Answer|Ans|ഉത്തരം|വിശദീകരണം|പരിഹാരം|വിവരണം|സൂചന)\s*(?:[\*\:\-]+(?:\s*|\n|$)|(?:\n|\\n))/i, '').trim();

  // 2. Detect subparts in text: (b) after prior text, or presence of both (a) and (b)
  const subsequentB = /\S+[\s\S]+?(?:^|\n|\s)(?:\(?[b-hB-H]\)?|[b-hB-H]\.|\(ii\)|\(2\))[\s\:\-\.]+/i.test(s);
  const hasA = /(?:^|\n|\s)\(?[aA1]\)?[\s\.\:\-]+/i.test(s);
  const hasB = /(?:^|\n|\s)(?:\(?[b-hB-H2]\)?|[b-hB-H]\.)[\s\.\:\-]+/i.test(s);
  const isMultiPart = subsequentB || (hasA && hasB);

  const startsWithSubpart = /^\(?[a-dA-D1-9]\)?[\s\.\:\-]+|^\(?[ivxIVX]+\)?[\s\.\:\-]+/i.test(s);

  // 3. Auto-heal missing (a) at start of multi-part answer:
  // e.g. "Rectifier (b) Circuit diagram of full wave rectifier" -> "(a) Rectifier (b) Circuit diagram..."
  if (isMultiPart && !startsWithSubpart) {
    s = `(a) ${s}`;
  }

  // 4. Strip leading answer letter ONLY if this is genuinely an MCQ (and NOT a multi-part answer with subparts)
  if (!isMultiPart && isMcqQuestion !== false) {
    if (markedAnswer) {
      s = s.replace(new RegExp(`^\\(?${markedAnswer}\\)?[\\s\\.\\:\\-]+`, 'i'), '').trim();
    } else if (isMcqQuestion === true) {
      // Only strip generic A-D if explicitly known to be MCQ
      s = s.replace(/^\(?[A-D]\)?[\s\.\:\-]+/i, '').trim();
    }
  }

  // 5. Place inline subparts on newlines for clean multi-part formatting if merged
  s = s.replace(/([^\n])\s*\(([b-hB-H])\)\s+([A-Z\u0D00-\u0D7F\$\\\*0-9])/g, '$1\n($2) $3');

  // 6. Thoroughly strip any accidental "Subquestion:" or "Sub-question:" or "Subpart" labels from answers
  s = s.replace(/(?:^|\n|\s+)\*?\*?Sub-?\s*questions?\s*(?:\(([a-zA-Z0-9]+)\)|([a-zA-Z0-9]+))?[\s\:\-\.]*\*?\*?/gi, (match, p1, p2) => {
    if (p1) return `\n(${p1}) `;
    if (p2) return `\n${p2}. `;
    return '\n';
  });
  s = s.replace(/(?:^|\n|\s+)\*?\*?Sub-?\s*parts?\s*(?:\(([a-zA-Z0-9]+)\)|([a-zA-Z0-9]+))?[\s\:\-\.]*\*?\*?/gi, (match, p1, p2) => {
    if (p1) return `\n(${p1}) `;
    if (p2) return `\n${p2}. `;
    return '\n';
  });
  s = s.replace(/\n{3,}/g, '\n\n').trim();
  // Strip stray font ligature glitches like ~ } ONLY when adjacent to Malayalam characters
  s = s.replace(/([\u0D00-\u0D7F])[~}]+/g, '$1').replace(/[~}]+([\u0D00-\u0D7F])/g, '$1');
  // Heal student readability (unwrap mistakenly wrapped prose, un-mash words and variables)
  s = healStudentReadability(s);
  return s;
}

function normalizeCdnUrls<T>(data: T): T {
  if (!data) return data;
  try {
    let str = JSON.stringify(data);
    if (str.includes('exam.trogon.info')) {
      str = str.replaceAll('exam.trogon.info', 'examcdn.trogon.info');
    }
    // Remove any link showing below the image like <a href="...">...</a>
    str = str.replace(/<a\s+[^>]*>[\s\S]*?<\/a>/gi, '');
    return JSON.parse(str);
  } catch {
    // fallback
  }
  return data;
}

export function renderLatex(text: string): string {
  if (!text) return '';

  let s = text;

  // Convert markdown images to HTML image containers if present
  s = s.replace(/!\[(diagram)\]\((https?:\/\/[^\s\)]+)\)/gi, '<div class="my-3 flex flex-col items-center"><img src="$2" alt="$1" class="max-h-64 rounded shadow-sm border border-slate-200" /></div>');
  s = s.replace(/!\[(math)\]\((https?:\/\/[^\s\)]+)\)/gi, '<img src="$2" alt="$1" class="inline-math-img" style="vertical-align:-0.38em;max-height:2.2em;display:inline-block;margin:0 1px;" />');

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

  // Restore flattened exponents with minus/dash, e.g. 10—15 or 10-15 -> 10^{-15}
  s = s.replace(/(?<=(?:×|x|\\times|\b)\s*10\s*)[—–-]\s*(\d{1,3})(?!\d)/gi, '^{-$1}');

  // Merge base and scientific notation with trailing superscript/subscript into cohesive math blocks:
  // e.g. "14.4 × 10$^{−15}$" -> "$14.4 \times 10^{-15}$", "10$^{−7}$" -> "$10^{-7}$", "q$_{1}$" -> "$q_{1}$"
  s = s.replace(/(\b\d+(?:\.\d+)?)\s*(?:×|x|\\times)\s*10\s*\$\^\{([^}]+)\}\$/gi, (_, p1, p2) => `$${p1} \\times 10^{${p2.replace(/\u2212/g, '-')}}$`);
  s = s.replace(/(\b[a-zA-Z0-9]+)\s*\$\^\{([^}]+)\}\$/g, (_, p1, p2) => `$${p1}^{${p2.replace(/\u2212/g, '-')}}$`);
  s = s.replace(/(\b[a-zA-Z0-9]+)\s*\$_\{([^}]+)\}\$/g, (_, p1, p2) => `$${p1}_{${p2}}$`);

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
      const rendered = katex.renderToString(`${base}_{${subStr}}`, { displayMode: false, throwOnError: false, strict: false });
      const id = `${MATH_PREFIX}${renderedMath.length}${MATH_SUFFIX}`;
      renderedMath.push(rendered.includes('katex-error') ? `<span class="inline-math">${base}_{${subStr}}</span>` : rendered);
      return id;
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
      if (rendered.includes('katex-error')) return match;
      const id = `${MATH_PREFIX}${renderedMath.length}${MATH_SUFFIX}`;
      renderedMath.push(rendered);
      return id;
    } catch {
      return match;
    }
  });

  // 5. Process raw square roots: \sqrt{...}, \sqrt[n]{...}
  s = s.replace(/\\sqrt(?:\[[^\]]+\])?(\{(?:[^{}]|\{[^{}]*\})*\})/g, (match) => {
    try {
      const rendered = katex.renderToString(match, { displayMode: false, throwOnError: false, strict: false });
      if (rendered.includes('katex-error')) return match;
      const id = `${MATH_PREFIX}${renderedMath.length}${MATH_SUFFIX}`;
      renderedMath.push(rendered);
      return id;
    } catch {
      return match;
    }
  });

  // 5.5 Process raw scientific notation powers before standalone \times:
  // e.g. 26.4 \times 10^{-12}, 1.5 \times 10^{-16}, 3 \times 10^{-10}, 10^{-12}
  s = s.replace(/(?<=^|[\s\(\[\{=+\-*/,~])(?:\d+(?:\.\d+)?\s*(?:\\times|×)\s*)?([a-zA-Z0-9]+|\))\^([0-9a-zA-Z\-]+|\{[^}]+\})(?=$|[\s\)\]},.;:!?=+\-*/~])/g, (match) => {
    try {
      const sanitized = sanitizeMathForKatex(match);
      const rendered = katex.renderToString(sanitized, { displayMode: false, throwOnError: false, strict: false });
      if (rendered.includes('katex-error')) return match;
      const id = `${MATH_PREFIX}${renderedMath.length}${MATH_SUFFIX}`;
      renderedMath.push(rendered);
      return id;
    } catch {
      return match;
    }
  });

  // 6. Process raw Greek letters & math symbols: \alpha, \beta, \times, \pm, \chi^2, etc.
  s = s.replace(/\\(alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|iota|kappa|lambda|mu|nu|xi|pi|rho|varrho|sigma|tau|upsilon|phi|varphi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|times|div|pm|mp|neq|ne|approx|leq|geq|le|ge|infty|cdot|circ|degree|sum|prod|int|log|ln|sin|cos|tan)(?:\^([0-9a-zA-Z]+|\{[^\}]+\})|_([0-9a-zA-Z]+|\{[^\}]+\}))?/g, (match) => {
    try {
      const rendered = katex.renderToString(match, { displayMode: false, throwOnError: false, strict: false });
      if (rendered.includes('katex-error')) return match;
      const id = `${MATH_PREFIX}${renderedMath.length}${MATH_SUFFIX}`;
      renderedMath.push(rendered);
      return id;
    } catch {
      return match;
    }
  });

  // 7. Degree symbol: 45° -> 45^\circ
  s = s.replace(/([0-9]+(?:\.[0-9]+)?)\s*°/g, (_, num) => {
    try {
      const rendered = katex.renderToString(`${num}^\\circ`, { displayMode: false, throwOnError: false, strict: false });
      if (rendered.includes('katex-error')) return `${num}°`;
      const id = `${MATH_PREFIX}${renderedMath.length}${MATH_SUFFIX}`;
      renderedMath.push(rendered);
      return id;
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
      const rendered = katex.renderToString(`${base}^{${supStr}}`, { displayMode: false, throwOnError: false, strict: false });
      if (rendered.includes('katex-error')) return `${base}^{${supStr}}`;
      const id = `${MATH_PREFIX}${renderedMath.length}${MATH_SUFFIX}`;
      renderedMath.push(rendered);
      return id;
    } catch {
      return `${base}^{${supStr}}`;
    }
  });

  // 9. Inline powers without $...$ e.g. x^2, 10^3, 2^{n+1}, m^2
  s = s.replace(/(?<=^|[\s\(\[\{=+\-*/,~])([a-zA-Z0-9]+|\))\^([0-9a-zA-Z\-]+|\{[^}]+\})(?=$|[\s\)\]},.;:!?=+\-*/~])/g, (match) => {
    try {
      const sanitized = sanitizeMathForKatex(match);
      const rendered = katex.renderToString(sanitized, { displayMode: false, throwOnError: false, strict: false });
      if (rendered.includes('katex-error')) return match;
      const id = `${MATH_PREFIX}${renderedMath.length}${MATH_SUFFIX}`;
      renderedMath.push(rendered);
      return id;
    } catch {
      return match;
    }
  });

  // 9b. Inline subscripts without $...$ e.g. q_0, q_{0}, a_1, x_i
  s = s.replace(/(?<=^|[\s\(\[\{=+\-*/,~])([a-zA-Z\u03B1-\u03C9\u0391-\u03A9]{1,2})_([0-9a-zA-Z\u03B1-\u03C9]{1,3}|\{[^}]+\})(?=$|[\s\)\]},.;:!?=+\-*/~])/g, (match) => {
    try {
      const rendered = katex.renderToString(match, { displayMode: false, throwOnError: false, strict: false });
      if (rendered.includes('katex-error')) return match;
      const id = `${MATH_PREFIX}${renderedMath.length}${MATH_SUFFIX}`;
      renderedMath.push(rendered);
      return id;
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

export interface ExtractedMediaAsset {
  src: string;
  isCloudflare: boolean;
  alt: string;
}

export function extractImagesFromQuestion(q: any): ExtractedMediaAsset[] {
  if (!q) return [];
  const assets: ExtractedMediaAsset[] = [];
  const seen = new Set<string>();

  const textsToScan: string[] = [
    q.stem || '',
    ...Object.values(q.options || {}).map((v: any) => (typeof v === 'string' ? v : '')),
    q.explanation || '',
    q.answer_text || '',
    typeof q.image_data === 'string' ? q.image_data : ''
  ];

  for (let text of textsToScan) {
    if (!text) continue;
    text = text.replaceAll('exam.trogon.info', 'examcdn.trogon.info');

    // 1. Direct URL or base64 data string
    if (text.startsWith('http://') || text.startsWith('https://') || text.startsWith('data:image/')) {
      if (!seen.has(text)) {
        seen.add(text);
        const isCf = text.includes('imagedelivery.net') || text.includes('cloudflare') || text.includes('trogon.info') || text.startsWith('http');
        assets.push({
          src: text,
          isCloudflare: isCf,
          alt: 'Question Asset'
        });
      }
    }

    // 2. Extracted from <img> tags
    const imgRegex = /<img[^>]+src=["']([^"']+)["'][^>]*>/gi;
    let match;
    while ((match = imgRegex.exec(text)) !== null) {
      let src = match[1];
      if (src) {
        src = src.replaceAll('exam.trogon.info', 'examcdn.trogon.info');
        if (!seen.has(src)) {
          seen.add(src);
          const altMatch = match[0].match(/alt=["']([^"']+)["']/i);
          const alt = altMatch ? altMatch[1] : 'Extracted Image';
          const isCf = src.includes('imagedelivery.net') || src.includes('cloudflare') || src.includes('trogon.info') || src.startsWith('http');
          assets.push({
            src,
            isCloudflare: isCf,
            alt
          });
        }
      }
    }
  }

  return assets;
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
  ai_available?: boolean;
  on_demand_provider?: string;
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
  const hasStem = Boolean((q?.stem && q.stem.trim().length > 0) || (q?.sub_parts && isValidSubparts(q.sub_parts)) || extractSubpartsFromText(q?.stem));
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
  const [uploadProgressText, setUploadProgressText] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [rescanStates, setRescanStates] = useState<Record<number, 'rescanning' | 'success' | 'error'>>({});
  const [copiedUrl, setCopiedUrl] = useState<string | null>(null);

  // Question numbers are labels, not reliable client-side IDs: some papers restart
  // numbering in a new section. Keep the selected row's position as well, so polling
  // cannot replace its diagrams with the first question that has the same number.
  const selectQuestion = (question: any, index: number) => {
    setSelectedQuestion({ ...question, __uiQuestionIndex: index });
  };

  const findSelectedQuestion = (questions: any[], current: any) => {
    const selectedIndex = current?.__uiQuestionIndex;
    if (Number.isInteger(selectedIndex)) {
      const atSelectedIndex = questions[selectedIndex];
      if (atSelectedIndex && atSelectedIndex.question_number === current.question_number) {
        return { ...atSelectedIndex, __uiQuestionIndex: selectedIndex };
      }
    }

    // Only fall back to a number when it identifies exactly one row.
    const matches = questions.filter((q: any) => q.question_number === current?.question_number);
    return matches.length === 1
      ? { ...matches[0], __uiQuestionIndex: questions.indexOf(matches[0]) }
      : null;
  };

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
      setSelectedQuestion({ ...updatedQuestion, __uiQuestionIndex: selectedQuestion?.__uiQuestionIndex });
      
      // Update the question in the jobResult list
      setJobResult((prev: any) => {
        if (!prev) return prev;
        const newQuestions = prev.questions.map((q: any, index: number) =>
          index === selectedQuestion?.__uiQuestionIndex ? updatedQuestion : q
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
          explanation: qData.explanation || qData.answer_text || qData.reasoning
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
      const markedAns = qData.marked_answer || qData.marked_correct_answer || valAns;
      const isValid = result.is_valid !== undefined ? result.is_valid : Boolean(valAns && markedAns && valAns.toUpperCase() === markedAns.toUpperCase());
      const correctedExp = result.corrected_explanation || result.explanation || qData.explanation || qData.answer_text;

      const updatedQuestion = {
        ...qData,
        stem: result.corrected_stem,
        options: result.corrected_options,
        question_type: result.question_type || qData.question_type,
        sub_parts: result.sub_parts || (isValidSubparts(qData.sub_parts) ? qData.sub_parts : null),
        marked_answer: markedAns,
        marked_correct_answer: markedAns,
        is_valid: isValid,
        validated_answer: valAns,
        reasoning: result.reasoning || correctedExp,
        explanation: correctedExp,
        answer_text: correctedExp,
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
        const newQuestions = prev.questions.map((q: any, index: number) =>
          index === qData.__uiQuestionIndex ? updatedQuestion : q
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

  const [isVerifyingSingle, setIsVerifyingSingle] = useState(false);

  const handleAiVerifySingle = async (qData: any) => {
    setIsVerifyingSingle(true);
    try {
      const res = await fetch(`${API_BASE}/verify_question_with_ai`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: selectedJobId,
          question_number: qData.question_number,
          stem: qData.stem,
          options: qData.options,
          marked_answer: qData.marked_answer || qData.marked_correct_answer,
          explanation: qData.explanation || qData.answer_text || qData.reasoning
        })
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to verify question with AI');
      }

      const result = await res.json();
      if (!result.success) {
        throw new Error(result.error || 'AI verification failed');
      }

      const valAns = result.validated_answer || (result.is_valid ? (qData.marked_answer || qData.marked_correct_answer) : null);
      const markedAns = qData.marked_answer || qData.marked_correct_answer || valAns;
      const isValid = result.is_valid !== undefined ? result.is_valid : Boolean(valAns && markedAns && valAns.toUpperCase() === markedAns.toUpperCase());
      const reasoning = result.reasoning || result.corrected_explanation || qData.reasoning || qData.explanation;

      const updatedQuestion = {
        ...qData,
        is_valid: isValid,
        validated_answer: valAns,
        reasoning: reasoning,
        token_usage: result.token_usage,
        pipeline_steps: {
          ...(qData.pipeline_steps || {}),
          send_errors_to_llms: {
            name: "send_errors_to_llms",
            label: "Send Errors to LLMs",
            status: "completed",
            details: isValid
              ? `AI verified answer (Agreed: Option ${valAns || 'Match'}).`
              : `AI disagreed: verified Option ${valAns || 'different'} vs marked ${markedAns || '—'}.`
          }
        }
      };

      setSelectedQuestion(updatedQuestion);
      setJobResult((prev: any) => {
        if (!prev) return prev;
        const newQuestions = prev.questions.map((q: any, index: number) =>
          index === qData.__uiQuestionIndex ? updatedQuestion : q
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
        const unvalidatedCount = newQuestions.filter((q: any) => !q.validated_answer).length;
        return {
          ...prev,
          questions: newQuestions,
          summary: {
            ...prev.summary,
            validated: validatedCount,
            flagged: flaggedCount,
            unvalidated: unvalidatedCount,
            token_usage: newUsage
          }
        };
      });
    } catch (err: any) {
      console.error('Failed to verify with AI', err);
      alert(err.message || 'Failed to verify question. Please check server logs.');
    } finally {
      setIsVerifyingSingle(false);
    }
  };

  const [isConvertingMath, setIsConvertingMath] = useState(false);
  const [mathStatusText, setMathStatusText] = useState<string>('');
  const [isRevalidating, setIsRevalidating] = useState(false);

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

  const revalidateJob = async (jobId: string) => {
    if (isRevalidating) return;
    setIsRevalidating(true);
    try {
      const res = await fetch(`${API_BASE}/pdfs/${jobId}/revalidate`, { method: 'POST' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Revalidation failed');
      }
      const data = await res.json();
      const normalizedResult = normalizeCdnUrls(data.result);
      setJobResult(normalizedResult);
      if (selectedQuestion && normalizedResult?.questions) {
        const matching = findSelectedQuestion(normalizedResult.questions, selectedQuestion);
        if (matching) setSelectedQuestion(matching);
      }
    } catch (e: any) {
      console.error('Failed to revalidate job:', e);
      alert(e.message || 'Failed to revalidate job');
    } finally {
      setIsRevalidating(false);
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
      if (!res.ok) throw new Error(`Could not load uploaded PDFs (${res.status})`);
      const data = await res.json();
      setJobs(data);
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : 'Could not load uploaded PDFs');
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
    const rawFiles = e.target.files;
    if (!rawFiles || rawFiles.length === 0) return;
    const fileList = Array.from(rawFiles).filter(f => f.name.toLowerCase().endsWith('.pdf'));
    if (fileList.length === 0) {
      setUploadError('Only PDF files (.pdf) are allowed');
      e.target.value = '';
      return;
    }

    setIsUploading(true);
    setUploadError(null);
    setUploadProgressText(fileList.length > 1 ? `Uploading ${fileList.length} PDFs...` : `Uploading ${fileList[0].name}...`);

    try {
      if (fileList.length === 1) {
        const formData = new FormData();
        formData.append('file', fileList[0]);
        const res = await fetch(`${API_BASE}/pdfs/upload`, {
          method: 'POST',
          body: formData,
        });
        if (!res.ok) {
          let detail = `Upload failed (${res.status})`;
          try {
            const errorData = await res.json();
            detail = errorData.detail || detail;
          } catch {}
          throw new Error(detail);
        }
      } else {
        const formData = new FormData();
        for (const file of fileList) {
          formData.append('files', file);
        }
        const res = await fetch(`${API_BASE}/pdfs/upload-bulk`, {
          method: 'POST',
          body: formData,
        });
        if (!res.ok) {
          let detail = `Bulk upload failed (${res.status})`;
          try {
            const errorData = await res.json();
            detail = errorData.detail || detail;
          } catch {}
          throw new Error(detail);
        }
      }
      await fetchJobs();
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Upload failed';
      setUploadError(message);
      console.error('Upload failed', e);
    } finally {
      setIsUploading(false);
      setUploadProgressText(null);
      e.target.value = '';
    }
  };

  const loadJobResult = async (jobId: string) => {
    setSelectedJobId(jobId);
    try {
      const res = await fetch(`${API_BASE}/pdfs/${jobId}/result`);
      if (!res.ok) {
        let detail = `Could not load PDF result (${res.status})`;
        try {
          const errorData = await res.json();
          detail = errorData.detail || detail;
        } catch {
          // Keep the HTTP status when the backend did not return JSON.
        }
        throw new Error(detail);
      }
      const data = await res.json();
      if (!data.result) throw new Error('This PDF has no processed result yet.');
      const normalizedResult = normalizeCdnUrls(data.result);
      setJobResult(normalizedResult);
      if (selectedQuestion && normalizedResult?.questions) {
        const matching = findSelectedQuestion(normalizedResult.questions, selectedQuestion);
        if (matching) setSelectedQuestion(matching);
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Could not load PDF result';
      setUploadError(message);
      console.error('Failed to load result', e);
    }
  };

  useEffect(() => {
    if (selectedQuestion && jobResult?.questions) {
      const matching = findSelectedQuestion(jobResult.questions, selectedQuestion);
      if (matching && (
        matching.explanation !== selectedQuestion.explanation || 
        matching.stem !== selectedQuestion.stem || 
        matching.is_valid !== selectedQuestion.is_valid ||
        matching.answer_text !== selectedQuestion.answer_text
      )) {
        setSelectedQuestion(matching);
      }
    }
  }, [jobResult]);

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
          ) : config?.ai_available ? (
            <div className="pill-badge" style={{ background: 'rgba(168, 85, 247, 0.12)', borderColor: 'rgba(168, 85, 247, 0.35)', color: '#c084fc' }} title="AI is ready on demand. Automatic validation on upload is turned off. Click 'Verify with AI' to run.">
              <span className="pill-dot" style={{ background: '#a855f7', boxShadow: '0 0 8px #a855f7' }} />
              AI On-Demand ({config?.on_demand_provider === 'groq' ? 'Groq' : 'Gemini'})
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
                {isUploading ? (uploadProgressText || 'Uploading & Queueing PDFs...') : 'Upload Question Paper PDF(s)'}
              </h3>
              <p style={{ color: 'var(--text-muted)', marginTop: '0.35rem', fontSize: '0.86rem' }}>
                {isUploading
                  ? 'Adding documents to sequential processing queue...'
                  : 'Select or drop single or multiple PDFs — processed sequentially one at a time'}
              </p>
              <input type="file" accept=".pdf" multiple onChange={handleFileUpload} style={{ display: 'none' }} disabled={isUploading} />
            </label>
          </div>
          {uploadError && (
            <div style={{ marginTop: '0.75rem', color: '#fca5a5', fontSize: '0.86rem' }} role="alert">
              {uploadError}
            </div>
          )}

          {(() => {
            const activeJob = jobs.find(j => j.status === 'processing' || j.status === 'validating');
            const queuedJobs = jobs.filter(j => j.status === 'pending' || j.status === 'queued');
            if (activeJob || queuedJobs.length > 0) {
              return (
                <div style={{
                  marginTop: '1.25rem',
                  padding: '0.75rem 1.1rem',
                  borderRadius: '12px',
                  background: 'rgba(56, 189, 248, 0.05)',
                  border: '1px solid rgba(56, 189, 248, 0.2)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: '1rem',
                  fontSize: '0.84rem'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', minWidth: 0 }}>
                    <span className="spin" style={{ display: 'inline-flex', color: '#38bdf8', fontSize: '1rem' }}>⟳</span>
                    <span style={{ color: '#e2e8f0', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {activeJob ? (
                        <>Processing: <strong style={{ color: '#38bdf8' }}>{activeJob.filename}</strong></>
                      ) : (
                        'Queue ready'
                      )}
                    </span>
                  </div>
                  {queuedJobs.length > 0 && (
                    <span style={{
                      flexShrink: 0,
                      padding: '0.2rem 0.65rem',
                      borderRadius: '20px',
                      background: 'rgba(148, 163, 184, 0.12)',
                      border: '1px solid rgba(148, 163, 184, 0.25)',
                      color: '#cbd5e1',
                      fontSize: '0.75rem',
                      fontWeight: 600
                    }}>
                      {queuedJobs.length} queued (1 by 1)
                    </span>
                  )}
                </div>
              );
            }
            return null;
          })()}

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
                style={{
                  padding: '1.25rem',
                  cursor: (job.status === 'done' || job.status === 'validating') ? 'pointer' : 'default',
                  opacity: (job.status === 'pending' || job.status === 'queued') ? 0.75 : 1
                }}
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
                    {(job.status === 'pending' || job.status === 'queued') && (
                      <span className="pill-badge" style={{ padding: '0.2rem 0.6rem', fontSize: '0.72rem', background: 'rgba(148, 163, 184, 0.1)', color: '#94a3b8', border: '1px solid rgba(148, 163, 184, 0.2)' }}>
                        <span className="pill-dot" style={{ background: '#94a3b8' }} /> Queued
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

                    {selectedJobId && (
                      <button
                        onClick={() => revalidateJob(selectedJobId)}
                        disabled={isRevalidating}
                        className={jobResult?.summary?.validated > 0 || jobResult?.summary?.flagged > 0 ? "btn btn-secondary" : "btn btn-primary"}
                        style={{
                          padding: '0.45rem 1rem',
                          borderRadius: '9999px',
                          fontSize: '0.82rem',
                          background: isRevalidating
                            ? 'rgba(168, 85, 247, 0.2)'
                            : (jobResult?.summary?.validated > 0 || jobResult?.summary?.flagged > 0
                                ? 'rgba(168, 85, 247, 0.1)'
                                : 'linear-gradient(135deg, #a855f7 0%, #6366f1 100%)'),
                          borderColor: 'rgba(168, 85, 247, 0.5)',
                          color: (jobResult?.summary?.validated > 0 || jobResult?.summary?.flagged > 0 ? '#c084fc' : '#ffffff'),
                          cursor: isRevalidating ? 'wait' : 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.45rem',
                          boxShadow: !(jobResult?.summary?.validated > 0 || jobResult?.summary?.flagged > 0)
                            ? '0 2px 10px rgba(168, 85, 247, 0.35)'
                            : 'none'
                        }}
                        title={jobResult?.summary?.validated > 0 || jobResult?.summary?.flagged > 0
                          ? "Re-run AI answer verification & text checking on all questions"
                          : "Run AI answer solving and validation on all questions in this paper"}
                      >
                        {isRevalidating ? (
                          <RefreshCw size={14} className="spin" color={jobResult?.summary?.validated > 0 || jobResult?.summary?.flagged > 0 ? "#c084fc" : "#ffffff"} />
                        ) : (jobResult?.summary?.validated > 0 || jobResult?.summary?.flagged > 0 ? (
                          <RefreshCw size={14} color="#c084fc" />
                        ) : (
                          <Sparkles size={14} color="#ffffff" />
                        ))}
                        <span>
                          {isRevalidating
                            ? 'Verifying with AI...'
                            : (jobResult?.summary?.validated > 0 || jobResult?.summary?.flagged > 0
                                ? 'Re-verify with AI'
                                : 'Verify with AI')}
                        </span>
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
                    {jobResult.questions.map((q: any, index: number) => (
                      <tr key={`${q.question_number}-${index}`} onClick={() => selectQuestion(q, index)}>
                        <td style={{ textAlign: 'center' }}>
                          <span className="q-num-pill">
                            {q.question_number}
                          </span>
                          <div style={{
                            marginTop: '3px',
                            fontSize: '0.62rem',
                            fontWeight: 700,
                            letterSpacing: '0.5px',
                            color: isMcq(q) ? '#38bdf8' : ((q.sub_parts && isValidSubparts(q.sub_parts, q.stem)) || (q.options && !isMcq(q) && isValidSubparts(q.options, q.stem)) || q.question_type === 'descriptive_subquestions') ? '#a78bfa' : '#94a3b8',
                            opacity: 0.9
                          }}>
                            {isMcq(q) ? 'MCQ' : ((q.sub_parts && isValidSubparts(q.sub_parts, q.stem)) || (q.options && !isMcq(q) && isValidSubparts(q.options, q.stem)) || q.question_type === 'descriptive_subquestions') ? 'SUB-Q' : 'DESC'}
                          </div>
                        </td>
                        <td style={{ maxWidth: '420px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          <span 
                            style={{ color: 'var(--text-secondary)' }}
                            dangerouslySetInnerHTML={{
                              __html: renderLatex(
                                (() => {
                                  let baseText = (healAssertionReasonQuestion(q).stem || '').trim();
                                  if (!baseText) {
                                    const parts = (q.sub_parts && isValidSubparts(q.sub_parts, q.stem) ? q.sub_parts : null) || 
                                      (q.options && !isMcq(q) && isValidSubparts(q.options, q.stem) ? q.options : null) || 
                                      extractSubpartsFromText(q.stem);
                                    if (parts && Object.keys(parts).length > 0) {
                                      baseText = Object.entries(parts).map(([k, v]) => `(${k}) ${v}`).join(' ');
                                    }
                                  }
                                  return (baseText
                                    .replace(/<img[^>]*alt="math"[^>]*\/?>/gi, ' [Equation] ')
                                    .replace(/<img[^>]*alt="diagram"[^>]*\/?>/gi, ' [Diagram] ')
                                    .replace(/<[^>]+>/g, ' ')
                                    .trim()
                                    .slice(0, 110)) || (isMcq(q) ? '[Question Choices]' : '[Descriptive Question]');
                                })()
                              )
                            }} 
                          />
                          {extractImagesFromQuestion(q).some(m => m.isCloudflare) && (
                            <span 
                              style={{ 
                                marginLeft: '8px', 
                                background: 'rgba(249, 115, 22, 0.14)', 
                                color: '#f97316', 
                                border: '1px solid rgba(249, 115, 22, 0.35)', 
                                padding: '1px 6px', 
                                borderRadius: '4px', 
                                fontSize: '0.68rem',
                                fontWeight: 700,
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: '3px',
                                verticalAlign: 'middle'
                              }}
                              title="Contains Cloudflare CDN Image"
                            >
                              <Cloud size={10} /> CDN
                            </span>
                          )}
                        </td>
                        <td>
                          {isMcq(q) ? (
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
                          ) : (
                            <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontStyle: 'italic' }}>N/A</span>
                          )}
                        </td>
                        <td>
                          {isMcq(q) ? (() => {
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
                          })() : (
                            <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontStyle: 'italic' }}>N/A</span>
                          )}
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
          <div key={`${selectedQuestion.__uiQuestionIndex ?? 'unindexed'}-${selectedQuestion.question_number}`}>
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
                  className="btn btn-secondary"
                  onClick={() => handleAiVerifySingle(selectedQuestion)}
                  disabled={isVerifyingSingle || isRewriting}
                  style={{
                    padding: '0.35rem 0.75rem',
                    borderRadius: '9999px',
                    fontSize: '0.78rem',
                    background: isVerifyingSingle ? 'rgba(168, 85, 247, 0.2)' : 'rgba(168, 85, 247, 0.1)',
                    borderColor: 'rgba(168, 85, 247, 0.4)',
                    color: '#c084fc',
                    border: '1px solid rgba(168, 85, 247, 0.4)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.4rem',
                    opacity: (isVerifyingSingle || isRewriting) ? 0.7 : 1,
                    cursor: (isVerifyingSingle || isRewriting) ? 'not-allowed' : 'pointer'
                  }}
                  title="Solves this question with AI and verifies if the marked answer is correct"
                >
                  {isVerifyingSingle ? (
                    <>
                      <RefreshCw size={12} className="spin" color="#c084fc" />
                      <span>Verifying...</span>
                    </>
                  ) : (
                    <>
                      <CheckCircle2 size={12} color="#c084fc" />
                      <span>Verify with AI</span>
                    </>
                  )}
                </button>
                <button
                  className="btn btn-primary"
                  onClick={() => handleAiRewrite(selectedQuestion)}
                  disabled={isRewriting || isVerifyingSingle}
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

            {/* Garbled Text Warning & OCR Fallback */}
            {selectedQuestion?.is_garbled && (
              <div className="detail-box" style={{ marginBottom: '1.25rem', borderLeft: '4px solid #f59e0b', background: 'rgba(245, 158, 11, 0.05)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <AlertTriangle size={18} color="#f59e0b" />
                    <div>
                      <h4 style={{ margin: 0, color: '#f59e0b', fontSize: '0.95rem' }}>Text Garbled (Offline OCR Failed)</h4>
                      <p style={{ margin: '4px 0 0', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        The offline OCR couldn't read the custom fonts or complex math correctly.
                      </p>
                    </div>
                  </div>
                  <button 
                    onClick={() => handleRescan(selectedQuestion.question_number, 'gemini')}
                    style={{
                      background: 'linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)',
                      color: 'white',
                      border: 'none',
                      borderRadius: '6px',
                      padding: '6px 12px',
                      fontSize: '0.8rem',
                      fontWeight: 600,
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      cursor: 'pointer',
                      boxShadow: '0 2px 4px rgba(99, 102, 241, 0.25)',
                      transition: 'transform 0.1s, filter 0.2s',
                    }}
                    onMouseOver={e => e.currentTarget.style.filter = 'brightness(1.1)'}
                    onMouseOut={e => e.currentTarget.style.filter = 'brightness(1)'}
                    onMouseDown={e => e.currentTarget.style.transform = 'scale(0.96)'}
                    onMouseUp={e => e.currentTarget.style.transform = 'scale(1)'}
                  >
                    <Sparkles size={14} /> Fix with Gemini OCR
                  </button>
                </div>
              </div>
            )}

            {/* Original Question Image Crop */}
            <div className="detail-box" style={{ marginBottom: '1.25rem' }}>
              <h4 style={{ margin: '0 0 1rem 0', fontSize: '0.95rem', color: '#f1f5f9' }}>Original Question Crop</h4>
              <div style={{ background: '#111827', borderRadius: '6px', padding: '1rem', border: '1px solid #374151', display: 'flex', justifyContent: 'center' }}>
                <img 
                  src={`${API_BASE}/pdfs/${selectedJobId}/questions/${selectedQuestion.question_number}/image`} 
                  alt={`Question ${selectedQuestion.question_number}`}
                  style={{ maxWidth: '100%', maxHeight: '400px', objectFit: 'contain' }}
                />
              </div>
            </div>

            {/* Question Stem Box (Hidden if question only has sub-questions, to avoid showing 2 times) */}
            {(() => {
              const healed = healAssertionReasonQuestion(selectedQuestion);
              const effectiveStem = healed.stem;
              const effectiveOptions = healed.options;

              const rawSub = selectedQuestion.sub_parts;
              let validSub = isValidSubparts(rawSub) ? rawSub : null;
              if (!validSub && effectiveOptions && !isMcq(selectedQuestion) && Object.keys(effectiveOptions).length > 0) {
                const candidate = Object.fromEntries(Object.entries(effectiveOptions).map(([k, v]) => [k.toLowerCase(), v]));
                if (isValidSubparts(candidate)) {
                  validSub = candidate;
                }
              }
              if (!validSub && !isMcq(selectedQuestion)) {
                validSub = extractSubpartsFromText(effectiveStem);
              }

              let displayStem = effectiveStem || '';
              // Auto-heal if rawSub had false subquestions (e.g. severed at "points A and B." with diagram):
              if (!validSub && rawSub && Object.keys(rawSub).length > 0) {
                const partsToRejoin: string[] = [];
                for (const [k, v] of Object.entries(rawSub)) {
                  const vStr = String(v || '').trim();
                  if (vStr) {
                    if (displayStem && /\b(?:points?|between|and|from|to)\s+[a-zA-Z]\s+and\s*$/i.test(displayStem)) {
                      partsToRejoin.push(`${k.toUpperCase()}. ${vStr}`);
                    } else {
                      partsToRejoin.push(vStr);
                    }
                  }
                }
                if (partsToRejoin.length > 0) {
                  displayStem = displayStem ? `${displayStem}\n${partsToRejoin.join('\n')}` : partsToRejoin.join('\n');
                }
              }

              const hasSubparts = Boolean(validSub && Object.keys(validSub).length > 0);
              const firstKey = validSub ? Object.keys(validSub).sort()[0] : undefined;
              const introStem = getIntroductoryStem(displayStem, hasSubparts, firstKey);

              // If there are sub-questions and no separate introductory stem, do NOT duplicate: hide the Question Stem box!
              if (hasSubparts && !introStem) {
                return null;
              }

              return (
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
                      __html: renderLatex(introStem || '<em style="color: var(--text-muted); font-size: 0.92rem;">No question stem text.</em>')
                    }}
                  />
                </div>
              );
            })()}

            {isMcq(selectedQuestion) || (healAssertionReasonQuestion(selectedQuestion).options && Object.keys(healAssertionReasonQuestion(selectedQuestion).options).length >= 2 && !isValidSubparts(selectedQuestion.sub_parts)) ? (
              <>
                {/* MCQ: Answer Options Box */}
                <div className="detail-box">
                  <div className="detail-box-title">Multiple Choice Options</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
                    {Object.entries(healAssertionReasonQuestion(selectedQuestion).options).map(([k, v]: [string, any]) => {
                      const valAns = selectedQuestion.validated_answer || (selectedQuestion.is_valid ? (selectedQuestion.marked_correct_answer || selectedQuestion.marked_answer) : null);
                      const markedAns = selectedQuestion.marked_correct_answer || selectedQuestion.marked_answer;
                      const isValidated = Boolean(valAns && valAns.toUpperCase() === k.toUpperCase());
                      const isMarked = Boolean(markedAns && markedAns.toUpperCase() === k.toUpperCase());
                      
                      const rawVal = typeof v === 'string' ? v : String(v || '');
                      const cleanVal = cleanOptionDisplay(rawVal, k);

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

                {/* MCQ: Answer Comparison Grid */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.25rem' }}>
                  <div className="detail-box" style={{ margin: 0 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                      <span className="detail-box-title" style={{ margin: 0 }}>Marked Exam Key</span>
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', background: 'rgba(255,255,255,0.06)', padding: '2px 6px', borderRadius: '4px' }}>Ground Truth</span>
                    </div>
                    {(() => {
                      const healedOpts = healAssertionReasonQuestion(selectedQuestion).options || {};
                      let mKey = (selectedQuestion.marked_correct_answer || selectedQuestion.marked_answer || '').trim();
                      const expText = selectedQuestion.answer_text || selectedQuestion.explanation;
                      if (expText && typeof expText === 'string') {
                        const m = expText.match(/^\s*[\(\[]\s*([A-Da-d1-4])\s*[\)\]]/);
                        if (m && (!mKey || (healedOpts && !healedOpts[mKey.toUpperCase()] && !healedOpts[mKey.toLowerCase()]))) {
                          mKey = m[1].toUpperCase();
                        }
                      }
                      const optText = healedOpts[mKey.toUpperCase()] || healedOpts[mKey.toLowerCase()] || (expText ? expText.replace(/^\s*[\(\[]\s*[A-Da-d1-4]\s*[\)\]][\s\:\.\-]*/, '') : '');
                      const cleanAns = cleanOptionDisplay(optText, mKey);

                      return (
                        <div style={{ marginTop: '0.35rem' }}>
                          <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem', flexWrap: 'wrap' }}>
                            <span style={{ fontSize: '1.75rem', fontWeight: 800, color: '#f8fafc' }}>
                              {mKey || '—'}
                            </span>
                            {cleanAns && (
                              <span
                                style={{ fontSize: '1.05rem', color: '#e2e8f0', lineHeight: '1.5' }}
                                dangerouslySetInnerHTML={{ __html: renderLatex(cleanAns) }}
                              />
                            )}
                          </div>
                        </div>
                      );
                    })()}
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
                    {(() => {
                      const valAns = selectedQuestion.validated_answer || (selectedQuestion.is_valid ? (selectedQuestion.marked_correct_answer || selectedQuestion.marked_answer) : null);
                      const healedOpts = healAssertionReasonQuestion(selectedQuestion).options || {};
                      const optText = valAns ? (healedOpts[valAns.toUpperCase()] || healedOpts[valAns.toLowerCase()] || '') : '';
                      const cleanAns = optText ? cleanOptionDisplay(optText, valAns) : '';

                      return (
                        <div style={{ marginTop: '0.35rem' }}>
                          <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem', flexWrap: 'wrap' }}>
                            <span style={{ 
                              fontSize: '1.75rem', 
                              fontWeight: 800, 
                              color: !valAns ? 'var(--text-muted)' : (selectedQuestion.is_valid ? 'var(--success)' : 'var(--warning)')
                            }}>
                              {valAns || '—'}
                            </span>
                            {cleanAns && (
                              <span
                                style={{ fontSize: '1.05rem', color: '#e2e8f0', lineHeight: '1.5' }}
                                dangerouslySetInnerHTML={{ __html: renderLatex(cleanAns) }}
                              />
                            )}
                          </div>
                        </div>
                      );
                    })()}
                  </div>
                </div>

                {/* MCQ: Solution / Explanation Box */}
                <div className="detail-box">
                  <div className="detail-box-title">Solution &amp; Explanation</div>
                  {(selectedQuestion.explanation || selectedQuestion.answer_text) ? (
                    <div
                      style={{
                        fontSize: '0.94rem',
                        lineHeight: '1.75',
                        color: 'var(--text-secondary)',
                        whiteSpace: 'pre-wrap'
                      }}
                      dangerouslySetInnerHTML={{ __html: renderLatex(cleanExplanationDisplay(selectedQuestion.explanation || selectedQuestion.answer_text, selectedQuestion.marked_correct_answer || selectedQuestion.marked_answer, isMcq(selectedQuestion))) }}
                    />
                  ) : (
                    <div style={{ fontSize: '0.88rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                      No automated explanation available for this offline question.
                    </div>
                  )}
                </div>
              </>
            ) : (
              <>
                {/* Descriptive: Question Parts Box */}
                {(() => {
                  let rawSub = selectedQuestion.sub_parts;
                  const isExplicitSubquestions = String(selectedQuestion.question_type).toLowerCase().trim() === 'descriptive_subquestions';
                  
                  let parts = (isExplicitSubquestions && rawSub) ? rawSub : null;
                  if (!parts) parts = isValidSubparts(rawSub, selectedQuestion.stem) ? rawSub : null;
                  if (!parts && selectedQuestion.options && !isMcq(selectedQuestion)) {
                    parts = (isExplicitSubquestions) ? selectedQuestion.options : (isValidSubparts(selectedQuestion.options, selectedQuestion.stem) ? selectedQuestion.options : null);
                  }
                  if (!parts && selectedQuestion.stem) {
                    parts = extractSubpartsFromText(selectedQuestion.stem);
                  }
                  
                  if (parts && (isExplicitSubquestions || isValidSubparts(parts, selectedQuestion.stem))) {
                    return (
                      <div className="detail-box">
                        <div className="detail-box-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            Question Parts
                            <span style={{
                              fontSize: '0.7rem',
                              color: '#a78bfa',
                              background: 'rgba(167, 139, 250, 0.1)',
                              border: '1px solid rgba(167, 139, 250, 0.25)',
                              padding: '2px 8px',
                              borderRadius: '9999px',
                              fontWeight: 600
                            }}>Descriptive</span>
                          </div>
                          {(!getIntroductoryStem(selectedQuestion.stem || '', true, Object.keys(parts).sort()[0]) && selectedQuestion.token_usage) && (
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
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                          {Object.entries(parts).map(([k, v]: [string, any]) => {
                            // Clean answer content from sub-part text (stop at "Answer:" marker)
                            const rawText = typeof v === 'string' ? v : String(v || '');
                            const answerMarkerIdx = rawText.search(/\*?\*?Answer:?\*?\*?/i);
                            let cleanText = answerMarkerIdx > 0 ? rawText.slice(0, answerMarkerIdx).trim() : rawText.trim();

                            // Ensure nested sub-subquestions (a), (b) or (i), (ii) break onto new lines
                            cleanText = cleanText.replace(/(?<=[^\n])\s+(\([ivxlcdmIVXLCDM]+\))\s*/g, '\n$1 ');
                            cleanText = cleanText.replace(/(?<=[?.!:;])\s*([ivxlcdmIVXLCDM]+\))\s*/g, '\n$1 ');
                            cleanText = cleanText.replace(/(?<=[?.!:;])\s*(\([1-9a-dA-D]\))\s*/g, '\n$1 ');
                            cleanText = cleanText.replace(/\n{3,}/g, '\n\n');

                            return (
                              <div
                                key={k}
                                style={{
                                  padding: '0.85rem 1rem',
                                  background: 'rgba(167, 139, 250, 0.04)',
                                  borderRadius: '10px',
                                  border: '1px solid rgba(167, 139, 250, 0.15)',
                                  display: 'flex',
                                  gap: '0.85rem',
                                  alignItems: 'flex-start'
                                }}
                              >
                                <span style={{
                                  background: 'rgba(167, 139, 250, 0.15)',
                                  color: '#c4b5fd',
                                  fontWeight: 700,
                                  fontSize: '0.85rem',
                                  padding: '0.2rem 0.55rem',
                                  borderRadius: '6px',
                                  minWidth: '28px',
                                  textAlign: 'center',
                                  flexShrink: 0
                                }}>
                                  ({k})
                                </span>
                                <span
                                  style={{ fontSize: '0.96rem', lineHeight: '1.6', color: 'var(--text-secondary)', whiteSpace: 'pre-line', wordBreak: 'break-word' }}
                                  dangerouslySetInnerHTML={{ __html: renderLatex(cleanText) }}
                                />
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    );
                  }
                  return null;
                })()}

                {/* Descriptive: Answer & Solution Box */}
                <div className="detail-box">
                  <div className="detail-box-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    Answer &amp; Solution
                  </div>
                  {(() => {
                    // Build answer text from multiple sources
                    let answerContent = selectedQuestion.explanation || selectedQuestion.answer_text || '';
                    
                    // If answer_text is empty, try to extract from sub_parts or options
                    if (!answerContent) {
                      const rawParts: Record<string, any> = selectedQuestion.sub_parts || selectedQuestion.options || {};
                      const parts: Record<string, any> = isValidSubparts(rawParts) 
                        ? rawParts 
                        : (extractSubpartsFromText(selectedQuestion.stem) || {});
                      for (const v of Object.values(parts)) {
                        const text = typeof v === 'string' ? v : String(v || '');
                        const answerIdx = text.search(/\*?\*?Answer:?\*?\*?/i);
                        if (answerIdx >= 0) {
                          answerContent = text.slice(answerIdx).trim();
                          break;
                        }
                      }
                    }
                    
                    if (answerContent || selectedQuestion.answer_latex) {
                      return (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                          {selectedQuestion.answer_latex && (
                            <div
                              style={{
                                fontSize: '1.05rem',
                                padding: '12px',
                                background: 'rgba(56, 189, 248, 0.05)',
                                borderLeft: '3px solid #38bdf8',
                                borderRadius: '4px',
                                overflowX: 'auto',
                              }}
                            >
                              {(() => {
                                try {
                                  return (
                                    <span 
                                      dangerouslySetInnerHTML={{ 
                                        __html: katex.renderToString(selectedQuestion.answer_latex, { displayMode: true, throwOnError: true, strict: false }) 
                                      }} 
                                    />
                                  );
                                } catch (e) {
                                  return (
                                    <span style={{ fontFamily: 'monospace', color: '#ef4444' }}>
                                      {selectedQuestion.answer_latex}
                                    </span>
                                  );
                                }
                              })()}
                            </div>
                          )}
                          {answerContent && (
                            <div
                              style={{
                                fontSize: '0.94rem',
                                lineHeight: '1.8',
                                color: 'var(--text-secondary)',
                                whiteSpace: 'pre-wrap',
                                wordBreak: 'break-word'
                              }}
                              dangerouslySetInnerHTML={{ __html: renderLatex(cleanExplanationDisplay(answerContent, selectedQuestion.marked_correct_answer || selectedQuestion.marked_answer, false)) }}
                            />
                          )}
                        </div>
                      );
                    }
                    return (
                      <div style={{ fontSize: '0.88rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                        No answer/solution available for this descriptive question.
                      </div>
                    );
                  })()}
                </div>
              </>
            )}
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
