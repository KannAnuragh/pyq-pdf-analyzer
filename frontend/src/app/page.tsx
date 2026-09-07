'use client';

import { useState, useEffect } from 'react';
import { UploadCloud, FileText, CheckCircle2, AlertTriangle, Trash2, ExternalLink, RefreshCw, Type, X } from 'lucide-react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

const API_BASE = 'http://localhost:8000';

function renderLatex(text: string): string {
  if (!text) return '';

  // 0. Protect fill-in-the-blanks (3 or more underscores) so they don't break LaTeX parsing
  const blanks: string[] = [];
  let s = text.replace(/_{3,}/g, (match) => {
    const id = `___BLANK_${blanks.length}___`;
    blanks.push(match);
    return id;
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
      return katex.renderToString(math.trim(), { displayMode: true, throwOnError: false, strict: false });
    } catch {
      return `$$${math}$$`;
    }
  });

  s = s.replace(/\\\[([\s\S]+?)\\\]/g, (_, math) => {
    try {
      return katex.renderToString(math.trim(), { displayMode: true, throwOnError: false, strict: false });
    } catch {
      return `\\[${math}\\]`;
    }
  });

  // 3. Process inline math: $...$ and \(...\)
  s = s.replace(/\$([^\$\n]+?)\$/g, (_, math) => {
    try {
      return katex.renderToString(math.trim(), { displayMode: false, throwOnError: false, strict: false });
    } catch {
      return `$${math}$`;
    }
  });

  s = s.replace(/\\\(([\s\S]+?)\\\)/g, (_, math) => {
    try {
      return katex.renderToString(math.trim(), { displayMode: false, throwOnError: false, strict: false });
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

  // 8. Superscript exponents: x², x³, x⁴, etc.
  s = s.replace(/([a-zA-Z0-9\)])([²³⁴ⁿ])/g, (_, base, sup) => {
    const map: Record<string, string> = { '²': '^2', '³': '^3', '⁴': '^4', 'ⁿ': '^n' };
    try {
      return katex.renderToString(`${base}${map[sup]}`, { displayMode: false, throwOnError: false, strict: false });
    } catch {
      return `${base}${sup}`;
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
    s = s.replace(
      `___BLANK_${i}___`,
      '<span style="display:inline-block;border-bottom:1.5px solid currentColor;min-width:48px;margin:0 4px;"></span>'
    );
  });

  return s;
}

interface AppConfig {
  gemini_enabled: boolean;
  gemini_status: {
    enabled: boolean;
    reason: string;
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

  const [isRescanning, setIsRescanning] = useState(false);
  const [config, setConfig] = useState<AppConfig | null>(null);

  const handleRescan = async (questionNumber: number) => {
    if (!selectedJobId) return;
    setIsRescanning(true);
    try {
      const res = await fetch(`${API_BASE}/pdfs/${selectedJobId}/rescan/${questionNumber}`, {
        method: 'POST'
      });
      if (!res.ok) {
        throw new Error('Failed to rescan question');
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
    } catch (err) {
      console.error('Failed to rescan', err);
      alert('Failed to rescan question. Please check server logs.');
    } finally {
      setIsRescanning(false);
    }
  };

  useEffect(() => {
    fetchConfig();
    fetchJobs();
    const interval = setInterval(() => {
      fetchJobs();
      fetchConfig();
      if (selectedJobId) {
        // If we have a selected job, check its status from the latest fetched jobs
        loadJobResult(selectedJobId);
      }
    }, 5000);
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
      <header className="header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h1 style={{ margin: 0 }}>PDF QA Agent</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.25rem', marginBottom: 0 }}>
            Automated Question Extraction, Math OCR & AI Answer Verification
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <button
            onClick={() => { setShowFontModal(true); loadFonts(); }}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.4rem',
              padding: '0.4rem 0.85rem',
              borderRadius: '20px',
              background: 'rgba(99, 102, 241, 0.12)',
              border: '1px solid rgba(99, 102, 241, 0.35)',
              fontSize: '0.82rem',
              color: '#a5b4fc',
              fontWeight: 600,
              cursor: 'pointer'
            }}
            title="View installed Malayalam fonts and install new font files"
          >
            <Type size={14} />
            Fonts {fontsData ? `(${fontsData.installed_count})` : ''}
          </button>
          {config?.gemini_enabled ? (
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.4rem 0.85rem',
                borderRadius: '20px',
                background: 'rgba(16, 185, 129, 0.12)',
                border: '1px solid rgba(16, 185, 129, 0.35)',
                fontSize: '0.82rem',
                color: '#34d399',
                fontWeight: 600
              }}
              title="Gemini AI is active. Questions will be solved and validated by Gemini."
            >
              <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: '#10b981', boxShadow: '0 0 8px #10b981' }} />
              Gemini AI: Enabled
            </div>
          ) : (
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.4rem 0.85rem',
                borderRadius: '20px',
                background: 'rgba(239, 68, 68, 0.12)',
                border: '1px solid rgba(239, 68, 68, 0.35)',
                fontSize: '0.82rem',
                color: '#f87171',
                fontWeight: 600
              }}
              title={config?.gemini_status?.reason || 'Gemini AI is disabled in .env (ENABLE_GEMINI=false)'}
            >
              <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: '#ef4444' }} />
              Gemini AI: Disabled (.env)
            </div>
          )}
        </div>
      </header>

      {!selectedJobId ? (
        <>
          <div className="card">
            <label className="upload-zone">
              <UploadCloud size={48} color="var(--primary)" style={{ marginBottom: '1rem' }} />
              <h3>{isUploading ? 'Uploading...' : 'Drag & Drop your PDF here'}</h3>
              <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>or click to browse files</p>
              <input type="file" accept=".pdf" onChange={handleFileUpload} style={{ display: 'none' }} disabled={isUploading} />
            </label>
          </div>

          <div className="jobs-grid">
            {jobs.map(job => (
              <div key={job.job_id} className="card job-card" onClick={() => (job.status === 'done' || job.status === 'validating') && loadJobResult(job.job_id)}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <FileText size={20} color="var(--text-muted)" />
                    <span style={{ fontWeight: 500 }}>{job.filename}</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span className={`badge ${job.status}`}>{job.status}</span>
                    {job.has_file && (
                      <a
                        href={`${API_BASE}/pdfs/${job.job_id}/file`}
                        target="_blank"
                        rel="noopener noreferrer"
                        onClick={(e) => e.stopPropagation()}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: 'var(--text-muted)',
                          cursor: 'pointer',
                          padding: '4px',
                          borderRadius: '4px',
                          display: 'flex',
                          alignItems: 'center'
                        }}
                        title="Open Scanned PDF"
                      >
                        <ExternalLink size={16} />
                      </a>
                    )}
                    <button
                      onClick={(e) => deleteJob(e, job.job_id)}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: 'var(--text-muted)',
                        cursor: 'pointer',
                        padding: '4px',
                        borderRadius: '4px',
                        display: 'flex',
                        alignItems: 'center'
                      }}
                      title="Delete PDF"
                    >
                      <Trash2 size={16} color="#ef4444" />
                    </button>
                  </div>
                </div>
                <div style={{ marginTop: '1rem', fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                  {new Date(job.created_at).toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div>
          <button className="btn" onClick={() => { setSelectedJobId(null); setJobResult(null); }}>
            ← Back to Dashboard
          </button>
          
          {jobResult ? (
            <div className="card" style={{ marginTop: '2rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <h2>{jobResult.source_file}</h2>
                  <a
                    href={`${API_BASE}/pdfs/${selectedJobId}/file`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn"
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '0.4rem',
                      padding: '0.35rem 0.75rem',
                      fontSize: '0.85rem',
                      textDecoration: 'none'
                    }}
                    title="Open Scanned PDF in new tab"
                  >
                    <ExternalLink size={14} />
                    <span>View PDF</span>
                  </a>
                </div>
                <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                  {jobResult.summary.validated > 0 || jobResult.summary.flagged > 0 ? (
                    <>
                      <div className="badge done">Validated: {jobResult.summary.validated}</div>
                      <div className="badge pending">Flagged: {jobResult.summary.flagged}</div>
                    </>
                  ) : (
                    <div
                      className="badge"
                      style={{
                        background: 'rgba(245, 158, 11, 0.12)',
                        border: '1px solid rgba(245, 158, 11, 0.3)',
                        color: '#fbbf24',
                        fontWeight: 600
                      }}
                      title="AI validation was not executed because Gemini is disabled in .env"
                    >
                      AI Validation: Disabled
                    </div>
                  )}
                  <div className="badge" style={{ background: 'rgba(255, 255, 255, 0.06)', color: 'var(--text-muted)' }}>
                    Total: {jobResult.summary.total || jobResult.questions?.length || 0}
                  </div>
                  {jobs.find(j => j.job_id === selectedJobId)?.status === 'validating' && (
                    <div className="badge validating">Validating...</div>
                  )}
                </div>
              </div>

              <table className="data-table">
                <thead>
                  <tr>
                    <th style={{ width: '50px' }}>Q#</th>
                    <th>Stem Preview</th>
                    <th style={{ width: '90px' }}>Marked</th>
                    <th style={{ width: '100px' }}>AI Validated</th>
                    <th style={{ width: '110px' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {jobResult.questions.map((q: any) => (
                    <tr key={q.question_number} onClick={() => setSelectedQuestion(q)} style={{ cursor: 'pointer' }}>
                      <td>{q.question_number}</td>
                      <td style={{ maxWidth: '350px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        <span dangerouslySetInnerHTML={{ __html: renderLatex(q.stem.slice(0, 100)) }} />
                      </td>
                      <td style={{ fontWeight: 600 }}>{q.marked_correct_answer || q.marked_answer || '-'}</td>
                      <td>
                        {q.validated_answer ? (
                          <span style={{ fontWeight: 600, color: q.is_valid ? 'var(--success)' : 'var(--warning)' }}>
                            {q.validated_answer}
                          </span>
                        ) : (
                          <span style={{ color: 'var(--text-muted)' }}>-</span>
                        )}
                      </td>
                      <td>
                        {q.validated_answer ? (
                          q.is_valid ? (
                            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', color: 'var(--success)', fontSize: '0.85rem', fontWeight: 600 }}>
                              <CheckCircle2 className="status-icon valid" size={16} /> Match
                            </span>
                          ) : (
                            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', color: 'var(--warning)', fontSize: '0.85rem', fontWeight: 600 }}>
                              <AlertTriangle className="status-icon invalid" size={16} /> Mismatch
                            </span>
                          )
                        ) : jobs.find(j => j.job_id === selectedJobId)?.status === 'validating' ? (
                          <span style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>Validating...</span>
                        ) : (
                          <span
                            style={{
                              fontSize: '0.78rem',
                              color: '#9ca3af',
                              background: 'rgba(255, 255, 255, 0.05)',
                              padding: '2px 8px',
                              borderRadius: '4px',
                              border: '1px solid rgba(255, 255, 255, 0.1)',
                              whiteSpace: 'nowrap'
                            }}
                          >
                            AI Disabled
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div style={{ marginTop: '2rem' }}>Loading results...</div>
          )}
        </div>
      )}

      {/* Detail Panel */}
      <div className={`overlay ${selectedQuestion ? 'open' : ''}`} onClick={() => setSelectedQuestion(null)} />
      <div className={`detail-panel ${selectedQuestion ? 'open' : ''}`}>
        {selectedQuestion && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <h2>Question {selectedQuestion.question_number}</h2>
                <button 
                  className="btn" 
                  onClick={() => handleRescan(selectedQuestion.question_number)}
                  disabled={isRescanning}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '0.4rem',
                    padding: '0.35rem 0.75rem',
                    fontSize: '0.85rem',
                    background: 'rgba(59, 130, 246, 0.15)',
                    color: '#3b82f6',
                    border: '1px solid rgba(59, 130, 246, 0.3)',
                    opacity: isRescanning ? 0.6 : 1,
                    cursor: isRescanning ? 'not-allowed' : 'pointer'
                  }}
                  title="Uses Gemini Vision to fix OCR errors by scanning the exact page"
                >
                  <RefreshCw size={14} className={isRescanning ? "spin" : ""} />
                  {isRescanning ? 'Scanning...' : 'Rescan Image (AI)'}
                </button>
              </div>
              <button className="btn" onClick={() => setSelectedQuestion(null)}>Close</button>
            </div>
            
            <div style={{ marginBottom: '1.5rem' }}>
              <h4 style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Stem</h4>
              <div
                style={{
                  background: 'rgba(255,255,255,0.03)',
                  padding: '1.25rem',
                  borderRadius: '8px',
                  whiteSpace: 'pre-wrap',
                  lineHeight: '1.8',
                  fontSize: '0.96rem',
                  border: '1px solid var(--panel-border)'
                }}
                dangerouslySetInnerHTML={{
                  __html: renderLatex(
                    selectedQuestion.stem
                      .replace(/(\s*)([(](?:[ivx]+|[a-e]|\d+)[)])/gi, '\n$2')
                      .replace(/(\s*)([a-h]\.\s+)/gi, '\n$2')
                      .replace(/(\s*)(Statement\s+[I|V|X\d]+[\s:]+)/gi, '\n$2')
                      .replace(/(\s*)(List\s+[I|V|X\d]+[\s:]+)/gi, '\n$2')
                      .trim()
                  )
                }}
              />
            </div>

            <div style={{ marginBottom: '1.5rem' }}>
              <h4 style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Options</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {Object.entries(selectedQuestion.options).map(([k, v]: [string, any]) => {
                  const isValidated = selectedQuestion.validated_answer?.toUpperCase() === k.toUpperCase();
                  const isMarked = (selectedQuestion.marked_correct_answer || selectedQuestion.marked_answer)?.toUpperCase() === k.toUpperCase();
                  
                  let borderColor = 'var(--panel-border)';
                  let bg = 'rgba(255,255,255,0.02)';
                  if (isValidated && isMarked) {
                    borderColor = 'var(--success)';
                    bg = 'rgba(16, 185, 129, 0.08)';
                  } else if (isValidated) {
                    borderColor = 'var(--primary)';
                    bg = 'rgba(59, 130, 246, 0.08)';
                  } else if (isMarked) {
                    borderColor = '#f59e0b';
                    bg = 'rgba(245, 158, 11, 0.08)';
                  }

                  return (
                    <div
                      key={k}
                      style={{
                        padding: '0.85rem 1rem',
                        background: bg,
                        borderRadius: '8px',
                        border: `1.5px solid ${borderColor}`,
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'flex-start',
                        gap: '1rem',
                        whiteSpace: 'pre-wrap',
                        lineHeight: '1.5'
                      }}
                    >
                      <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start', flex: 1 }}>
                        <span style={{
                          background: isValidated ? 'var(--primary)' : isMarked ? '#f59e0b' : 'rgba(255,255,255,0.1)',
                          color: '#fff',
                          fontWeight: 700,
                          fontSize: '0.85rem',
                          padding: '0.2rem 0.55rem',
                          borderRadius: '6px',
                          minWidth: '28px',
                          textAlign: 'center'
                        }}>
                          {k}
                        </span>
                        <span
                          style={{ fontSize: '0.95rem' }}
                          dangerouslySetInnerHTML={{ __html: renderLatex(v) }}
                        />
                      </div>
                      <div style={{ display: 'flex', gap: '0.5rem', flexShrink: 0 }}>
                        {isValidated && isMarked && (
                          <span className="badge done" style={{ fontSize: '0.7rem' }}>AI Match ✅</span>
                        )}
                        {isValidated && !isMarked && (
                          <span className="badge processing" style={{ fontSize: '0.7rem' }}>AI Choice</span>
                        )}
                        {isMarked && !isValidated && (
                          <span className="badge pending" style={{ fontSize: '0.7rem', background: 'rgba(245, 158, 11, 0.2)', color: '#fbbf24', border: '1px solid rgba(245, 158, 11, 0.4)' }}>Marked Key</span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem' }}>
              <div style={{ flex: 1, padding: '1rem', background: 'rgba(255,255,255,0.03)', borderRadius: '8px', border: '1px solid var(--panel-border)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <h4 style={{ color: 'var(--text-muted)', margin: 0 }}>Marked Answer</h4>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', background: 'rgba(255,255,255,0.05)', padding: '1px 6px', borderRadius: '4px' }}>Exam Key</span>
                </div>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{selectedQuestion.marked_correct_answer || selectedQuestion.marked_answer || 'None'}</div>
              </div>
              <div style={{ flex: 1, padding: '1rem', background: 'rgba(255,255,255,0.03)', borderRadius: '8px', border: '1px solid var(--panel-border)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <h4 style={{ color: 'var(--text-muted)', margin: 0 }}>AI Validated Answer</h4>
                  {selectedQuestion.validated_answer ? (
                    <span style={{ fontSize: '0.72rem', color: selectedQuestion.is_valid ? 'var(--success)' : 'var(--warning)', fontWeight: 600 }}>
                      {selectedQuestion.is_valid ? 'Match' : 'Mismatch'}
                    </span>
                  ) : (
                    <span style={{ fontSize: '0.72rem', color: '#f87171', background: 'rgba(239, 68, 68, 0.1)', padding: '1px 6px', borderRadius: '4px' }}>Disabled in .env</span>
                  )}
                </div>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: selectedQuestion.validated_answer ? (selectedQuestion.is_valid ? 'var(--success)' : 'var(--warning)') : '#9ca3af' }}>
                  {selectedQuestion.validated_answer || 'None'}
                </div>
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <h4 style={{ color: 'var(--text-muted)', margin: 0 }}>Solver Reasoning</h4>
                {!selectedQuestion.validated_answer && (
                  <span style={{ fontSize: '0.75rem', color: '#f87171' }}>Gemini AI Offline</span>
                )}
              </div>
              <div
                style={{
                  background: !selectedQuestion.validated_answer ? 'rgba(239, 68, 68, 0.04)' : 'rgba(255,255,255,0.03)',
                  padding: '1.25rem',
                  borderRadius: '8px',
                  fontSize: '0.92rem',
                  lineHeight: '1.7',
                  color: 'var(--text-main)',
                  whiteSpace: 'pre-wrap',
                  border: !selectedQuestion.validated_answer ? '1px solid rgba(239, 68, 68, 0.2)' : '1px solid var(--panel-border)'
                }}
                dangerouslySetInnerHTML={{ __html: renderLatex(selectedQuestion.reasoning || 'No reasoning provided.') }}
              />
              {selectedQuestion.used_web_search && (
                <div style={{ marginTop: '0.75rem', fontSize: '0.82rem', color: 'var(--warning)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <span>🌐</span> Web search was consulted to verify this answer.
                </div>
              )}
            </div>
          </div>
        )}
      </div>

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
