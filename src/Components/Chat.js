import { useEffect, useMemo, useRef, useState } from 'react';
import { JUDGE_ANALYSIS_EVENT } from './upload';
import { LAWYER_ANALYSIS_EVENT } from './lawyerUpload';
import './Chat.css';

const ChatIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M5 17.5 3.5 21l4.25-1.5A9 9 0 1 0 5 17.5Z" />
    <path d="M8 10h8M8 14h5" />
  </svg>
);

const SendIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="m4 4 16 8-16 8 3-8-3-8Z" />
    <path d="M7 12h13" />
  </svg>
);

function Chat({ activeRole }) {
  const [open, setOpen] = useState(false);
  const [contexts, setContexts] = useState({ judge: null, lawyer: null });
  const [messagesByRole, setMessagesByRole] = useState({ judge: [], lawyer: [] });
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const endRef = useRef(null);
  const inputRef = useRef(null);

  const supportedRole = activeRole === 'judge' || activeRole === 'lawyer' ? activeRole : null;
  const context = supportedRole ? contexts[supportedRole] : null;
  const messages = useMemo(
    () => (supportedRole ? messagesByRole[supportedRole] : []),
    [messagesByRole, supportedRole],
  );

  useEffect(() => {
    const receiveAnalysis = (pipeline) => (event) => {
      const detail = event.detail || {};
      if (detail.status === 'complete' && detail.documentId) {
        setContexts((current) => ({
          ...current,
          [pipeline]: {
            documentId: detail.documentId,
            filename: detail.savedFilename || 'Analysed PDF',
            ragIndexed: Boolean(detail.ragIndexed),
            ragError: detail.ragError || '',
          },
        }));
        setMessagesByRole((current) => ({
          ...current,
          [pipeline]: [{
            role: 'assistant',
            content: detail.ragIndexed
              ? `The ${pipeline} analysis is indexed in ChromaDB. Ask me anything contained in the saved analysis.`
              : `The ${pipeline} analysis is saved. RAG indexing will retry when you ask a question. Make sure Ollama and nomic-embed-text are available.`,
          }],
        }));
        setError('');
      } else if (detail.status === 'ready' || detail.status === 'loading') {
        setContexts((current) => ({ ...current, [pipeline]: null }));
        setMessagesByRole((current) => ({ ...current, [pipeline]: [] }));
      }
    };

    const receiveJudge = receiveAnalysis('judge');
    const receiveLawyer = receiveAnalysis('lawyer');
    window.addEventListener(JUDGE_ANALYSIS_EVENT, receiveJudge);
    window.addEventListener(LAWYER_ANALYSIS_EVENT, receiveLawyer);
    return () => {
      window.removeEventListener(JUDGE_ANALYSIS_EVENT, receiveJudge);
      window.removeEventListener(LAWYER_ANALYSIS_EVENT, receiveLawyer);
    };
  }, []);

  useEffect(() => {
    if (open) {
      endRef.current?.scrollIntoView({ behavior: 'smooth' });
      if (!loading) inputRef.current?.focus();
    }
  }, [messages, loading, open]);

  const sendQuestion = async (event) => {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || !context || !supportedRole || loading) return;

    const priorMessages = messages;
    const userMessage = { role: 'user', content: trimmed };
    setMessagesByRole((current) => ({
      ...current,
      [supportedRole]: [...current[supportedRole], userMessage],
    }));
    setQuestion('');
    setError('');
    setLoading(true);

    const baseUrl = (process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000').replace(/\/$/, '');
    try {
      const response = await fetch(`${baseUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pipeline: supportedRole,
          document_id: context.documentId,
          question: trimmed,
          history: priorMessages.slice(-10),
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error || `Chat request failed with status ${response.status}.`);
      setMessagesByRole((current) => ({
        ...current,
        [supportedRole]: [...current[supportedRole], {
          role: 'assistant',
          content: payload.answer,
          source: payload.source,
        }],
      }));
    } catch (requestError) {
      setError(requestError.message || 'The question could not be answered.');
    } finally {
      setLoading(false);
    }
  };

  const statusText = !supportedRole
    ? 'Select Judge or Lawyer to use case chat.'
    : context
      ? `${context.filename}${context.ragIndexed ? ' · RAG indexed' : ' · RAG indexing pending'}`
      : `Run a ${supportedRole} PDF analysis to enable chat.`;

  return (
    <aside className={`case-chat ${open ? 'open' : ''}`} aria-label="Case analysis chatbot">
      {open && (
        <section className="chat-panel" role="dialog" aria-label="Ask about the saved case analysis">
          <header className="chat-header">
            <div className="chat-heading-mark"><ChatIcon /></div>
            <div>
              <strong>Case Analysis Chat</strong>
              <span>{supportedRole ? `${supportedRole} view` : 'No active view'}</span>
            </div>
            <button type="button" onClick={() => setOpen(false)} aria-label="Close case chat">×</button>
          </header>

          <div className="chat-context" title={statusText}>
            <span className={context ? 'ready' : ''} />
            <p>{statusText}</p>
          </div>

          <div className="chat-messages" aria-live="polite">
            {!messages.length && (
              <div className="chat-empty">
                <ChatIcon />
                <strong>Questions stay grounded in your analysis.</strong>
                <p>Upload and analyse a PDF, then ask about facts, evidence, issues, risks, or next actions.</p>
              </div>
            )}
            {messages.map((message, index) => (
              <div className={`chat-message ${message.role}`} key={`${message.role}-${index}`}>
                <p>{message.content}</p>
                {message.source === 'langchain_chroma_rag' && <span>LangChain · Chroma · Llama</span>}
              </div>
            ))}
            {loading && <div className="chat-message assistant typing"><i /><i /><i /></div>}
            <div ref={endRef} />
          </div>

          {error && <p className="chat-error" role="alert">{error}</p>}
          <form className="chat-form" onSubmit={sendQuestion}>
            <textarea
              ref={inputRef}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder={context ? 'Ask about this saved analysis…' : 'Analyse a PDF to begin…'}
              aria-label="Question about the saved analysis"
              maxLength={2000}
              disabled={!context || loading}
              rows={2}
            />
            <button type="submit" disabled={!context || !question.trim() || loading} aria-label="Send question">
              <SendIcon />
            </button>
          </form>
          <p className="chat-disclaimer">RAG answers use retrieved saved-analysis JSON only. Verify against the source file.</p>
        </section>
      )}

      <button
        className="chat-launcher"
        type="button"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
        aria-label={open ? 'Close case analysis chat' : 'Open case analysis chat'}
      >
        <ChatIcon />
        <span>{open ? 'Close' : 'Ask LexBrief'}</span>
        {context && !open && <i aria-label="Analysis ready" />}
      </button>
    </aside>
  );
}

export default Chat;
