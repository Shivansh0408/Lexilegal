import { useRef, useState } from 'react';

export const LAWYER_ANALYSIS_EVENT = 'lexbrief:lawyer-analysis';

const UploadIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M12 16V4m0 0L7 9m5-5 5 5M5 14v5h14v-5" />
  </svg>
);

function publishLawyerState(detail) {
  window.dispatchEvent(new CustomEvent(LAWYER_ANALYSIS_EVENT, { detail }));
}

function LawyerUpload() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const selectFile = (candidate) => {
    setError('');
    if (!candidate) return;
    if (candidate.type !== 'application/pdf' && !candidate.name.toLowerCase().endsWith('.pdf')) {
      setFile(null);
      setError('Please choose a PDF file.');
      return;
    }
    setFile(candidate);
    publishLawyerState({ status: 'ready', analysis: null, error: '', cached: false });
  };

  const analyse = async () => {
    if (!file || loading) return;
    setLoading(true);
    setError('');
    publishLawyerState({ status: 'loading', analysis: null, error: '', cached: false });
    const formData = new FormData();
    formData.append('file', file);
    const baseUrl = (process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000').replace(/\/$/, '');

    try {
      const response = await fetch(`${baseUrl}/api/lawyer/analyze`, { method: 'POST', body: formData });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error || `Analysis failed with status ${response.status}.`);
      publishLawyerState({
        status: 'complete',
        analysis: payload.analysis,
        error: '',
        cached: Boolean(payload.cached),
        savedFilename: payload.saved_filename,
        documentId: payload.document_id,
        pipeline: 'lawyer',
        ragIndexed: Boolean(payload.rag_indexed),
        ragError: payload.rag_error || '',
      });
      document.getElementById('lawyer-analysis')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (requestError) {
      const message = requestError.message || 'The PDF could not be analysed.';
      setError(message);
      publishLawyerState({ status: 'error', analysis: null, error: message, cached: false, documentId: null, pipeline: 'lawyer' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="pdf-upload-card lawyer-upload-card" aria-labelledby="lawyer-pdf-upload-title">
      <div className="upload-heading">
        <div><p className="upload-kicker">LEGAL STRATEGY ANALYSIS</p><h3 id="lawyer-pdf-upload-title">Upload the lawyer case file</h3></div>
        <span>PDF ONLY</span>
      </div>
      <div
        className={`upload-dropzone ${dragging ? 'dragging' : ''}`}
        onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => { event.preventDefault(); setDragging(false); }}
        onDrop={(event) => { event.preventDefault(); setDragging(false); selectFile(event.dataTransfer.files?.[0]); }}
      >
        <UploadIcon />
        <strong>{file ? file.name : 'Drop a case-file PDF here'}</strong>
        <p>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB selected` : 'or choose one from your computer'}</p>
        <button type="button" className="file-picker" onClick={() => inputRef.current?.click()}>
          {file ? 'Choose a different PDF' : 'Choose PDF'}
        </button>
        <input ref={inputRef} type="file" accept="application/pdf,.pdf" onChange={(event) => selectFile(event.target.files?.[0])} hidden />
      </div>
      {error && <p className="upload-error" role="alert">{error}</p>}
      <button className="analyse-file-button" type="button" disabled={!file || loading} onClick={analyse}>
        {loading ? 'Building strategy brief…' : 'Build lawyer strategy brief'}
      </button>
      <p className="upload-notice">The result is saved for reuse and must be verified by qualified counsel.</p>
    </section>
  );
}

export default LawyerUpload;
