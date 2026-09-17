import { useRef, useState } from 'react';

export const JUDGE_ANALYSIS_EVENT = 'lexbrief:judge-analysis';

const UploadIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M12 16V4m0 0L7 9m5-5 5 5M5 14v5h14v-5" />
  </svg>
);

function publishAnalysisState(detail) {
  window.dispatchEvent(new CustomEvent(JUDGE_ANALYSIS_EVENT, { detail }));
}

function Upload() {
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
    publishAnalysisState({ status: 'ready', analysis: null, error: '' });
  };

  const analyse = async () => {
    if (!file || loading) return;
    setLoading(true);
    setError('');
    publishAnalysisState({ status: 'loading', analysis: null, error: '' });
    const formData = new FormData();
    formData.append('file', file);
    const baseUrl = (process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000').replace(/\/$/, '');

    try {
      const response = await fetch(`${baseUrl}/api/judge/analyze`, {
        method: 'POST',
        body: formData,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.error || `Analysis failed with status ${response.status}.`);
      }
      publishAnalysisState({
        status: 'complete',
        analysis: payload.analysis,
        error: '',
        cached: Boolean(payload.cached),
        savedFilename: payload.saved_filename,
        documentId: payload.document_id,
        pipeline: 'judge',
        ragIndexed: Boolean(payload.rag_indexed),
        ragError: payload.rag_error || '',
      });
      document.getElementById('judge-analysis')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (requestError) {
      const message = requestError.message || 'The PDF could not be analysed.';
      setError(message);
      publishAnalysisState({ status: 'error', analysis: null, error: message, documentId: null, pipeline: 'judge' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="pdf-upload-card" aria-labelledby="pdf-upload-title">
      <div className="upload-heading">
        <div>
          <p className="upload-kicker">JUDICIAL FACT ANALYSIS</p>
          <h3 id="pdf-upload-title">Upload the case file</h3>
        </div>
        <span>PDF ONLY</span>
      </div>

      <div
        className={`upload-dropzone ${dragging ? 'dragging' : ''}`}
        onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => { event.preventDefault(); setDragging(false); }}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          selectFile(event.dataTransfer.files?.[0]);
        }}
      >
        <UploadIcon />
        <strong>{file ? file.name : 'Drop a court-file PDF here'}</strong>
        <p>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB selected` : 'or choose one from your computer'}</p>
        <button type="button" className="file-picker" onClick={() => inputRef.current?.click()}>
          {file ? 'Choose a different PDF' : 'Choose PDF'}
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          onChange={(event) => selectFile(event.target.files?.[0])}
          hidden
        />
      </div>

      {error && <p className="upload-error" role="alert">{error}</p>}
      <button className="analyse-file-button" type="button" disabled={!file || loading} onClick={analyse}>
        {loading ? 'Analysing file…' : 'Extract facts for judicial review'}
      </button>
      <p className="upload-notice">The output supports review; it does not make findings or issue a judgment.</p>
    </section>
  );
}

export default Upload;
