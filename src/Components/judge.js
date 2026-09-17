import { useEffect, useState } from 'react';
import { JUDGE_ANALYSIS_EVENT } from './upload';

function textValue(value) {
  if (typeof value === 'string') return value;
  if (value === null || value === undefined) return '';
  return JSON.stringify(value);
}

function Judge() {
  const [view, setView] = useState({ status: 'idle', analysis: null, error: '', cached: false });

  useEffect(() => {
    const receive = (event) => setView(event.detail || { status: 'idle', analysis: null, error: '' });
    window.addEventListener(JUDGE_ANALYSIS_EVENT, receive);
    return () => window.removeEventListener(JUDGE_ANALYSIS_EVENT, receive);
  }, []);

  if (view.status === 'idle' || view.status === 'ready') return null;

  if (view.status === 'loading') {
    return (
      <section className="judge-card judge-loading" id="judge-analysis" aria-live="polite">
        <span className="analysis-spinner" />
        <div><p>ANALYSIS IN PROGRESS</p><h3>Reading the complete court file…</h3></div>
      </section>
    );
  }

  if (view.status === 'error') {
    return (
      <section className="judge-card judge-error" id="judge-analysis" role="alert">
        <p>ANALYSIS COULD NOT BE COMPLETED</p>
        <h3>{view.error}</h3>
      </section>
    );
  }

  const analysis = view.analysis;
  if (!analysis) return null;

  return (
    <section className="judge-card" id="judge-analysis" aria-labelledby="judge-output-title">
      <header className="judge-header">
        <div>
          <p>JUDICIAL REVIEW BRIEF</p>
          <h2 id="judge-output-title">{analysis.case_title || analysis.document?.filename || 'Uploaded case file'}</h2>
          <small>{analysis.document?.filename}{view.cached ? ' · Reused saved analysis' : ' · Newly analysed and saved'}</small>
        </div>
        <div className="judge-count"><strong>{analysis.fact_count}</strong><span>FACT{analysis.fact_count === 1 ? '' : 'S'}</span></div>
      </header>

      <div className="judge-overview">
        <span>CASE OVERVIEW</span>
        <p>{analysis.case_overview || 'No overview was available.'}</p>
      </div>

      {analysis.matched_laws?.length > 0 && (
        <div className="matched-laws">
          <span>DICTIONARY CONTEXT</span>
          <div>{analysis.matched_laws.map((law) => <i key={law}>{law}</i>)}</div>
        </div>
      )}

      <div className="judge-section">
        <div className="judge-section-title"><span>01</span><h3>Material facts</h3></div>
        <div className="fact-list">
          {analysis.facts.map((fact, index) => (
            <article className="fact-card" key={`${textValue(fact.fact)}-${index}`}>
              <div className="fact-meta">
                <span>FACT {String(index + 1).padStart(2, '0')}</span>
                <i className={`fact-status ${fact.status || 'unclear'}`}>{fact.status || 'unclear'}</i>
                {fact.confidence && <i>{fact.confidence} confidence</i>}
              </div>
              <h4>{textValue(fact.fact)}</h4>
              {fact.support && <p><strong>Support:</strong> {textValue(fact.support)}</p>}
              {fact.asserted_by && <p><strong>Asserted by:</strong> {textValue(fact.asserted_by)}</p>}
              {fact.legal_significance && <p><strong>Why it matters:</strong> {textValue(fact.legal_significance)}</p>}
              {fact.page_reference && <small>{textValue(fact.page_reference)}</small>}
            </article>
          ))}
        </div>
      </div>

      {analysis.parties?.length > 0 && (
        <div className="judge-section">
          <div className="judge-section-title"><span>02</span><h3>Parties and roles</h3></div>
          <div className="analysis-grid">
            {analysis.parties.map((party, index) => (
              <article key={`${party.name}-${index}`}><strong>{party.name || 'Unnamed party'}</strong><span>{party.role}</span><p>{party.description}</p></article>
            ))}
          </div>
        </div>
      )}

      {analysis.chronology?.length > 0 && (
        <div className="judge-section">
          <div className="judge-section-title"><span>03</span><h3>Chronology</h3></div>
          <ol className="chronology-list">
            {analysis.chronology.map((item, index) => <li key={`${item.date}-${index}`}><time>{item.date || 'Unknown date'}</time><p>{item.event}</p>{item.source && <small>{item.source}</small>}</li>)}
          </ol>
        </div>
      )}

      {analysis.evidence?.length > 0 && (
        <div className="judge-section">
          <div className="judge-section-title"><span>04</span><h3>Evidence map</h3></div>
          <div className="evidence-table" role="table">
            {analysis.evidence.map((item, index) => (
              <div className="evidence-row" role="row" key={`${item.item}-${index}`}>
                <span>{item.type || 'other'}</span><strong>{item.item}</strong><p>{item.proves_or_rebuts}</p><small>{item.reliability_note}</small>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="judge-two-column">
        {analysis.disputed_facts?.length > 0 && (
          <div className="judge-section compact">
            <div className="judge-section-title"><span>05</span><h3>Disputed points</h3></div>
            <ul>{analysis.disputed_facts.map((item, index) => <li key={`${item.issue}-${index}`}><strong>{item.issue}</strong><p>{item.positions}</p><small>{item.supporting_material}</small></li>)}</ul>
          </div>
        )}
        {analysis.questions_for_determination?.length > 0 && (
          <div className="judge-section compact">
            <div className="judge-section-title"><span>06</span><h3>Questions to determine</h3></div>
            <ul>{analysis.questions_for_determination.map((item, index) => <li key={`${textValue(item)}-${index}`}>{textValue(item)}</li>)}</ul>
          </div>
        )}
      </div>

      {(analysis.credibility_and_reliability?.length > 0 || analysis.missing_information?.length > 0) && (
        <div className="judge-two-column">
          {analysis.credibility_and_reliability?.length > 0 && (
            <div className="judge-section compact"><div className="judge-section-title"><span>07</span><h3>Reliability observations</h3></div><ul>{analysis.credibility_and_reliability.map((item, index) => <li key={`${textValue(item)}-${index}`}>{textValue(item)}</li>)}</ul></div>
          )}
          {analysis.missing_information?.length > 0 && (
            <div className="judge-section compact"><div className="judge-section-title"><span>08</span><h3>Missing information</h3></div><ul>{analysis.missing_information.map((item, index) => <li key={`${textValue(item)}-${index}`}>{textValue(item)}</li>)}</ul></div>
          )}
        </div>
      )}

      {analysis.analysis_limitations?.length > 0 && (
        <div className="analysis-limitations"><strong>Limitations</strong>{analysis.analysis_limitations.map((item, index) => <p key={`${textValue(item)}-${index}`}>{textValue(item)}</p>)}</div>
      )}
      <footer className="judge-disclaimer">{analysis.disclaimer}</footer>
    </section>
  );
}

export default Judge;
