import { useEffect, useState } from 'react';
import { LAWYER_ANALYSIS_EVENT } from './lawyerUpload';

function textValue(value) {
  if (typeof value === 'string' || typeof value === 'number') return String(value);
  if (value === null || value === undefined) return '';
  return JSON.stringify(value);
}

function humanize(value) {
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function DetailValue({ value }) {
  if (Array.isArray(value)) return <ul>{value.map((item, index) => <li key={`${textValue(item)}-${index}`}>{textValue(item)}</li>)}</ul>;
  return <p>{textValue(value)}</p>;
}

function StrategySection({ number, title, items }) {
  if (!Array.isArray(items) || items.length === 0) return null;
  return (
    <div className="judge-section lawyer-section">
      <div className="judge-section-title"><span>{String(number).padStart(2, '0')}</span><h3>{title}</h3></div>
      <div className="lawyer-entry-list">
        {items.map((item, index) => (
          <article className="lawyer-entry" key={`${title}-${index}`}>
            {typeof item === 'object' && item !== null
              ? Object.entries(item).map(([key, value]) => value !== null && value !== '' && (!Array.isArray(value) || value.length > 0) && (
                <div className="lawyer-entry-field" key={key}><strong>{humanize(key)}</strong><DetailValue value={value} /></div>
              ))
              : <p>{textValue(item)}</p>}
          </article>
        ))}
      </div>
    </div>
  );
}

const sections = [
  ['objectives', 'Objectives'],
  ['case_theories', 'Competing case theories'],
  ['elements_and_issues', 'Elements and legal issues'],
  ['strengths', 'Strategic strengths'],
  ['vulnerabilities', 'Vulnerabilities and responses'],
  ['contradictions', 'Record contradictions'],
  ['evidence_assessment', 'Evidence assessment'],
  ['procedural_opportunities', 'Procedural opportunities'],
  ['legal_arguments', 'Arguments to develop'],
  ['investigation_plan', 'Investigation plan'],
  ['disclosure_requests', 'Disclosure and production requests'],
  ['witness_plan', 'Witness plan'],
  ['applications_and_motions', 'Applications and motions'],
  ['negotiation_considerations', 'Negotiation considerations'],
  ['hearing_trial_plan', 'Hearing and trial plan'],
  ['deadlines_and_preservation', 'Deadlines and preservation'],
  ['risk_register', 'Risk register'],
  ['unresolved_questions', 'Unresolved questions'],
  ['missing_information', 'Missing information'],
  ['next_actions', 'Prioritized next actions'],
  ['ethics_and_safety', 'Professional safeguards'],
];

function Lawyer() {
  const [view, setView] = useState({ status: 'idle', analysis: null, error: '', cached: false });

  useEffect(() => {
    const receive = (event) => setView(event.detail || { status: 'idle', analysis: null, error: '', cached: false });
    window.addEventListener(LAWYER_ANALYSIS_EVENT, receive);
    return () => window.removeEventListener(LAWYER_ANALYSIS_EVENT, receive);
  }, []);

  if (view.status === 'idle' || view.status === 'ready') return null;
  if (view.status === 'loading') return (
    <section className="judge-card judge-loading lawyer-card" id="lawyer-analysis" aria-live="polite">
      <span className="analysis-spinner" /><div><p>STRATEGY ANALYSIS IN PROGRESS</p><h3>Mapping the complete record for counsel…</h3></div>
    </section>
  );
  if (view.status === 'error') return (
    <section className="judge-card judge-error lawyer-card" id="lawyer-analysis" role="alert"><p>ANALYSIS COULD NOT BE COMPLETED</p><h3>{view.error}</h3></section>
  );

  const analysis = view.analysis;
  if (!analysis) return null;
  return (
    <section className="judge-card lawyer-card" id="lawyer-analysis" aria-labelledby="lawyer-output-title">
      <header className="judge-header lawyer-header">
        <div>
          <p>LAWYER STRATEGY WORKING BRIEF</p>
          <h2 id="lawyer-output-title">{analysis.case_title || analysis.document?.filename || 'Uploaded case file'}</h2>
          <small>{analysis.document?.filename}{view.cached ? ' · Reused saved analysis' : ' · Newly analysed and saved'}</small>
        </div>
        <div className="judge-count"><strong>{analysis.fact_count}</strong><span>STRATEGY FACT{analysis.fact_count === 1 ? '' : 'S'}</span></div>
      </header>
      <div className="judge-overview"><span>CASE OVERVIEW</span><p>{analysis.case_overview || 'No overview was available.'}</p></div>
      {analysis.client_position && <div className="lawyer-position"><span>CLIENT POSITION</span><p>{analysis.client_position}</p></div>}
      {analysis.matched_laws?.length > 0 && <div className="matched-laws"><span>LEGAL RESEARCH CONTEXT</span><div>{analysis.matched_laws.map((law) => <i key={law}>{law}</i>)}</div></div>}

      <div className="judge-section">
        <div className="judge-section-title"><span>01</span><h3>Strategy facts</h3></div>
        <div className="fact-list">
          {analysis.strategy_facts.map((fact, index) => (
            <article className="fact-card lawyer-fact-card" key={`${textValue(fact.fact)}-${index}`}>
              <div className="fact-meta"><span>FACT {String(index + 1).padStart(2, '0')}</span><i>{fact.status || 'unclear'}</i>{fact.confidence && <i>{fact.confidence} confidence</i>}</div>
              <h4>{textValue(fact.fact)}</h4>
              {fact.strategy_use && <p><strong>Strategy use:</strong> {textValue(fact.strategy_use)}</p>}
              {fact.source && <p><strong>Source:</strong> {textValue(fact.source)}</p>}
              {fact.page_reference && <small>{textValue(fact.page_reference)}</small>}
            </article>
          ))}
        </div>
      </div>

      {sections.map(([field, title], index) => <StrategySection key={field} number={index + 2} title={title} items={analysis[field]} />)}
      {analysis.analysis_limitations?.length > 0 && <div className="analysis-limitations"><strong>Limitations</strong>{analysis.analysis_limitations.map((item, index) => <p key={`${textValue(item)}-${index}`}>{textValue(item)}</p>)}</div>}
      <footer className="judge-disclaimer">{analysis.disclaimer}</footer>
    </section>
  );
}

export default Lawyer;
