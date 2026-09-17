import { useMemo, useState } from 'react';
import './Home.css';
import Upload from './upload';
import Judge from './judge';
import LawyerUpload from './lawyerUpload';
import Lawyer from './lawyer';
import Chat from './Chat';

const roles = [
  { id: 'judge', number: '01', label: 'Judge', detail: 'Precedents, ratio decidendi, conflicts, and procedural posture.' },
  { id: 'lawyer', number: '02', label: 'Lawyer', detail: 'Arguments, authorities, risks, and practical litigation strategy.' },
  { id: 'public', number: '03', label: 'General public', detail: 'Plain-language facts, the decision, and what it means in practice.' },
];

const ArrowIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
);

const ScaleMark = () => (
  <svg className="brand-mark" viewBox="0 0 44 44" aria-hidden="true">
    <path d="M22 7v28M13 12h18M9 34h26M10 38h24" />
    <path d="m13 12-6 11h12L13 12Zm18 0-6 11h12L31 12Z" />
    <path d="M7 23c0 4 12 4 12 0M25 23c0 4 12 4 12 0" />
  </svg>
);

function Home() {
  const [selectedRole, setSelectedRole] = useState('lawyer');
  const [query, setQuery] = useState('');
  const activeRole = useMemo(() => roles.find((role) => role.id === selectedRole), [selectedRole]);
  const analysisCopy = selectedRole === 'lawyer'
    ? {
        eyebrow: 'LAWYER CASE STRATEGY',
        title: <>Build the record.<br />Plan the next move.</>,
        description: 'Upload a text-based PDF to extract source-grounded strategy facts, strengths, vulnerabilities, evidence issues, investigation tasks, disclosure needs, witness themes, risks, and prioritized next actions.',
      }
    : selectedRole === 'judge'
      ? {
          eyebrow: 'JUDICIAL CASE ANALYSIS',
          title: <>Extract the record.<br />Keep the decision human.</>,
          description: 'Upload a text-based PDF to identify material facts, evidence, disputed positions, chronology, reliability concerns, and questions requiring judicial determination.',
        }
      : {
          eyebrow: 'PLAIN-LANGUAGE VIEW',
          title: <>Understand the record.<br />Without legal shorthand.</>,
          description: 'The public PDF analysis component is not connected yet. Select Judge or Lawyer above to open the corresponding upload and output component.',
        };

  const submitQuery = (event) => {
    event.preventDefault();
    if (!query.trim()) return;
    // The summary API will be connected in the next backend phase.
    document.getElementById('workspace')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <main className="home-shell">
      <nav className="navbar" aria-label="Primary navigation">
        <a className="brand" href="#top" aria-label="LexBrief home"><ScaleMark /><span>LEXBRIEF</span></a>
        <div className="nav-links">
          <a href="#how-it-works">How it works</a>
          <a href="#roles">For professionals</a>
          <a className="nav-button" href="#workspace">Enter workspace</a>
        </div>
      </nav>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow"><span />ROLE-AWARE LEGAL INTELLIGENCE</p>
          <h1>One case.<br />The perspective<br /><em>you need.</em></h1>
          <p className="hero-description">Turn dense judgments into precise, role-specific briefs. LexBrief adapts the same authority for the bench, the bar, and everyone else.</p>
          <div className="hero-actions">
            <a className="primary-action" href="#workspace">Summarize a case <ArrowIcon /></a>
            <a className="text-action" href="#how-it-works">See how it works</a>
          </div>
        </div>

        <div className="hero-visual" aria-label="Example role-based legal summary">
          <div className="document-stack document-stack-one" />
          <div className="document-stack document-stack-two" />
          <article className="sample-document">
            <header><span>SUPREME COURT · 2025</span><span>12 PAGES</span></header>
            <p className="sample-citation">CIVIL APPEAL NO. 1842</p>
            <h2>Arden Industries<br />v. State Commission</h2>
            <div className="sample-rule" />
            <p className="sample-label">CORE HOLDING</p>
            <p className="sample-text">Administrative discretion remains subject to proportionality and a recorded statement of reasons.</p>
            <div className="sample-lines"><i /><i /><i /><i /></div>
            <footer><span>ANALYSED FOR</span><strong>{activeRole.label.toUpperCase()}</strong></footer>
          </article>
          <div className="insight-card"><span className="insight-index">{activeRole.number}</span><div><p>CURRENT LENS</p><strong>{activeRole.label}</strong></div></div>
        </div>
      </section>

      <section className="role-section" id="roles">
        <div className="section-heading">
          <p className="eyebrow light"><span />CHOOSE YOUR PERSPECTIVE</p>
          <h2>The law reads differently<br />from every seat.</h2>
        </div>
        <div className="role-grid">
          {roles.map((role) => (
            <button className={`role-card ${selectedRole === role.id ? 'active' : ''}`} key={role.id} type="button" onClick={() => setSelectedRole(role.id)} aria-pressed={selectedRole === role.id}>
              <span className="role-number">{role.number}</span>
              <span className="role-name">{role.label}</span>
              <span className="role-detail">{role.detail}</span>
              <span className="role-arrow"><ArrowIcon /></span>
            </button>
          ))}
        </div>
      </section>

      <section className="workspace-section" id="workspace">
        <div className="workspace-copy">
          <p className="eyebrow"><span />START WITH THE SOURCE</p>
          <h2>Bring the judgment.<br />We’ll find the signal.</h2>
          <p>Paste case text, a citation, or a judgment URL. Your brief will be structured for a <strong>{activeRole.label.toLowerCase()}</strong>.</p>
        </div>
        <form className="query-panel" onSubmit={submitQuery}>
          <div className="query-meta"><span>YOUR LEGAL MATERIAL</span><span>ROLE · {activeRole.label.toUpperCase()}</span></div>
          <textarea value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Paste a citation, judgment URL, or case text here…" aria-label="Legal material" />
          <div className="query-footer">
            <p>AI-generated legal research support. Always verify primary sources.</p>
            <button type="submit" disabled={!query.trim()}>Generate brief <ArrowIcon /></button>
          </div>
        </form>
      </section>

      <section className="judge-workspace" id="pdf-analysis">
        <div className="judge-workspace-intro">
          <p className="eyebrow"><span />{analysisCopy.eyebrow}</p>
          <h2>{analysisCopy.title}</h2>
          <p>{analysisCopy.description}</p>
        </div>
        {selectedRole === 'judge' && <><Upload /><Judge /></>}
        {selectedRole === 'lawyer' && <><LawyerUpload /><Lawyer /></>}
      </section>

      <section className="process-section" id="how-it-works">
        <div><span>01</span><strong>Provide the authority</strong><p>Paste the judgment, citation, or source URL.</p></div>
        <div><span>02</span><strong>Select your role</strong><p>Choose the legal lens that matches your work.</p></div>
        <div><span>03</span><strong>Receive a focused brief</strong><p>Review structured facts, holdings, and implications.</p></div>
      </section>

      <footer className="site-footer">
        <a className="brand footer-brand" href="#top"><ScaleMark /><span>LEXBRIEF</span></a>
        <p>Clarity for every side of the law.</p>
        <span>© 2026 LEXBRIEF</span>
      </footer>
      <Chat activeRole={selectedRole} />
    </main>
  );
}

export default Home;
