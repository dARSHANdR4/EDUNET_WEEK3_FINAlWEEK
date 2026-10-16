import { useEffect, useState } from 'react';
import { Activity, Sprout, FlaskConical, HeartPulse, ShieldCheck, ArrowUpRight, LoaderCircle, Leaf, Gauge, Info } from 'lucide-react';

const API = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');
const cropFields = [
  ['N', 'Nitrogen (N)', '90', 0, 200], ['P', 'Phosphorus (P)', '42', 0, 200], ['K', 'Potassium (K)', '43', 0, 250],
  ['temperature', 'Temperature °C', '22', -10, 60], ['humidity', 'Humidity %', '80', 0, 100],
  ['ph', 'Soil pH', '6.5', 0, 14], ['rainfall', 'Rainfall mm', '200', 0, 5000]
];
const diabetesFields = [
  ['Pregnancies','Pregnancies','1',0,20], ['Glucose','Glucose (mg/dL)','120',0,300], ['BloodPressure','Blood pressure (mm Hg)','70',0,200],
  ['SkinThickness','Skin thickness (mm)','20',0,100], ['Insulin','Insulin (µU/mL)','79',0,900], ['BMI','BMI','25.8',0,80],
  ['DiabetesPedigreeFunction','Diabetes pedigree function','0.5',0,3], ['Age','Age','33',1,120]
];
const fertilizerFields = [
  ['Temperature','Temperature °C','26',-10,60], ['Humidity','Humidity %','52',0,100], ['Moisture','Soil moisture','38',0,100],
  ['Soil Type','Soil type','Loamy'], ['Crop Type','Crop type','Maize'],
  ['Nitrogen','Nitrogen','37',0,200], ['Potassium','Potassium','0',0,200], ['Phosphorous','Phosphorous','0',0,200]
];
const cropOptions = ['Barley','Cotton','Ground Nuts','Maize','Millets','Oil seeds','Paddy','Pulses','Sugarcane','Tobacco','Wheat'];
const soilOptions = ['Black','Clayey','Loamy','Red','Sandy'];
const cropTypeOptions = ['Barley','Cotton','Ground Nuts','Maize','Millets','Oil seeds','Paddy','Pulses','Sugarcane','Tobacco','Wheat'];
const modules = [
  { id: 'crop', title: 'Crop recommendation', sub: 'Find a crop suited to your soil and climate.', icon: Sprout, tone: 'green', endpoint: '/api/crop/predict', fields: cropFields, action: 'Recommend crop' },
  { id: 'fertilizer', title: 'Fertilizer recommendation', sub: 'Get a fertilizer suggestion from soil and crop inputs.', icon: FlaskConical, tone: 'violet', endpoint: '/api/fertilizer/predict', fields: fertilizerFields, action: 'Recommend fertilizer' },
  { id: 'diabetes', title: 'Diabetes screening', sub: 'Explore a model prediction from health measurements.', icon: HeartPulse, tone: 'rose', endpoint: '/api/diabetes/predict', fields: diabetesFields, action: 'Run screening' }
];

function Field({ field, value, onChange }) {
  const [key, label, initial, min, max] = field;
  const options = key === 'Soil Type' ? soilOptions : key === 'Crop Type' ? cropTypeOptions : null;
  return <label className="field"><span>{label}</span>{options ?
    <select value={value} onChange={e => onChange(key, e.target.value)}>{options.map(o => <option key={o}>{o}</option>)}</select> :
    <input type="number" value={value} min={min} max={max} step={key === 'ph' || key === 'BMI' || key === 'DiabetesPedigreeFunction' ? '0.1' : '1'} onChange={e => onChange(key, e.target.value)} required />}</label>;
}
function App() {
  const [active, setActive] = useState('crop');
  const [values, setValues] = useState(() => Object.fromEntries(modules.map(m => [m.id, Object.fromEntries(m.fields.map(f => [f[0], f[2]]))])));
  const [results, setResults] = useState({});
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking');
  const current = modules.find(m => m.id === active);
  const Icon = current.icon;
  const update = (key, value) => setValues(prev => ({ ...prev, [active]: { ...prev[active], [key]: value } }));
  async function checkApi() {
    setApiStatus('checking');
    try { const r = await fetch(API + '/health'); setApiStatus(r.ok ? 'online' : 'offline'); }
    catch { setApiStatus('offline'); }
  }
  useEffect(() => { checkApi(); }, []);
  async function submit(e) {
    e.preventDefault(); setLoading(true); setResults(prev => ({ ...prev, [active]: null }));
    try {
      const payload = Object.fromEntries(Object.entries(values[active]).map(([k,v]) => [k, typeof v === 'string' && v.trim() !== '' && !['Soil Type','Crop Type'].includes(k) ? Number(v) : v]));
      const response = await fetch(API + current.endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || 'The prediction service could not process this request.');
      setResults(prev => ({ ...prev, [active]: { data } }));
    } catch (err) { setResults(prev => ({ ...prev, [active]: { error: err.message } })); }
    finally { setLoading(false); }
  }
  const result = results[active];
  const output = result?.data?.prediction ?? result?.data?.result ?? result?.data?.label;
  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><Activity size={22}/></div><div><b>AI Studio</b><small>Prediction workspace</small></div></div>
      <div className="nav-label">WORKSPACE</div>
      {modules.map(m => { const I = m.icon; return <button key={m.id} className={'nav-item ' + (active === m.id ? 'selected' : '')} onClick={() => setActive(m.id)}><I size={18}/><span>{m.title}</span><ArrowUpRight size={15} className="nav-arrow"/></button>; })}
      <div className="sidebar-bottom"><div className="status-card"><span className={'status-dot ' + apiStatus}></span><div><b>API connection</b><small>{apiStatus === 'online' ? 'Backend reachable' : apiStatus === 'checking' ? 'Checking service…' : 'Backend not connected'}</small></div><button onClick={checkApi} title="Check API again">↻</button></div><p>Models are served through a separate Python API.</p></div>
    </aside>
    <main className="main">
      <header className="topbar"><div className="breadcrumb">Workspace <span>/</span> <b>{current.title}</b></div><div className="top-pill"><span className="live-dot"></span> Prototype environment</div></header>
      <section className="hero"><div><div className="eyebrow"><span className={'eyebrow-icon ' + current.tone}><Icon size={15}/></span> MACHINE LEARNING MODULE</div><h1>{current.title}</h1><p>{current.sub}</p></div><div className="hero-badge"><ShieldCheck size={17}/> Input validation enabled</div></section>
      <section className="stats-grid">
        <div className="stat-card"><div className="stat-icon blue"><Gauge size={18}/></div><div><small>Inference mode</small><b>On demand</b></div></div>
        <div className="stat-card"><div className="stat-icon green"><Leaf size={18}/></div><div><small>Prediction modules</small><b>03 available</b></div></div>
        <div className="stat-card"><div className="stat-icon amber"><Activity size={18}/></div><div><small>Service status</small><b>{apiStatus === 'online' ? 'Connected' : apiStatus === 'checking' ? 'Checking…' : 'Needs setup'}</b></div></div>
      </section>
      <div className="content-grid">
        <section className="panel form-panel"><div className="panel-heading"><div><span className="section-kicker">INPUT PARAMETERS</span><h2>Enter your data</h2></div><span className="required-tag">* Required</span></div>
          <form onSubmit={submit}><div className={'fields-grid ' + (active === 'diabetes' ? 'three-col' : '')}>{current.fields.map(f => <Field key={f[0]} field={f} value={values[active][f[0]]} onChange={update}/>)}</div>
            <div className="form-note"><Info size={16}/><span>{active === 'diabetes' ? 'Use measured values where possible. This model is a learning/demo tool, not a medical diagnosis.' : 'Predictions depend on the training data and the range of inputs it represents.'}</span></div>
            <button className="submit-button" type="submit" disabled={loading}>{loading ? <><LoaderCircle className="spin" size={17}/> Running model…</> : <>{current.action}<ArrowUpRight size={17}/></>}</button>
          </form>
        </section>
        <section className="panel result-panel"><div className="panel-heading"><div><span className="section-kicker">MODEL OUTPUT</span><h2>Prediction result</h2></div><span className="result-symbol"><Activity size={18}/></span></div>
          {!result && <div className="empty-state"><div className="empty-illustration"><Icon size={30}/></div><h3>Your result will appear here</h3><p>Fill in the parameters and run the model to see the prediction returned by the API.</p><div className="pipeline"><span>Input</span><i></i><span>API</span><i></i><span>ML model</span></div></div>}
          {result?.error && <div className="error-box"><b>Prediction unavailable</b><p>{result.error}</p><small>Start the backend and configure the required model datasets, then retry.</small></div>}
          {result?.data && <div className="result-content"><div className="success-label"><span className="status-dot online"></span> Prediction returned</div><div className="prediction-value">{String(output ?? 'Result received')}</div><p className="result-caption">Returned by the connected model service.</p><div className="result-details">{Object.entries(result.data).filter(([k]) => !['prediction','result','label'].includes(k)).map(([k,v]) => <div key={k}><span>{k.replaceAll('_',' ')}</span><b>{typeof v === 'object' ? JSON.stringify(v) : String(v)}</b></div>)}</div><button className="secondary-button" onClick={() => setResults(prev => ({ ...prev, [active]: null }))}>Clear result</button></div>}
          <div className="privacy-note"><ShieldCheck size={16}/><span>Inputs are sent to the configured API for inference. Do not submit identifying or sensitive personal details.</span></div>
        </section>
      </div>
      <footer>AI Prediction Studio <span>•</span> React frontend + FastAPI middleware <span>•</span> Demo build 0.1</footer>
    </main>
  </div>;
}
export default App;