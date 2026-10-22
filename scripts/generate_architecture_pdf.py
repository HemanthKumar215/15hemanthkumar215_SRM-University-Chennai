"""Generate docs/architecture.pdf and editable source docs/architecture.html / docs/architecture.svg."""

from pathlib import Path
from playwright.sync_api import sync_playwright

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

HTML_SOURCE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GlycoTwin End-to-End System Architecture</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
  
  @page {
    size: A4 landscape;
    margin: 10mm;
  }
  
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }
  
  body {
    background: #0B1220;
    color: #F8FAFC;
    font-family: 'Plus Jakarta Sans', sans-serif;
    padding: 24px 32px;
    width: 100%;
    height: 100%;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid rgba(45, 212, 191, 0.3);
    padding-bottom: 12px;
    margin-bottom: 20px;
  }

  .title-group h1 {
    font-size: 22pt;
    font-weight: 800;
    background: linear-gradient(135deg, #2DD4BF 0%, #38BDF8 60%, #8B5CF6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .title-group p {
    font-size: 10pt;
    color: #94A3B8;
    margin-top: 3px;
  }

  .badge-group {
    text-align: right;
    font-size: 9pt;
    color: #CBD5E1;
  }

  .badge-group span {
    display: inline-block;
    background: rgba(45, 212, 191, 0.12);
    border: 1px solid rgba(45, 212, 191, 0.35);
    color: #2DD4BF;
    padding: 4px 10px;
    border-radius: 6px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Grid Layout */
  .architecture-flow {
    display: grid;
    grid-template-columns: 1.1fr 1fr 1.15fr 1.15fr 1.1fr;
    gap: 14px;
    height: calc(100% - 90px);
  }

  .stage-col {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .stage-header {
    font-size: 9.5pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 6px 10px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .header-data { background: rgba(56, 189, 248, 0.15); color: #38BDF8; border-left: 3px solid #38BDF8; }
  .header-ingest { background: rgba(45, 212, 191, 0.15); color: #2DD4BF; border-left: 3px solid #2DD4BF; }
  .header-model { background: rgba(139, 92, 246, 0.15); color: #A78BFA; border-left: 3px solid #8B5CF6; }
  .header-twin { background: rgba(251, 191, 36, 0.15); color: #FBBF24; border-left: 3px solid #FBBF24; }
  .header-ui { background: rgba(52, 211, 153, 0.15); color: #34D399; border-left: 3px solid #34D399; }

  .arch-card {
    background: #121B2E;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 12px 14px;
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
  }

  .arch-card h3 {
    font-size: 10.5pt;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .arch-card .tag {
    font-size: 7pt;
    font-family: 'JetBrains Mono', monospace;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 600;
  }

  .tag-blue { background: rgba(56, 189, 248, 0.15); color: #38BDF8; }
  .tag-teal { background: rgba(45, 212, 191, 0.15); color: #2DD4BF; }
  .tag-purple { background: rgba(139, 92, 246, 0.15); color: #C084FC; }
  .tag-amber { background: rgba(251, 191, 36, 0.15); color: #FBBF24; }
  .tag-green { background: rgba(52, 211, 153, 0.15); color: #34D399; }

  .arch-card ul {
    list-style: none;
    font-size: 8pt;
    color: #94A3B8;
    line-height: 1.45;
    margin-top: 4px;
  }

  .arch-card ul li {
    position: relative;
    padding-left: 14px;
    margin-bottom: 4px;
  }

  .arch-card ul li::before {
    content: "•";
    position: absolute;
    left: 2px;
    color: #2DD4BF;
    font-weight: bold;
  }

  .code-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 7.2pt;
    color: #CBD5E1;
    background: rgba(15, 23, 42, 0.7);
    padding: 4px 6px;
    border-radius: 4px;
    margin-top: 6px;
    border-left: 2px solid #2DD4BF;
  }

  .connector-arrow {
    text-align: center;
    color: rgba(45, 212, 191, 0.5);
    font-size: 11pt;
    line-height: 1;
    margin: -3px 0;
  }

  .footer-bar {
    margin-top: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    padding-top: 10px;
    font-size: 7.8pt;
    color: #64748B;
  }

  /* ---------- Interactive walkthrough (screen only; added by the script below) ---------- */
  .tour-ui { display: none; }
  body.interactive .tour-ui { display: flex; }
  body.interactive { padding-bottom: 96px; }
  body.interactive .stage-col { transition: opacity .3s ease, filter .3s ease, transform .3s ease; }
  body.interactive .stage-header { cursor: pointer; user-select: none; }
  body.interactive .stage-header:hover { filter: brightness(1.25); }
  body.interactive .arch-card { transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease; }
  body.interactive .arch-card:hover {
    transform: translateY(-3px); border-color: rgba(56, 189, 248, 0.45);
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.45);
  }
  body.interactive.focusing .stage-col { opacity: .22; filter: grayscale(.7); }
  body.interactive.focusing .stage-col.focus { opacity: 1; filter: none; transform: translateY(-4px); }
  body.interactive .stage-col.focus .arch-card { border-color: rgba(255, 255, 255, 0.22); }
  body.interactive.focusing .stage-col.done { opacity: .55; filter: grayscale(.3); }
  @media (prefers-reduced-motion: reduce) {
    body.interactive .stage-col, body.interactive .arch-card { transition: none; }
  }

  #tour {
    position: fixed; left: 50%; bottom: 16px; transform: translateX(-50%); z-index: 10;
    width: min(980px, calc(100vw - 32px)); align-items: center; gap: 14px;
    padding: 10px 12px 10px 18px; border-radius: 14px;
    background: rgba(15, 23, 42, 0.94); border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 12px 34px rgba(0, 0, 0, 0.45); backdrop-filter: blur(8px);
  }
  #tour-text { flex: 1; font-size: 10.5pt; line-height: 1.45; color: #CBD5E1; }
  #tour-text b { color: #F8FAFC; }
  #tour-step { font: 700 10pt 'JetBrains Mono', monospace; color: #64748B; white-space: nowrap; }
  #tour button {
    font: 700 10pt 'Plus Jakarta Sans', sans-serif; color: #E2E8F0; background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 9px; height: 36px; padding: 0 14px; cursor: pointer; white-space: nowrap;
  }
  #tour button:hover { background: rgba(255, 255, 255, 0.14); }
  #tour button:focus-visible { outline: 2px solid #38BDF8; outline-offset: 1px; }
  #tour button.primary { background: linear-gradient(135deg, #2DD4BF, #38BDF8); color: #0B1220; border: 0; }
  #tour button:disabled { opacity: .35; cursor: default; }

  @media print {
    body.interactive { padding-bottom: 0; }
    .tour-ui, body.interactive .tour-ui { display: none !important; }
    body.interactive .stage-col { opacity: 1 !important; filter: none !important; transform: none !important; }
    body.interactive .arch-card { transform: none !important; box-shadow: none !important; }
  }
</style>
</head>
<body>

<div class="header">
  <div class="title-group">
    <h1>GlycoTwin • System Architecture</h1>
    <p>Type 2 Diabetes Physiological Digital Twin & Predictive Decision Support Engine</p>
  </div>
  <div class="badge-group">
    <div>Happiest Health Challenge 2026</div>
    <div style="margin-top:4px;"><span>Multi-Modal Deep Fusion (GRU + MLP)</span></div>
  </div>
</div>

<div class="architecture-flow">

  <!-- 1. DATA SOURCES -->
  <div class="stage-col">
    <div class="stage-header header-data">1. Ingestion Sources</div>
    
    <div class="arch-card">
      <h3>Synthetic Cohort <span class="tag tag-blue">500 Pts × 30 d</span></h3>
      <ul>
        <li>Nathan et al. 2008 calibration (r = 0.98)</li>
        <li>Gamma postprandial curves (30–60m peak)</li>
        <li>Full hypnograms & circadian HR / HRV</li>
        <li>Realistic steps (2,000–12,000/day)</li>
      </ul>
      <div class="code-sub">src/data_gen/</div>
    </div>

    <div class="connector-arrow">▼</div>

    <div class="arch-card">
      <h3>Wearable Adapters <span class="tag tag-teal">Real Devices</span></h3>
      <ul>
        <li>Abbott FreeStyle Libre (CSV export)</li>
        <li>Dexcom Clarity CGM format</li>
        <li>Apple Health CSV export</li>
        <li>Simulated AI-READI, CGMacros, Shanghai</li>
      </ul>
      <div class="code-sub">src/data_real/</div>
    </div>
  </div>

  <!-- 2. INGESTION & PIPELINE -->
  <div class="stage-col">
    <div class="stage-header header-ingest">2. In-Memory Validation</div>

    <div class="arch-card">
      <h3>Validator & Sanitizer <span class="tag tag-teal">In-Memory</span></h3>
      <ul>
        <li>Auto-unit conversion (mmol/L → mg/dL)</li>
        <li>5-minute grid uniform resampling</li>
        <li>Cubic spline imputation (<30m gaps)</li>
        <li>Strict 24h duration gate (≥288 rows)</li>
      </ul>
      <div class="code-sub">src/ingestion/validator.py</div>
    </div>

    <div class="connector-arrow">▼</div>

    <div class="arch-card">
      <h3>Leakage Auditing <span class="tag tag-green">Zero-Leakage</span></h3>
      <ul>
        <li>Disjoint patient splits (Train/Val/Test)</li>
        <li>Temporal causality: 24h lookback strictly precedes 2h forecast horizon</li>
        <li>Target: G ≥ 180 mg/dL within 2h given G &lt; 180</li>
      </ul>
      <div class="code-sub">src/features/dataset.py</div>
    </div>
  </div>

  <!-- 3. MODEL ARCHITECTURE -->
  <div class="stage-col">
    <div class="stage-header header-model">3. Multimodal Fusion</div>

    <div class="arch-card">
      <h3>Dynamic Sensor GRU <span class="tag tag-purple">Lookback</span></h3>
      <ul>
        <li>Input: [288 timesteps × 5 channels]</li>
        <li>Channels: CGM, HR, HRV, steps, sleep stage</li>
        <li>2-Layer GRU (64 units, dropout=0.3)</li>
        <li>Final hidden state + LayerNorm → 64-d</li>
      </ul>
      <div class="code-sub">Sensor Branch: 0.9210 AUROC</div>
    </div>

    <div class="connector-arrow">▼</div>

    <div class="arch-card">
      <h3>Static EHR MLP <span class="tag tag-purple">Demographics</span></h3>
      <ul>
        <li>Input: 27 encoded EHR features</li>
        <li>Demographics, labs, diagnoses, meds, genetics</li>
        <li>2-Layer MLP (27 → 64 → 32, LayerNorm)</li>
        <li>Cross-modal concatenation head (96-d)</li>
      </ul>
      <div class="code-sub">Fusion Model: 0.9008 AUROC</div>
    </div>
  </div>

  <!-- 4. TWIN STATE & SIMULATION -->
  <div class="stage-col">
    <div class="stage-header header-twin">4. Digital Twin Engine</div>

    <div class="arch-card">
      <h3>PatientTwin State <span class="tag tag-amber">Encapsulation</span></h3>
      <ul>
        <li>Real-time sensor trace & rolling stats</li>
        <li>Calibrated baseline physiological profile</li>
        <li>Active 2h spike probability & risk tier</li>
        <li>Status: Optimal, Watch, or Hyperglycemic</li>
      </ul>
      <div class="code-sub">src/twin/patient_twin.py</div>
    </div>

    <div class="connector-arrow">▼</div>

    <div class="arch-card">
      <h3>What-If Simulator <span class="tag tag-amber">Counterfactual</span></h3>
      <ul>
        <li>Post-meal brisk walk (GLUT4 translocation)</li>
        <li>Carbohydrate challenge / snack omission</li>
        <li>Sleep hygiene improvement</li>
        <li>Projected risk change (↑/↓ pts)</li>
      </ul>
      <div class="code-sub">PatientTwin.what_if()</div>
    </div>
  </div>

  <!-- 5. CLINICAL UI & WORKFLOW -->
  <div class="stage-col">
    <div class="stage-header header-ui">5. Clinician Interface</div>

    <div class="arch-card">
      <h3>Decision Dashboard <span class="tag tag-green">Streamlit</span></h3>
      <ul>
        <li>10-second Executive Summary & Result Card</li>
        <li>Interactive CGM spline & 2h forecast cone</li>
        <li>Ambulatory Glucose Profile (AGP) modal day</li>
        <li>Clinical Light & Dark Theme (WCAG AA)</li>
      </ul>
      <div class="code-sub">app/streamlit_app.py</div>
    </div>

    <div class="connector-arrow">▼</div>

    <div class="arch-card">
      <h3>Explainability & Audit <span class="tag tag-green">Occlusion</span></h3>
      <ul>
        <li>Signed horizontal feature impacts (% risk)</li>
        <li>Scientific Model Report & fair baselines</li>
        <li>F1/Precision/Recall vs threshold curve</li>
        <li>In-memory private drag-and-drop upload</li>
      </ul>
      <div class="code-sub">app/ui_theme.py</div>
    </div>
  </div>

</div>

<div class="footer-bar">
  <div>GlycoTwin Architecture Specification • MIT Open Source License • 100% In-Memory Privacy</div>
  <div>Benchmarks: LightGBM Fair 0.9699 | Fusion Model 0.9008 | Transfer Learning 0.9338 AUROC</div>
</div>

<div id="tour" class="tour-ui" role="region" aria-label="Architecture walkthrough">
  <span id="tour-step"></span>
  <div id="tour-text" aria-live="polite"></div>
  <button id="tour-prev" aria-label="Previous stage">&larr;</button>
  <button id="tour-next" class="primary">&#9654; Walk through</button>
  <button id="tour-reset" aria-label="Show all stages">Show all</button>
</div>

<script>
(function () {
  var cols = Array.prototype.slice.call(document.querySelectorAll('.stage-col'));
  if (!cols.length) return;
  var body = document.body;
  body.classList.add('interactive');

  var captions = [
    '<b>Two data streams in.</b> Static EHR profiles (demographics, labs, diagnoses, meds, genetic risk) and 5-minute wearable streams (CGM, heart rate, HRV, steps, sleep) come from the synthetic cohort or real-device exports.',
    '<b>Validated in memory.</b> Uploads are format-detected (Libre, Dexcom, Apple Health), unit-checked, resampled to a 5-minute grid and split by patient with strict time ordering, so nothing leaks into the future.',
    '<b>Fusion model.</b> A 2-layer GRU encodes the last 24 h of sensors (64-d) and an MLP encodes 27 EHR features (32-d); the concatenated 96-d vector predicts P(glucose &ge; 180 mg/dL within 2 h).',
    '<b>The digital twin.</b> PatientTwin keeps a rolling sensor window, re-predicts as readings arrive, and runs what-if counterfactuals (walk, skip or add a meal, better sleep) to show the change in risk.',
    '<b>The clinician acts.</b> The Streamlit dashboard shows the forecast cone, risk tier, feature explanations and what-if results so the doctor can choose an intervention before the spike happens.'
  ];
  var $ = function (id) { return document.getElementById(id); };
  var step = -1;

  function render() {
    body.classList.toggle('focusing', step >= 0);
    cols.forEach(function (c, i) {
      c.classList.toggle('focus', i === step);
      c.classList.toggle('done', step >= 0 && i < step);
    });
    if (step < 0) {
      $('tour-step').textContent = '';
      $('tour-text').innerHTML = 'Click a stage heading to focus it, or walk through the data flow step by step (&larr; / &rarr; keys).';
      $('tour-next').innerHTML = '&#9654; Walk through';
    } else {
      $('tour-step').textContent = (step + 1) + ' / ' + cols.length;
      $('tour-text').innerHTML = captions[step] || '';
      $('tour-next').innerHTML = step === cols.length - 1 ? 'Finish' : 'Next &rarr;';
    }
    $('tour-prev').disabled = step <= 0;
  }
  function go(i) { step = Math.max(-1, Math.min(cols.length - 1, i)); render(); }

  $('tour-next').onclick = function () { go(step === cols.length - 1 ? -1 : step + 1); };
  $('tour-prev').onclick = function () { go(step - 1); };
  $('tour-reset').onclick = function () { go(-1); };
  cols.forEach(function (c, i) {
    var h = c.querySelector('.stage-header');
    h.setAttribute('role', 'button');
    h.setAttribute('tabindex', '0');
    h.title = 'Focus this stage';
    var pick = function () { go(step === i ? -1 : i); };
    h.addEventListener('click', pick);
    h.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
  });
  document.addEventListener('keydown', function (e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === 'ArrowRight') go(step + 1);
    else if (e.key === 'ArrowLeft') go(step - 1);
    else if (e.key === 'Escape') go(-1);
    else return;
    e.preventDefault();
  });
  render();
})();
</script>

</body>
</html>
"""


def main():
    html_path = DOCS_DIR / "architecture.html"
    pdf_path = DOCS_DIR / "architecture.pdf"
    svg_path = DOCS_DIR / "architecture.svg"

    html_path.write_text(HTML_SOURCE, encoding="utf-8")
    print(f"Wrote editable HTML source: {html_path}")

    # Generate high quality vector PDF via Edge/Playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=EDGE_PATH if Path(EDGE_PATH).exists() else None,
                                    headless=True)
        page = browser.new_page()
        page.set_content(HTML_SOURCE, wait_until="networkidle")
        page.pdf(
            path=str(pdf_path),
            format="A4",
            landscape=True,
            scale=0.9,  # keeps the whole diagram on a single A4 page
            print_background=True,
            margin=dict(top="8mm", bottom="8mm", left="8mm", right="8mm"),
        )
        browser.close()
    print(f"Generated vector architecture PDF: {pdf_path}")


if __name__ == "__main__":
    main()
