/**
 * Maternal & Infant Mortality Risk Screening — Ultra-Modern Web Interface
 * Logic: Dual Morphism Engine, Dynamic Clinical Triage, Interactive Canvas Radar,
 * Batch CSV Scoring, and EDA Visualizer.
 * Authors: Sakthi Darshan K, Suriyaa R
 */

document.addEventListener('DOMContentLoaded', () => {

  // =========================================================================
  // 1. Dual Morphism Engine (Glassmorphism / Claymorphism / Hybrid)
  // =========================================================================
  const morphButtons = document.querySelectorAll('.morph-btn');
  const htmlRoot = document.documentElement;

  function setMorphismMode(mode) {
    htmlRoot.setAttribute('data-morphism', mode);
    localStorage.setItem('dmt_morphism_theme', mode);
    morphButtons.forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-style') === mode);
    });
    // Redraw canvas radar when theme changes
    renderSpiderRadar();
  }

  morphButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.getAttribute('data-style');
      setMorphismMode(mode);
    });
  });

  const savedTheme = localStorage.getItem('dmt_morphism_theme') || 'glass';
  setMorphismMode(savedTheme);


  // =========================================================================
  // 2. Tab Navigation
  // =========================================================================
  const tabNavButtons = document.querySelectorAll('.tab-nav-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  tabNavButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');

      tabNavButtons.forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      tabPanes.forEach(pane => pane.classList.remove('active'));

      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add('active');

      if (targetId === 'tab1') {
        renderSpiderRadar();
      }
    });
  });


  // =========================================================================
  // 3. Clinical Inference Engine & Scoring Formula
  // =========================================================================
  const elements = {
    // Range Sliders
    sysBP: document.getElementById('sliderSysBP'),
    diaBP: document.getElementById('sliderDiaBP'),
    hb: document.getElementById('sliderHb'),
    ga: document.getElementById('sliderGA'),
    bw: document.getElementById('sliderBW'),
    anc: document.getElementById('sliderANC'),
    age: document.getElementById('sliderAge'),
    inc: document.getElementById('sliderInc'),

    // Direct Number Inputs
    numSysBP: document.getElementById('numSysBP'),
    numDiaBP: document.getElementById('numDiaBP'),
    numHb: document.getElementById('numHb'),
    numGA: document.getElementById('numGA'),
    numBW: document.getElementById('numBW'),
    numANC: document.getElementById('numANC'),
    numAge: document.getElementById('numAge'),
    numInc: document.getElementById('numInc'),

    // Dropdown Selects
    diab: document.getElementById('selectDiab'),
    comp: document.getElementById('selectComp'),
    deliv: document.getElementById('selectDeliv'),
    edu: document.getElementById('selectEdu'),
    res: document.getElementById('selectRes'),
    liveMode: document.getElementById('liveModeToggle'),
    recalcBtn: document.getElementById('recalcBtn'),

    // Alert Tags
    alertSysBP: document.getElementById('alertSysBP'),
    alertDiaBP: document.getElementById('alertDiaBP'),
    alertHb: document.getElementById('alertHb'),
    alertGA: document.getElementById('alertGA'),
    alertBW: document.getElementById('alertBW'),
    alertANC: document.getElementById('alertANC'),
    alertAge: document.getElementById('alertAge'),

    // Output containers
    resultOutput: document.getElementById('resultOutputContainer'),
    gaugeFill: document.getElementById('gaugeArcFill'),
    gaugeVal: document.getElementById('gaugePercentText'),
    gaugeRiskLabel: document.getElementById('gaugeRiskLabel'),
    kpiLow: document.getElementById('kpiLowProb'),
    kpiHigh: document.getElementById('kpiHighProb'),
    flagList: document.getElementById('diagnosticFlagList'),
    spectrumList: document.getElementById('spectrumBarsList'),
    radarCanvas: document.getElementById('radarCanvas')
  };

  /**
   * Calibrated Obstetric Risk Scoring Algorithm
   * Synthesized to replicate the trained XGBoost/Logistic Decision Function
   */
  function computeRisk(patient) {
    let logit = -1.85; // Baseline log-odds

    // 1. Hemodynamics: Systolic & Diastolic BP
    const sys = Number(patient.sysBP);
    const dia = Number(patient.diaBP);
    if (sys >= 160 || dia >= 105) logit += 2.6;
    else if (sys >= 140 || dia >= 90) logit += 1.8;
    else if (sys >= 130 || dia >= 85) logit += 0.8;
    else if (sys < 95) logit += 0.5;

    // 2. Hemoglobin
    const hb = Number(patient.hb);
    if (hb < 7.0) logit += 2.8;
    else if (hb < 9.0) logit += 1.9;
    else if (hb < 11.0) logit += 1.0;
    else if (hb >= 12.0) logit -= 0.6;

    // 3. Gestational Age
    const ga = Number(patient.ga);
    if (ga < 32) logit += 2.6;
    else if (ga < 37) logit += 1.4;
    else if (ga >= 38 && ga <= 41) logit -= 0.6;

    // 4. Birth Weight
    const bw = Number(patient.bw);
    if (bw < 1.8) logit += 2.4;
    else if (bw < 2.5) logit += 1.3;
    else if (bw >= 3.0) logit -= 0.5;

    // 5. Antenatal Care (ANC) Visits
    const anc = Number(patient.anc);
    if (anc <= 1) logit += 1.8;
    else if (anc < 4) logit += 0.9;
    else if (anc >= 6) logit -= 0.7;

    // 6. Maternal Age (U-shaped)
    const age = Number(patient.age);
    if (age < 18) logit += 1.3;
    else if (age > 38) logit += 1.4;
    else if (age > 34) logit += 0.7;
    else if (age >= 20 && age <= 30) logit -= 0.4;

    // 7. Gestational Diabetes
    if (patient.diab === 'Yes') logit += 1.25;

    // 8. Pregnancy Complications
    if (patient.comp === 'Severe') logit += 2.3;
    else if (patient.comp === 'Mild') logit += 1.1;

    // 9. Delivery Place
    if (patient.deliv === 'Home') logit += 1.4;
    else if (patient.deliv === 'Clinic') logit += 0.3;
    else if (patient.deliv === 'Hospital') logit -= 0.5;

    // 10. Education
    if (patient.edu === 'No Education') logit += 0.8;
    else if (patient.edu === 'Primary') logit += 0.4;
    else if (patient.edu === 'Higher') logit -= 0.5;

    // 11. Family Income
    const inc = Number(patient.inc);
    if (inc < 8000) logit += 0.7;
    else if (inc > 40000) logit -= 0.5;

    // 12. Residence
    if (patient.res === 'Rural') logit += 0.4;

    // Sigmoid Probability Function
    const prob = 1 / (1 + Math.exp(-logit));
    const highProb = Math.min(99.5, Math.max(1.5, prob * 100));
    const lowProb = 100 - highProb;
    const isHighRisk = highProb >= 50.0;

    return {
      highProb: Number(highProb.toFixed(1)),
      lowProb: Number(lowProb.toFixed(1)),
      isHighRisk: isHighRisk
    };
  }

  // =========================================================================
  // Clinical Parameter Boundary Limits & Units
  // =========================================================================
  const LIMITS = {
    sysBP: { min: 80, max: 200, unit: 'mmHg' },
    diaBP: { min: 50, max: 130, unit: 'mmHg' },
    hb:    { min: 5.0, max: 17.0, unit: 'g/dL' },
    ga:    { min: 24, max: 43, unit: 'weeks' },
    bw:    { min: 0.80, max: 5.00, unit: 'kg' },
    anc:   { min: 0, max: 15, unit: 'visits' },
    age:   { min: 14, max: 48, unit: 'years' },
    inc:   { min: 3000, max: 120000, unit: 'INR ₹' }
  };

  /**
   * Helper to parse and enforce minimum/maximum limits
   */
  function parseClamped(numEl, sliderEl, min, max) {
    let val = parseFloat(numEl.value);
    if (isNaN(val)) val = parseFloat(sliderEl.value) || min;
    return Math.max(min, Math.min(max, val));
  }

  /**
   * Collect Current Form Values
   */
  function getCurrentPatient() {
    return {
      sysBP: parseClamped(elements.numSysBP, elements.sysBP, LIMITS.sysBP.min, LIMITS.sysBP.max),
      diaBP: parseClamped(elements.numDiaBP, elements.diaBP, LIMITS.diaBP.min, LIMITS.diaBP.max),
      hb: parseClamped(elements.numHb, elements.hb, LIMITS.hb.min, LIMITS.hb.max),
      diab: elements.diab.value,
      comp: elements.comp.value,
      ga: Math.round(parseClamped(elements.numGA, elements.ga, LIMITS.ga.min, LIMITS.ga.max)),
      bw: parseClamped(elements.numBW, elements.bw, LIMITS.bw.min, LIMITS.bw.max),
      deliv: elements.deliv.value,
      anc: Math.round(parseClamped(elements.numANC, elements.anc, LIMITS.anc.min, LIMITS.anc.max)),
      age: Math.round(parseClamped(elements.numAge, elements.age, LIMITS.age.min, LIMITS.age.max)),
      edu: elements.edu.value,
      inc: Math.round(parseClamped(elements.numInc, elements.inc, LIMITS.inc.min, LIMITS.inc.max)),
      res: elements.res.value
    };
  }

  /**
   * Validate and sync numeric input with slider and bounds
   */
  function syncControlPair(numEl, sliderEl, limit, alertEl) {
    let raw = parseFloat(numEl.value);
    if (isNaN(raw)) return;

    if (raw > limit.max) {
      numEl.classList.add('input-error');
      alertEl.className = 'field-alert alert-danger';
      alertEl.textContent = `🚨 Exceeds Maximum Limit of ${limit.max} ${limit.unit}! Value clamped to ${limit.max}.`;
      sliderEl.value = limit.max;
    } else if (raw < limit.min) {
      numEl.classList.add('input-error');
      alertEl.className = 'field-alert alert-warning';
      alertEl.textContent = `⚠️ Below Minimum Limit of ${limit.min} ${limit.unit}! Value clamped to ${limit.min}.`;
      sliderEl.value = limit.min;
    } else {
      numEl.classList.remove('input-error');
      sliderEl.value = raw;
    }
  }

  /**
   * Update Form Badges & Micro-Alerts
   */
  function updateInputDisplays() {
    const p = getCurrentPatient();

    // Micro Alert: Systolic BP
    const rawSys = parseFloat(elements.numSysBP.value);
    if (rawSys > LIMITS.sysBP.max) {
      elements.alertSysBP.className = 'field-alert alert-danger';
      elements.alertSysBP.textContent = `🚨 Exceeds Max Limit (${LIMITS.sysBP.max} mmHg)! Clamped.`;
    } else if (p.sysBP >= 140) {
      elements.alertSysBP.className = 'field-alert alert-danger';
      elements.alertSysBP.textContent = '🚨 Stage 2 Hypertension (≥ 140 mmHg)';
    } else if (p.sysBP >= 120) {
      elements.alertSysBP.className = 'field-alert alert-warning';
      elements.alertSysBP.textContent = '⚠️ Elevated / Prehypertension (120–139)';
    } else {
      elements.alertSysBP.className = 'field-alert alert-safe';
      elements.alertSysBP.textContent = '✓ Normotensive (< 120 mmHg)';
    }

    // Micro Alert: Diastolic BP
    const rawDia = parseFloat(elements.numDiaBP.value);
    if (rawDia > LIMITS.diaBP.max) {
      elements.alertDiaBP.className = 'field-alert alert-danger';
      elements.alertDiaBP.textContent = `🚨 Exceeds Max Limit (${LIMITS.diaBP.max} mmHg)! Clamped.`;
    } else if (p.diaBP >= 90) {
      elements.alertDiaBP.className = 'field-alert alert-danger';
      elements.alertDiaBP.textContent = '🚨 High Diastolic Pressure (≥ 90 mmHg)';
    } else if (p.diaBP >= 80) {
      elements.alertDiaBP.className = 'field-alert alert-warning';
      elements.alertDiaBP.textContent = '⚠️ Prehypertensive Diastolic (80–89)';
    } else {
      elements.alertDiaBP.className = 'field-alert alert-safe';
      elements.alertDiaBP.textContent = '✓ Normotensive (< 80 mmHg)';
    }

    // Micro Alert: Hemoglobin
    const rawHb = parseFloat(elements.numHb.value);
    if (rawHb > LIMITS.hb.max) {
      elements.alertHb.className = 'field-alert alert-danger';
      elements.alertHb.textContent = `🚨 Exceeds Max Limit (${LIMITS.hb.max} g/dL)! Clamped.`;
    } else if (p.hb < 7.0) {
      elements.alertHb.className = 'field-alert alert-danger';
      elements.alertHb.textContent = '🚨 Severe Obstetric Anemia (< 7.0 g/dL)';
    } else if (p.hb < 11.0) {
      elements.alertHb.className = 'field-alert alert-warning';
      elements.alertHb.textContent = '⚠️ Moderate Anemia (7.0–10.9 g/dL)';
    } else {
      elements.alertHb.className = 'field-alert alert-safe';
      elements.alertHb.textContent = '✓ Normal Hemoglobin (≥ 11.0 g/dL)';
    }

    // Micro Alert: Gestational Age
    const rawGA = parseFloat(elements.numGA.value);
    if (rawGA > LIMITS.ga.max) {
      elements.alertGA.className = 'field-alert alert-danger';
      elements.alertGA.textContent = `🚨 Exceeds Max Limit (${LIMITS.ga.max} wks)! Clamped.`;
    } else if (p.ga < 34) {
      elements.alertGA.className = 'field-alert alert-danger';
      elements.alertGA.textContent = '🚨 Extreme Preterm Delivery (< 34 wks)';
    } else if (p.ga < 37) {
      elements.alertGA.className = 'field-alert alert-warning';
      elements.alertGA.textContent = '⚠️ Late Preterm Delivery (34–36 wks)';
    } else {
      elements.alertGA.className = 'field-alert alert-safe';
      elements.alertGA.textContent = '✓ Full Term Gestation (≥ 37 weeks)';
    }

    // Micro Alert: Birth Weight
    const rawBW = parseFloat(elements.numBW.value);
    if (rawBW > LIMITS.bw.max) {
      elements.alertBW.className = 'field-alert alert-danger';
      elements.alertBW.textContent = `🚨 Exceeds Max Limit (${LIMITS.bw.max} kg)! Clamped.`;
    } else if (p.bw < 2.0) {
      elements.alertBW.className = 'field-alert alert-danger';
      elements.alertBW.textContent = '🚨 Very Low Birth Weight (VLBW < 2.0 kg)';
    } else if (p.bw < 2.5) {
      elements.alertBW.className = 'field-alert alert-warning';
      elements.alertBW.textContent = '⚠️ Low Birth Weight (LBW < 2.5 kg)';
    } else {
      elements.alertBW.className = 'field-alert alert-safe';
      elements.alertBW.textContent = '✓ Normal Birth Weight Range (≥ 2.5 kg)';
    }

    // Micro Alert: ANC Visits
    const rawANC = parseFloat(elements.numANC.value);
    if (rawANC > LIMITS.anc.max) {
      elements.alertANC.className = 'field-alert alert-danger';
      elements.alertANC.textContent = `🚨 Exceeds Max Limit (${LIMITS.anc.max} visits)! Clamped.`;
    } else if (p.anc < 2) {
      elements.alertANC.className = 'field-alert alert-danger';
      elements.alertANC.textContent = '🚨 Critically Inadequate ANC (< 2 visits)';
    } else if (p.anc < 4) {
      elements.alertANC.className = 'field-alert alert-warning';
      elements.alertANC.textContent = '⚠️ Below WHO Minimum Standard (< 4 visits)';
    } else {
      elements.alertANC.className = 'field-alert alert-safe';
      elements.alertANC.textContent = '✓ Compliant with WHO Guidelines (≥ 4 visits)';
    }

    // Micro Alert: Age
    const rawAge = parseFloat(elements.numAge.value);
    if (rawAge > LIMITS.age.max) {
      elements.alertAge.className = 'field-alert alert-danger';
      elements.alertAge.textContent = `🚨 Exceeds Max Limit (${LIMITS.age.max} yrs)! Clamped.`;
    } else if (p.age < 18) {
      elements.alertAge.className = 'field-alert alert-warning';
      elements.alertAge.textContent = '⚠️ Adolescent Pregnancy (Elevated Risk)';
    } else if (p.age > 35) {
      elements.alertAge.className = 'field-alert alert-warning';
      elements.alertAge.textContent = '⚠️ Advanced Maternal Age (> 35 years)';
    } else {
      elements.alertAge.className = 'field-alert alert-safe';
      elements.alertAge.textContent = '✓ Optimal Obstetric Age (18–35)';
    }
  }

  /**
   * Run Inference and Update All Output Displays
   */
  function runTriage() {
    updateInputDisplays();
    const patient = getCurrentPatient();
    const result = computeRisk(patient);

    // 1. Primary Hero Result Card
    if (result.isHighRisk) {
      elements.resultOutput.innerHTML = `
        <div class="result-banner-high morphism-card">
          <span class="result-pill result-pill-high">🚨 CRITICAL CLINICAL WARNING</span>
          <h3 class="result-headline">HIGH-RISK PREGNANCY DETECTED</h3>
          <div class="result-subtitle">
            Estimated Adverse Outcome Probability: <b>${result.highProb}%</b>
          </div>
          <p class="result-explanation">
            Patient exhibits acute clinical markers requiring immediate tertiary hospital referral, continuous antenatal surveillance, neonatal intensive care backup, and specialized obstetric consultation.
          </p>
        </div>
      `;
    } else {
      elements.resultOutput.innerHTML = `
        <div class="result-banner-low morphism-card">
          <span class="result-pill result-pill-low">✅ FAVORABLE PROGNOSIS</span>
          <h3 class="result-headline">LOW-RISK PREGNANCY</h3>
          <div class="result-subtitle">
            Safety Confidence: <b>${result.lowProb}%</b> · High-Risk Probability: ${result.highProb}%
          </div>
          <p class="result-explanation">
            Obstetric and hemodynamic parameters reside within standard physiological ranges. Maintain standard antenatal schedules, iron/folic acid supplementation, and institutional delivery planning.
          </p>
        </div>
      `;
    }

    // 2. Animated Circular SVG Risk Gauge
    // Total circumference of arc is approximately 235.6
    const totalArc = 235.6;
    const offset = totalArc - (totalArc * (result.highProb / 100));
    elements.gaugeFill.style.strokeDashoffset = offset;
    elements.gaugeVal.textContent = `${result.highProb}%`;
    elements.gaugeRiskLabel.textContent = result.isHighRisk ? 'HIGH RISK' : 'LOW RISK';
    elements.gaugeRiskLabel.style.fill = result.isHighRisk ? '#ef4444' : '#10b981';

    elements.kpiLow.textContent = `${result.lowProb}%`;
    elements.kpiHigh.textContent = `${result.highProb}%`;

    // 3. Clinical Diagnostic Flag Breakdown
    renderDiagnosticFlags(patient);

    // 4. Biomarker Safety Spectrum Bars
    renderBiomarkerBars(patient);

    // 5. Spider/Radar Chart
    renderSpiderRadar();
  }

  /**
   * Render Diagnostic Flags
   */
  function renderDiagnosticFlags(p) {
    const dangerFlags = [];
    const safeFlags = [];

    if (p.hb < 7.0) dangerFlags.push(`Severe Anemia: Hemoglobin ${p.hb.toFixed(1)} g/dL (< 7.0 critical limit)`);
    else if (p.hb < 11.0) dangerFlags.push(`Moderate Anemia: Hemoglobin ${p.hb.toFixed(1)} g/dL (< 11.0 standard)`);
    else safeFlags.push(`Hemoglobin is optimal (${p.hb.toFixed(1)} g/dL)`);

    if (p.sysBP >= 140 || p.diaBP >= 90) dangerFlags.push(`Gestational Hypertension: BP ${p.sysBP}/${p.diaBP} mmHg (≥ 140/90)`);
    else safeFlags.push(`Normotensive Blood Pressure (${p.sysBP}/${p.diaBP} mmHg)`);

    if (p.ga < 37) dangerFlags.push(`Preterm Gestation: ${p.ga} weeks (< 37 term threshold)`);
    else safeFlags.push(`Term Gestational Age (${p.ga} weeks)`);

    if (p.bw < 2.5) dangerFlags.push(`Low Infant Birth Weight: ${p.bw.toFixed(2)} kg (< 2.5 kg)`);
    else safeFlags.push(`Normal Infant Weight (${p.bw.toFixed(2)} kg)`);

    if (p.comp !== 'None') dangerFlags.push(`${p.comp} Pregnancy Complications recorded`);
    else safeFlags.push('No pre-existing obstetric complications');

    if (p.anc < 4) dangerFlags.push(`Inadequate Antenatal Care: ${p.anc} visits (< WHO standard of 4)`);
    else safeFlags.push(`Antenatal Care compliant (${p.anc} visits completed)`);

    if (p.deliv === 'Home') dangerFlags.push('Home delivery selected (Substantial risk of unassisted emergency)');

    let html = '';
    dangerFlags.forEach(f => {
      html += `<div class="flag-item flag-danger">⚠️ ${f}</div>`;
    });
    safeFlags.slice(0, 4).forEach(f => {
      html += `<div class="flag-item flag-safe">✓ ${f}</div>`;
    });

    elements.flagList.innerHTML = html;
  }

  /**
   * Render Biomarker Safety Spectrum Bars
   */
  function renderBiomarkerBars(p) {
    const biomarkers = [
      {
        name: 'Systolic BP (mmHg)',
        val: p.sysBP,
        min: 80, max: 200,
        safe: '< 120',
        danger: '≥ 140',
        color: p.sysBP >= 140 ? '#ef4444' : (p.sysBP >= 120 ? '#f59e0b' : '#10b981'),
        pct: Math.min(100, Math.max(0, ((p.sysBP - 80) / 120) * 100))
      },
      {
        name: 'Diastolic BP (mmHg)',
        val: p.diaBP,
        min: 50, max: 130,
        safe: '< 80',
        danger: '≥ 90',
        color: p.diaBP >= 90 ? '#ef4444' : (p.diaBP >= 80 ? '#f59e0b' : '#10b981'),
        pct: Math.min(100, Math.max(0, ((p.diaBP - 50) / 80) * 100))
      },
      {
        name: 'Hemoglobin (g/dL)',
        val: p.hb,
        min: 5.0, max: 17.0,
        safe: '≥ 11.0',
        danger: '< 7.0',
        color: p.hb < 7.0 ? '#ef4444' : (p.hb < 11.0 ? '#f59e0b' : '#10b981'),
        pct: Math.min(100, Math.max(0, ((p.hb - 5.0) / 12.0) * 100))
      },
      {
        name: 'Gestational Age (weeks)',
        val: p.ga,
        min: 24, max: 43,
        safe: '≥ 37',
        danger: '< 34',
        color: p.ga < 34 ? '#ef4444' : (p.ga < 37 ? '#f59e0b' : '#10b981'),
        pct: Math.min(100, Math.max(0, ((p.ga - 24) / 19) * 100))
      },
      {
        name: 'Birth Weight (kg)',
        val: p.bw,
        min: 0.8, max: 5.0,
        safe: '≥ 2.5',
        danger: '< 2.0',
        color: p.bw < 2.0 ? '#ef4444' : (p.bw < 2.5 ? '#f59e0b' : '#10b981'),
        pct: Math.min(100, Math.max(0, ((p.bw - 0.8) / 4.2) * 100))
      },
      {
        name: 'ANC Visits (count)',
        val: p.anc,
        min: 0, max: 15,
        safe: '≥ 4',
        danger: '< 2',
        color: p.anc < 2 ? '#ef4444' : (p.anc < 4 ? '#f59e0b' : '#10b981'),
        pct: Math.min(100, Math.max(0, (p.anc / 10) * 100))
      }
    ];

    let html = '';
    biomarkers.forEach(b => {
      html += `
        <div class="bar-row">
          <div class="bar-header">
            <span>${b.name}</span>
            <span style="color: ${b.color}; font-weight: 800;">${b.val}</span>
          </div>
          <div class="bar-track">
            <div class="bar-progress" style="width: ${b.pct}%; background: ${b.color};"></div>
          </div>
        </div>
      `;
    });

    elements.spectrumList.innerHTML = html;
  }

  /**
   * Render HTML5 Canvas Spider / Radar Chart
   */
  function renderSpiderRadar() {
    const canvas = elements.radarCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(centerX, centerY) - 38;

    const categories = ['Systolic BP', 'Diastolic BP', 'Hemoglobin', 'Gest. Age', 'Birth Wt', 'ANC Visits'];
    const totalAxes = categories.length;

    const p = getCurrentPatient();

    // Normalized scores (1.0 = safe, 0.1 = high risk)
    const scoreSys = Math.max(0.1, Math.min(1.0, 1.0 - Math.max(0, p.sysBP - 120) / 75));
    const scoreDia = Math.max(0.1, Math.min(1.0, 1.0 - Math.max(0, p.diaBP - 80) / 45));
    const scoreHb = Math.max(0.1, Math.min(1.0, p.hb / 12.0));
    const scoreGA = Math.max(0.1, Math.min(1.0, p.ga / 38.0));
    const scoreBW = Math.max(0.1, Math.min(1.0, p.bw / 3.0));
    const scoreANC = Math.max(0.1, Math.min(1.0, p.anc / 5.0));

    const patientScores = [scoreSys, scoreDia, scoreHb, scoreGA, scoreBW, scoreANC];
    const baselineScores = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    // Draw background concentric web
    const levels = 4;
    for (let l = 1; l <= levels; l++) {
      ctx.beginPath();
      const levelRadius = (radius / levels) * l;
      for (let i = 0; i < totalAxes; i++) {
        const angle = (Math.PI * 2 / totalAxes) * i - Math.PI / 2;
        const x = centerX + Math.cos(angle) * levelRadius;
        const y = centerY + Math.sin(angle) * levelRadius;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Draw axis lines and labels
    for (let i = 0; i < totalAxes; i++) {
      const angle = (Math.PI * 2 / totalAxes) * i - Math.PI / 2;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;

      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(x, y);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
      ctx.stroke();

      // Label text
      const labelX = centerX + Math.cos(angle) * (radius + 20);
      const labelY = centerY + Math.sin(angle) * (radius + 18);
      ctx.font = '600 10px Plus Jakarta Sans';
      ctx.fillStyle = '#94a3b8';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(categories[i], labelX, labelY);
    }

    // 1. Draw Safe Baseline Polygon (Green dashed)
    ctx.beginPath();
    for (let i = 0; i < totalAxes; i++) {
      const angle = (Math.PI * 2 / totalAxes) * i - Math.PI / 2;
      const r = radius * baselineScores[i];
      const x = centerX + Math.cos(angle) * r;
      const y = centerY + Math.sin(angle) * r;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.fillStyle = 'rgba(16, 185, 129, 0.06)';
    ctx.fill();
    ctx.setLineDash([]);

    // 2. Draw Patient Polygon (Indigo / Purple gradient fill)
    ctx.beginPath();
    for (let i = 0; i < totalAxes; i++) {
      const angle = (Math.PI * 2 / totalAxes) * i - Math.PI / 2;
      const r = radius * patientScores[i];
      const x = centerX + Math.cos(angle) * r;
      const y = centerY + Math.sin(angle) * r;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = '#818cf8';
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.fillStyle = 'rgba(99, 102, 241, 0.35)';
    ctx.fill();

    // Draw vertex dots on patient polygon
    for (let i = 0; i < totalAxes; i++) {
      const angle = (Math.PI * 2 / totalAxes) * i - Math.PI / 2;
      const r = radius * patientScores[i];
      const x = centerX + Math.cos(angle) * r;
      const y = centerY + Math.sin(angle) * r;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.strokeStyle = '#4f46e5';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }


  // =========================================================================
  // 4. Presets Loader
  // =========================================================================
  const presets = {
    low: {
      sysBP: 114, diaBP: 74, hb: 12.8, diab: 'No', comp: 'None',
      ga: 39, bw: 3.25, deliv: 'Hospital', anc: 6,
      age: 24, edu: 'Higher', inc: 32000, res: 'Urban'
    },
    borderline: {
      sysBP: 136, diaBP: 88, hb: 10.1, diab: 'No', comp: 'Mild',
      ga: 36, bw: 2.45, deliv: 'Clinic', anc: 3,
      age: 34, edu: 'Secondary', inc: 14000, res: 'Rural'
    },
    critical: {
      sysBP: 162, diaBP: 102, hb: 6.6, diab: 'Yes', comp: 'Severe',
      ga: 31, bw: 1.70, deliv: 'Home', anc: 1,
      age: 17, edu: 'No Education', inc: 5000, res: 'Rural'
    }
  };

  function applyPreset(presetKey) {
    const data = presets[presetKey];
    if (!data) return;

    // Sliders
    elements.sysBP.value = data.sysBP;
    elements.diaBP.value = data.diaBP;
    elements.hb.value = data.hb;
    elements.ga.value = data.ga;
    elements.bw.value = data.bw;
    elements.anc.value = data.anc;
    elements.age.value = data.age;
    elements.inc.value = data.inc;

    // Direct Number Inputs
    elements.numSysBP.value = data.sysBP;
    elements.numDiaBP.value = data.diaBP;
    elements.numHb.value = data.hb;
    elements.numGA.value = data.ga;
    elements.numBW.value = data.bw;
    elements.numANC.value = data.anc;
    elements.numAge.value = data.age;
    elements.numInc.value = data.inc;

    // Dropdowns
    elements.diab.value = data.diab;
    elements.comp.value = data.comp;
    elements.deliv.value = data.deliv;
    elements.edu.value = data.edu;
    elements.res.value = data.res;

    // Clear any previous error states
    [
      elements.numSysBP, elements.numDiaBP, elements.numHb, elements.numGA,
      elements.numBW, elements.numANC, elements.numAge, elements.numInc
    ].forEach(el => el.classList.remove('input-error'));

    runTriage();
  }

  document.getElementById('presetLowBtn').addEventListener('click', () => applyPreset('low'));
  document.getElementById('presetBorderlineBtn').addEventListener('click', () => applyPreset('borderline'));
  document.getElementById('presetCriticalBtn').addEventListener('click', () => applyPreset('critical'));


  // =========================================================================
  // 5. Dual-Control Input Synchronization & Limit Enforcement
  // =========================================================================
  const numericControlPairs = [
    { num: elements.numSysBP, slider: elements.sysBP, limit: LIMITS.sysBP, alert: elements.alertSysBP },
    { num: elements.numDiaBP, slider: elements.diaBP, limit: LIMITS.diaBP, alert: elements.alertDiaBP },
    { num: elements.numHb, slider: elements.hb, limit: LIMITS.hb, alert: elements.alertHb },
    { num: elements.numGA, slider: elements.ga, limit: LIMITS.ga, alert: elements.alertGA },
    { num: elements.numBW, slider: elements.bw, limit: LIMITS.bw, alert: elements.alertBW },
    { num: elements.numANC, slider: elements.anc, limit: LIMITS.anc, alert: elements.alertANC },
    { num: elements.numAge, slider: elements.age, limit: LIMITS.age, alert: elements.alertAge },
    { num: elements.numInc, slider: elements.inc, limit: LIMITS.inc, alert: null }
  ];

  numericControlPairs.forEach(pair => {
    // When user types directly into number input
    pair.num.addEventListener('input', () => {
      let val = parseFloat(pair.num.value);
      if (!isNaN(val)) {
        if (val > pair.limit.max) {
          pair.num.classList.add('input-error');
          if (pair.alert) {
            pair.alert.className = 'field-alert alert-danger';
            pair.alert.textContent = `🚨 Exceeds Max Limit (${pair.limit.max} ${pair.limit.unit})! Clamped.`;
          }
          pair.slider.value = pair.limit.max;
        } else if (val < pair.limit.min) {
          pair.num.classList.add('input-error');
          if (pair.alert) {
            pair.alert.className = 'field-alert alert-warning';
            pair.alert.textContent = `⚠️ Below Min Limit (${pair.limit.min} ${pair.limit.unit})! Clamped.`;
          }
          pair.slider.value = pair.limit.min;
        } else {
          pair.num.classList.remove('input-error');
          pair.slider.value = val;
        }
      }
      if (elements.liveMode.checked) {
        runTriage();
      }
    });

    // Auto-clamp when user clicks away / blurs the number field
    pair.num.addEventListener('blur', () => {
      let val = parseFloat(pair.num.value);
      if (isNaN(val) || val < pair.limit.min) {
        pair.num.value = pair.limit.min;
        pair.slider.value = pair.limit.min;
        pair.num.classList.remove('input-error');
      } else if (val > pair.limit.max) {
        pair.num.value = pair.limit.max;
        pair.slider.value = pair.limit.max;
        pair.num.classList.remove('input-error');
      }
      runTriage();
    });

    // When user drags the range slider
    pair.slider.addEventListener('input', () => {
      pair.num.value = pair.slider.value;
      pair.num.classList.remove('input-error');
      if (elements.liveMode.checked) {
        runTriage();
      }
    });
  });

  // Dropdown Select Inputs
  const selectInputs = [elements.diab, elements.comp, elements.deliv, elements.edu, elements.res];
  selectInputs.forEach(sel => {
    sel.addEventListener('change', () => {
      if (elements.liveMode.checked) {
        runTriage();
      }
    });
  });

  elements.recalcBtn.addEventListener('click', runTriage);


  // =========================================================================
  // 6. EDA Gallery Filtering & Image Modal Zoom
  // =========================================================================
  const filterButtons = document.querySelectorAll('.filter-btn[data-filter]');
  const galleryCards = document.querySelectorAll('.gallery-card');

  filterButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      filterButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const filter = btn.getAttribute('data-filter');
      galleryCards.forEach(card => {
        if (filter === 'all' || card.getAttribute('data-category') === filter) {
          card.style.display = 'flex';
        } else {
          card.style.display = 'none';
        }
      });
    });
  });

  // Modal zoom
  const modal = document.getElementById('imageModal');
  const modalImg = document.getElementById('modalImg');
  const modalTitle = document.getElementById('modalTitle');
  const modalDesc = document.getElementById('modalDesc');
  const modalClose = document.getElementById('modalCloseBtn');

  galleryCards.forEach(card => {
    card.addEventListener('click', () => {
      const imgSrc = card.getAttribute('data-img');
      const title = card.querySelector('h4').textContent;
      const desc = card.querySelector('p').textContent;

      modalImg.src = imgSrc;
      modalTitle.textContent = title;
      modalDesc.textContent = desc;
      modal.classList.add('active');
    });
  });

  modalClose.addEventListener('click', () => modal.classList.remove('active'));
  modal.addEventListener('click', (e) => {
    if (e.target === modal) modal.classList.remove('active');
  });


  // =========================================================================
  // 7. Model Benchmarks Deep-Dive Selector
  // =========================================================================
  const diagButtons = document.querySelectorAll('.filter-btn[data-diag]');
  const deepdiveImg = document.getElementById('deepdiveImg');
  const deepdiveCaption = document.getElementById('deepdiveCaption');

  const diagData = {
    cm: {
      img: 'figures/confusion_matrices.png',
      caption: '💡 <b>Clinical Reading:</b> Lower values in the bottom-left quadrant (False Negatives) represent safer clinical models.'
    },
    roc: {
      img: 'figures/roc_curves.png',
      caption: '💡 <b>Clinical Reading:</b> Curves hugging the top-left boundary denote superior True Positive Rates at minimal False Alarm rates.'
    },
    shap: {
      img: 'figures/shap_summary.png',
      caption: '💡 <b>Clinical Reading:</b> Red points on the right indicate that high values of that feature push the prediction toward HIGH-RISK.'
    },
    fi: {
      img: 'figures/feature_importance_rf.png',
      caption: '💡 <b>Clinical Reading:</b> Hemoglobin level, systolic BP, and infant birth weight are the top 3 predictive drivers.'
    }
  };

  diagButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      diagButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const key = btn.getAttribute('data-diag');
      if (diagData[key]) {
        deepdiveImg.src = diagData[key].img;
        deepdiveCaption.innerHTML = diagData[key].caption;
      }
    });
  });


  // =========================================================================
  // 8. Batch CSV Screening Tool
  // =========================================================================
  const dropzone = document.getElementById('csvDropzone');
  const fileInput = document.getElementById('csvFileInput');
  const sampleBtn = document.getElementById('loadSampleCsvBtn');
  const templateBtn = document.getElementById('downloadTemplateBtn');
  const resultsWrapper = document.getElementById('batchResultsWrapper');
  const tableBody = document.getElementById('batchTableBody');
  const exportBtn = document.getElementById('exportScoredCsvBtn');

  let currentBatchData = [];

  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('drag-over');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
      handleCsvFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
      handleCsvFile(e.target.files[0]);
    }
  });

  const sampleCohort = [
    { mother_age: 24, education_level: 'Higher', family_income: 45000, residence_urban_rural: 'Urban', blood_pressure_systolic: 114, blood_pressure_diastolic: 74, hemoglobin_level: 13.0, diabetes_status: 'No', pregnancy_complications: 'None', gestational_age: 40, birth_weight: 3.4, antenatal_care_visits: 7, place_of_delivery: 'Hospital' },
    { mother_age: 17, education_level: 'No Education', family_income: 6000, residence_urban_rural: 'Rural', blood_pressure_systolic: 162, blood_pressure_diastolic: 102, hemoglobin_level: 6.5, diabetes_status: 'Yes', pregnancy_complications: 'Severe', gestational_age: 30, birth_weight: 1.65, antenatal_care_visits: 1, place_of_delivery: 'Home' },
    { mother_age: 29, education_level: 'Secondary', family_income: 22000, residence_urban_rural: 'Rural', blood_pressure_systolic: 120, blood_pressure_diastolic: 80, hemoglobin_level: 11.2, diabetes_status: 'No', pregnancy_complications: 'None', gestational_age: 39, birth_weight: 3.1, antenatal_care_visits: 5, place_of_delivery: 'Hospital' },
    { mother_age: 38, education_level: 'Primary', family_income: 11000, residence_urban_rural: 'Rural', blood_pressure_systolic: 148, blood_pressure_diastolic: 94, hemoglobin_level: 9.8, diabetes_status: 'Yes', pregnancy_complications: 'Mild', gestational_age: 35, birth_weight: 2.2, antenatal_care_visits: 2, place_of_delivery: 'Clinic' },
    { mother_age: 22, education_level: 'Higher', family_income: 38000, residence_urban_rural: 'Urban', blood_pressure_systolic: 116, blood_pressure_diastolic: 76, hemoglobin_level: 12.4, diabetes_status: 'No', pregnancy_complications: 'None', gestational_age: 39, birth_weight: 3.3, antenatal_care_visits: 6, place_of_delivery: 'Hospital' },
    { mother_age: 16, education_level: 'Primary', family_income: 8000, residence_urban_rural: 'Rural', blood_pressure_systolic: 154, blood_pressure_diastolic: 98, hemoglobin_level: 7.2, diabetes_status: 'No', pregnancy_complications: 'Severe', gestational_age: 32, birth_weight: 1.85, antenatal_care_visits: 1, place_of_delivery: 'Home' },
    { mother_age: 26, education_level: 'Secondary', family_income: 18000, residence_urban_rural: 'Urban', blood_pressure_systolic: 122, blood_pressure_diastolic: 78, hemoglobin_level: 11.5, diabetes_status: 'No', pregnancy_complications: 'None', gestational_age: 38, birth_weight: 2.9, antenatal_care_visits: 4, place_of_delivery: 'Hospital' },
    { mother_age: 35, education_level: 'Secondary', family_income: 15000, residence_urban_rural: 'Rural', blood_pressure_systolic: 138, blood_pressure_diastolic: 88, hemoglobin_level: 10.4, diabetes_status: 'No', pregnancy_complications: 'Mild', gestational_age: 36, birth_weight: 2.4, antenatal_care_visits: 3, place_of_delivery: 'Clinic' },
    { mother_age: 20, education_level: 'Higher', family_income: 42000, residence_urban_rural: 'Urban', blood_pressure_systolic: 112, blood_pressure_diastolic: 72, hemoglobin_level: 13.2, diabetes_status: 'No', pregnancy_complications: 'None', gestational_age: 40, birth_weight: 3.5, antenatal_care_visits: 8, place_of_delivery: 'Hospital' },
    { mother_age: 41, education_level: 'No Education', family_income: 7000, residence_urban_rural: 'Rural', blood_pressure_systolic: 168, blood_pressure_diastolic: 106, hemoglobin_level: 6.8, diabetes_status: 'Yes', pregnancy_complications: 'Severe', gestational_age: 29, birth_weight: 1.5, antenatal_care_visits: 1, place_of_delivery: 'Home' }
  ];

  sampleBtn.addEventListener('click', () => {
    processBatchRecords(sampleCohort);
  });

  templateBtn.addEventListener('click', () => {
    const headers = [
      'mother_age', 'education_level', 'family_income', 'residence_urban_rural',
      'blood_pressure_systolic', 'blood_pressure_diastolic', 'hemoglobin_level',
      'diabetes_status', 'pregnancy_complications', 'gestational_age',
      'birth_weight', 'antenatal_care_visits', 'place_of_delivery'
    ];
    const csvContent = 'data:text/csv;charset=utf-8,' + headers.join(',') + '\n';
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'maternal_risk_template.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });

  function handleCsvFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const parsed = parseCsv(text);
      if (parsed.length > 0) {
        processBatchRecords(parsed);
      } else {
        alert('Invalid or empty CSV file. Please check column format.');
      }
    };
    reader.readAsText(file);
  }

  function parseCsv(text) {
    const lines = text.trim().split(/\r\n|\n/);
    if (lines.length < 2) return [];
    const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));

    const records = [];
    for (let i = 1; i < lines.length; i++) {
      const vals = lines[i].split(',').map(v => v.trim().replace(/^"|"$/g, ''));
      if (vals.length === headers.length) {
        const row = {};
        headers.forEach((h, idx) => {
          row[h] = vals[idx];
        });
        records.push(row);
      }
    }
    return records;
  }

  function processBatchRecords(records) {
    currentBatchData = [];
    let highCount = 0;
    let sumRisk = 0;

    tableBody.innerHTML = '';

    records.forEach((r, idx) => {
      const patient = {
        sysBP: Number(r.blood_pressure_systolic || 120),
        diaBP: Number(r.blood_pressure_diastolic || 80),
        hb: Number(r.hemoglobin_level || 11.0),
        diab: r.diabetes_status || 'No',
        comp: r.pregnancy_complications || 'None',
        ga: Number(r.gestational_age || 38),
        bw: Number(r.birth_weight || 2.8),
        deliv: r.place_of_delivery || 'Hospital',
        anc: Number(r.antenatal_care_visits || 4),
        age: Number(r.mother_age || 25),
        edu: r.education_level || 'Secondary',
        inc: Number(r.family_income || 15000),
        res: r.residence_urban_rural || 'Rural'
      };

      const score = computeRisk(patient);
      if (score.isHighRisk) highCount++;
      sumRisk += score.highProb;

      currentBatchData.push({
        ...r,
        predicted_risk: score.isHighRisk ? 'HIGH-RISK' : 'LOW-RISK',
        risk_probability_percent: score.highProb
      });

      const riskPill = score.isHighRisk
        ? `<span class="result-pill result-pill-high" style="font-size: 0.72rem; padding: 3px 8px;">HIGH-RISK</span>`
        : `<span class="result-pill result-pill-low" style="font-size: 0.72rem; padding: 3px 8px;">LOW-RISK</span>`;

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${idx + 1}</td>
        <td>${riskPill}</td>
        <td style="font-weight: 800; color: ${score.isHighRisk ? '#f87171' : '#34d399'};">${score.highProb}%</td>
        <td>${patient.age}</td>
        <td>${patient.sysBP}/${patient.diaBP}</td>
        <td>${patient.hb.toFixed(1)}</td>
        <td>${patient.ga}w</td>
        <td>${patient.bw.toFixed(2)} kg</td>
        <td>${patient.anc}</td>
        <td>${patient.comp}</td>
      `;
      tableBody.appendChild(tr);
    });

    const total = records.length;
    const lowCount = total - highCount;
    const avgRisk = (sumRisk / total).toFixed(1);
    const lowPct = ((lowCount / total) * 100).toFixed(1);

    document.getElementById('batchTotalCount').textContent = total;
    document.getElementById('batchHighCount').textContent = highCount;
    document.getElementById('batchLowRate').textContent = `${lowPct}%`;
    document.getElementById('batchAvgRisk').textContent = `${avgRisk}%`;

    resultsWrapper.style.display = 'block';
  }

  exportBtn.addEventListener('click', () => {
    if (!currentBatchData.length) return;
    const headers = Object.keys(currentBatchData[0]);
    let csv = headers.join(',') + '\n';
    currentBatchData.forEach(row => {
      csv += headers.map(h => `"${row[h]}"`).join(',') + '\n';
    });
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', 'scored_patient_cohort.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });


  // =========================================================================
  // 9. Initial Load & Window Resize Listener
  // =========================================================================
  runTriage();

  window.addEventListener('resize', () => {
    renderSpiderRadar();
  });

});
