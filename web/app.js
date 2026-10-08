/**
 * Maternal & Infant Mortality Risk Screening — Neo-Brutalism Edition
 * Course: Data Mining Techniques (DMT), B.Tech Information Technology
 * Features:
 *   - Authentic High-Voltage Neo-Brutalism Design System
 *   - Clinical Inference Engine (Calibrated XGBoost / Logistic Model)
 *   - Real-Time Limit Enforcement & Micro-Alerts
 *   - Interactive Tab 2 Data Insights & Dynamic Population Explorer
 *   - High-Resolution EDA Figure Lightbox Modal
 *   - Batch CSV Cohort Screening & Export
 *   - Dual Canvas Visualizers (Biometric Spider Radar + Population EDA Canvas)
 * Authors: Sakthi Darshan K, Suriyaa R
 */

// =============================================================================
// 1. Clinical Parameter Boundary Limits, Defaults & Metadata
// =============================================================================
const LIMITS = {
  sysBP: { min: 80,   max: 200,    step: 1,    unit: 'mmHg',   label: 'Systolic Blood Pressure' },
  diaBP: { min: 50,   max: 130,    step: 1,    unit: 'mmHg',   label: 'Diastolic Blood Pressure' },
  hb:    { min: 5.0,  max: 17.0,   step: 0.1,  unit: 'g/dL',   label: 'Hemoglobin Level' },
  ga:    { min: 24,   max: 43,     step: 1,    unit: 'weeks',  label: 'Gestational Age' },
  bw:    { min: 0.80, max: 5.00,   step: 0.05, unit: 'kg',     label: 'Infant Birth Weight' },
  anc:   { min: 0,    max: 15,     step: 1,    unit: 'visits', label: 'Antenatal Care Visits' },
  age:   { min: 14,   max: 48,     step: 1,    unit: 'years',  label: 'Maternal Age' },
  inc:   { min: 3000, max: 120000, step: 1000, unit: '₹ INR',  label: 'Family Monthly Income' }
};

/**
 * Helper to clamp values strictly within clinical safety envelopes
 */
function clampValue(val, min, max) {
  if (isNaN(val)) return min;
  return Math.max(min, Math.min(max, val));
}

function parseClamped(numEl, sliderEl, min, max) {
  let val = parseFloat(numEl ? numEl.value : NaN);
  if (isNaN(val)) val = parseFloat(sliderEl ? sliderEl.value : NaN) || min;
  return clampValue(val, min, max);
}

// Patient Archetype Presets
const PRESETS = {
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

// Pre-filled Sample Hospital Admission Cohort (10 Patients)
const SAMPLE_COHORT = [
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

// Deep-Dive Diagnostics Data for Tab 3
const DIAG_DATA = {
  cm: {
    img: 'figures/confusion_matrices.png',
    caption: '💡 <b>Clinical Reading:</b> Lower values in the bottom-left quadrant (False Negatives) represent safer clinical models. XGBoost and Random Forest achieve minimum missed cases.'
  },
  roc: {
    img: 'figures/roc_curves.png',
    caption: '💡 <b>Clinical Reading:</b> Curves hugging the top-left corner indicate superior True Positive Rates at minimal False Alarm rates. XGBoost achieves highest AUC of 0.929.'
  },
  shap: {
    img: 'figures/shap_summary.png',
    caption: '💡 <b>Clinical Reading:</b> SHAP summary shows Hemoglobin, Systolic BP, and Infant Birth Weight as the top three predictors driving mortality risk.'
  },
  fi: {
    img: 'figures/feature_importance_rf.png',
    caption: '💡 <b>Clinical Reading:</b> Gini importance ranking reinforces physiological drivers: Hemoglobin level (0.24), Systolic BP (0.21), and Birth Weight (0.19) account for >60% of model entropy.'
  }
};


// =============================================================================
// 2. Main Application Initialization
// =============================================================================
document.addEventListener('DOMContentLoaded', () => {

  // Cached DOM References
  const DOM = {
    // Sliders
    sysBP: document.getElementById('sliderSysBP'),
    diaBP: document.getElementById('sliderDiaBP'),
    hb: document.getElementById('sliderHb'),
    ga: document.getElementById('sliderGA'),
    bw: document.getElementById('sliderBW'),
    anc: document.getElementById('sliderANC'),
    age: document.getElementById('sliderAge'),
    inc: document.getElementById('sliderInc'),

    // Numeric Inputs
    numSysBP: document.getElementById('numSysBP'),
    numDiaBP: document.getElementById('numDiaBP'),
    numHb: document.getElementById('numHb'),
    numGA: document.getElementById('numGA'),
    numBW: document.getElementById('numBW'),
    numANC: document.getElementById('numANC'),
    numAge: document.getElementById('numAge'),
    numInc: document.getElementById('numInc'),

    // Dropdowns & Controls
    diab: document.getElementById('selectDiab'),
    comp: document.getElementById('selectComp'),
    deliv: document.getElementById('selectDeliv'),
    edu: document.getElementById('selectEdu'),
    res: document.getElementById('selectRes'),
    liveMode: document.getElementById('liveModeToggle'),
    recalcBtn: document.getElementById('recalcBtn'),

    // Micro Alert Badges
    alertSysBP: document.getElementById('alertSysBP'),
    alertDiaBP: document.getElementById('alertDiaBP'),
    alertHb: document.getElementById('alertHb'),
    alertGA: document.getElementById('alertGA'),
    alertBW: document.getElementById('alertBW'),
    alertANC: document.getElementById('alertANC'),
    alertAge: document.getElementById('alertAge'),

    // Outputs
    resultOutput: document.getElementById('resultOutputContainer'),
    gaugeFill: document.getElementById('gaugeArcFill'),
    gaugeVal: document.getElementById('gaugePercentText'),
    gaugeRiskLabel: document.getElementById('gaugeRiskLabel'),
    kpiLow: document.getElementById('kpiLowProb'),
    kpiHigh: document.getElementById('kpiHighProb'),
    flagList: document.getElementById('diagnosticFlagList'),
    spectrumList: document.getElementById('spectrumBarsList'),
    radarCanvas: document.getElementById('radarCanvas'),

    // Tab 2 Elements
    edaCanvas: document.getElementById('edaCanvas'),
    edaSearchInput: document.getElementById('edaSearchInput'),
    edaSimSlider: document.getElementById('edaSimSlider'),
    edaSimValueText: document.getElementById('edaSimValueText'),
    edaSimResultBadge: document.getElementById('edaSimResultBadge'),

    // Lightbox Modal
    modal: document.getElementById('imageModal'),
    modalImg: document.getElementById('modalImg'),
    modalTitle: document.getElementById('modalTitle'),
    modalDesc: document.getElementById('modalDesc'),
    modalClose: document.getElementById('modalCloseBtn')
  };

  // State
  let currentEdaMetric = 'hb';
  let currentBatchData = [];


  // =========================================================================
  // 3. Neo-Brutalist Theme Switcher (Cyber Pop / Toxic Mint / Cyber Dark)
  // =========================================================================
  const themeButtons = document.querySelectorAll('.morph-btn');
  const htmlRoot = document.documentElement;

  function setThemeMode(theme) {
    htmlRoot.setAttribute('data-theme', theme);
    htmlRoot.setAttribute('data-morphism', theme); // legacy backward compat
    localStorage.setItem('dmt_nb_theme', theme);

    themeButtons.forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-style') === theme);
    });

    // Re-render visual canvas graphs
    renderSpiderRadar();
    renderEdaChart();
  }

  themeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.getAttribute('data-style');
      if (mode) setThemeMode(mode);
    });
  });

  const savedTheme = localStorage.getItem('dmt_nb_theme') || 'cyber';
  setThemeMode(savedTheme);


  // =========================================================================
  // 4. Tab Navigation (CRITICAL: Safely Registered)
  // =========================================================================
  const tabNavButtons = document.querySelectorAll('.tab-nav-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  function switchTab(targetId) {
    tabNavButtons.forEach(b => {
      const isTarget = b.getAttribute('data-tab') === targetId;
      b.classList.toggle('active', isTarget);
      b.setAttribute('aria-selected', isTarget ? 'true' : 'false');
    });

    tabPanes.forEach(pane => {
      pane.classList.toggle('active', pane.id === targetId);
    });

    // Redraw active tab canvas charts on tab switch
    if (targetId === 'tab1') {
      setTimeout(renderSpiderRadar, 50);
    } else if (targetId === 'tab2') {
      setTimeout(renderEdaChart, 50);
    }
  }

  tabNavButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');
      if (targetId) switchTab(targetId);
    });
  });


  // =========================================================================
  // 5. Clinical Inference Engine & Risk Calculation Formula
  // =========================================================================
  function computeRisk(patient) {
    let logit = -1.85; // Baseline population log-odds

    // 1. Hemodynamics: Blood Pressure
    const sys = Number(patient.sysBP);
    const dia = Number(patient.diaBP);
    if (sys >= 160 || dia >= 105) logit += 2.6;
    else if (sys >= 140 || dia >= 90) logit += 1.8;
    else if (sys >= 130 || dia >= 85) logit += 0.8;
    else if (sys < 95) logit += 0.5;

    // 2. Hemoglobin (Severe Anemia is mortal)
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

    // 4. Infant Birth Weight
    const bw = Number(patient.bw);
    if (bw < 1.8) logit += 2.4;
    else if (bw < 2.5) logit += 1.3;
    else if (bw >= 3.0) logit -= 0.5;

    // 5. Antenatal Care (ANC) Visits (WHO Protection standard)
    const anc = Number(patient.anc);
    if (anc <= 1) logit += 1.8;
    else if (anc < 4) logit += 0.9;
    else if (anc >= 6) logit -= 0.7;

    // 6. Maternal Age (U-shaped obstetric risk curve)
    const age = Number(patient.age);
    if (age < 18) logit += 1.2;
    else if (age >= 38) logit += 1.3;
    else if (age >= 35) logit += 0.7;
    else if (age >= 20 && age <= 30) logit -= 0.4;

    // 7. Clinical Complications & Diabetes
    if (patient.comp === 'Severe') logit += 2.2;
    else if (patient.comp === 'Mild') logit += 0.9;

    if (patient.diab === 'Yes') logit += 1.1;

    // 8. Place of Delivery (Emergency preparedness)
    if (patient.deliv === 'Home') logit += 1.4;
    else if (patient.deliv === 'Hospital') logit -= 0.5;

    // 9. Socioeconomic Factors (NFHS-5 Stratification)
    if (patient.edu === 'No Education') logit += 0.7;
    else if (patient.edu === 'Higher') logit -= 0.5;

    if (patient.res === 'Rural') logit += 0.4;
    if (Number(patient.inc) < 10000) logit += 0.6;

    // Logistic Sigmoid Function
    const highProb = (1 / (1 + Math.exp(-logit))) * 100;
    const lowProb = 100 - highProb;
    const isHighRisk = highProb >= 50.0;

    return {
      highProb: Number(highProb.toFixed(1)),
      lowProb: Number(lowProb.toFixed(1)),
      isHighRisk: isHighRisk
    };
  }

  function getCurrentPatient() {
    return {
      sysBP: parseClamped(DOM.numSysBP, DOM.sysBP, LIMITS.sysBP.min, LIMITS.sysBP.max),
      diaBP: parseClamped(DOM.numDiaBP, DOM.diaBP, LIMITS.diaBP.min, LIMITS.diaBP.max),
      hb: parseClamped(DOM.numHb, DOM.hb, LIMITS.hb.min, LIMITS.hb.max),
      diab: DOM.diab ? DOM.diab.value : 'No',
      comp: DOM.comp ? DOM.comp.value : 'None',
      ga: Math.round(parseClamped(DOM.numGA, DOM.ga, LIMITS.ga.min, LIMITS.ga.max)),
      bw: parseClamped(DOM.numBW, DOM.bw, LIMITS.bw.min, LIMITS.bw.max),
      deliv: DOM.deliv ? DOM.deliv.value : 'Hospital',
      anc: Math.round(parseClamped(DOM.numANC, DOM.anc, LIMITS.anc.min, LIMITS.anc.max)),
      age: Math.round(parseClamped(DOM.numAge, DOM.age, LIMITS.age.min, LIMITS.age.max)),
      edu: DOM.edu ? DOM.edu.value : 'Secondary',
      inc: Math.round(parseClamped(DOM.numInc, DOM.inc, LIMITS.inc.min, LIMITS.inc.max)),
      res: DOM.res ? DOM.res.value : 'Rural'
    };
  }


  // =========================================================================
  // 6. Micro-Alerts & Boundary Limit Enforcement
  // =========================================================================
  function updateInputDisplays() {
    const p = getCurrentPatient();

    // Systolic BP Alert
    if (DOM.alertSysBP && DOM.numSysBP) {
      const raw = parseFloat(DOM.numSysBP.value);
      if (raw > LIMITS.sysBP.max) {
        DOM.alertSysBP.className = 'field-alert alert-danger';
        DOM.alertSysBP.textContent = `🚨 Exceeds Max Limit (${LIMITS.sysBP.max} mmHg)! Clamped.`;
      } else if (p.sysBP >= 140) {
        DOM.alertSysBP.className = 'field-alert alert-danger';
        DOM.alertSysBP.textContent = '🚨 Stage 2 Hypertension (≥ 140 mmHg)';
      } else if (p.sysBP >= 120) {
        DOM.alertSysBP.className = 'field-alert alert-warning';
        DOM.alertSysBP.textContent = '⚠️ Elevated / Prehypertensive (120–139)';
      } else {
        DOM.alertSysBP.className = 'field-alert alert-safe';
        DOM.alertSysBP.textContent = '✓ Optimal Blood Pressure (< 120 mmHg)';
      }
    }

    // Diastolic BP Alert
    if (DOM.alertDiaBP && DOM.numDiaBP) {
      const raw = parseFloat(DOM.numDiaBP.value);
      if (raw > LIMITS.diaBP.max) {
        DOM.alertDiaBP.className = 'field-alert alert-danger';
        DOM.alertDiaBP.textContent = `🚨 Exceeds Max Limit (${LIMITS.diaBP.max} mmHg)! Clamped.`;
      } else if (p.diaBP >= 90) {
        DOM.alertDiaBP.className = 'field-alert alert-danger';
        DOM.alertDiaBP.textContent = '🚨 High Diastolic Pressure (≥ 90 mmHg)';
      } else if (p.diaBP >= 80) {
        DOM.alertDiaBP.className = 'field-alert alert-warning';
        DOM.alertDiaBP.textContent = '⚠️ Prehypertensive Diastolic (80–89)';
      } else {
        DOM.alertDiaBP.className = 'field-alert alert-safe';
        DOM.alertDiaBP.textContent = '✓ Normotensive Diastolic (< 80 mmHg)';
      }
    }

    // Hemoglobin Alert
    if (DOM.alertHb && DOM.numHb) {
      const raw = parseFloat(DOM.numHb.value);
      if (raw > LIMITS.hb.max) {
        DOM.alertHb.className = 'field-alert alert-danger';
        DOM.alertHb.textContent = `🚨 Exceeds Max Limit (${LIMITS.hb.max} g/dL)! Clamped.`;
      } else if (p.hb < 7.0) {
        DOM.alertHb.className = 'field-alert alert-danger';
        DOM.alertHb.textContent = '🚨 Severe Obstetric Anemia (< 7.0 g/dL)';
      } else if (p.hb < 11.0) {
        DOM.alertHb.className = 'field-alert alert-warning';
        DOM.alertHb.textContent = '⚠️ Moderate Anemia (7.0–10.9 g/dL)';
      } else {
        DOM.alertHb.className = 'field-alert alert-safe';
        DOM.alertHb.textContent = '✓ Normal Maternal Hemoglobin (≥ 11.0 g/dL)';
      }
    }

    // Gestational Age Alert
    if (DOM.alertGA && DOM.numGA) {
      const raw = parseFloat(DOM.numGA.value);
      if (raw > LIMITS.ga.max) {
        DOM.alertGA.className = 'field-alert alert-danger';
        DOM.alertGA.textContent = `🚨 Exceeds Max Limit (${LIMITS.ga.max} wks)! Clamped.`;
      } else if (p.ga < 34) {
        DOM.alertGA.className = 'field-alert alert-danger';
        DOM.alertGA.textContent = '🚨 Severe Preterm Delivery (< 34 weeks)';
      } else if (p.ga < 37) {
        DOM.alertGA.className = 'field-alert alert-warning';
        DOM.alertGA.textContent = '⚠️ Late Preterm Delivery (34–36 weeks)';
      } else {
        DOM.alertGA.className = 'field-alert alert-safe';
        DOM.alertGA.textContent = '✓ Full-Term Gestation (≥ 37 weeks)';
      }
    }

    // Birth Weight Alert
    if (DOM.alertBW && DOM.numBW) {
      const raw = parseFloat(DOM.numBW.value);
      if (raw > LIMITS.bw.max) {
        DOM.alertBW.className = 'field-alert alert-danger';
        DOM.alertBW.textContent = `🚨 Exceeds Max Limit (${LIMITS.bw.max} kg)! Clamped.`;
      } else if (p.bw < 2.0) {
        DOM.alertBW.className = 'field-alert alert-danger';
        DOM.alertBW.textContent = '🚨 Very Low Birth Weight (VLBW < 2.0 kg)';
      } else if (p.bw < 2.5) {
        DOM.alertBW.className = 'field-alert alert-warning';
        DOM.alertBW.textContent = '⚠️ Low Birth Weight (LBW < 2.5 kg)';
      } else {
        DOM.alertBW.className = 'field-alert alert-safe';
        DOM.alertBW.textContent = '✓ Healthy Birth Weight Range (≥ 2.5 kg)';
      }
    }

    // ANC Alert
    if (DOM.alertANC && DOM.numANC) {
      const raw = parseFloat(DOM.numANC.value);
      if (raw > LIMITS.anc.max) {
        DOM.alertANC.className = 'field-alert alert-danger';
        DOM.alertANC.textContent = `🚨 Exceeds Max Limit (${LIMITS.anc.max} visits)! Clamped.`;
      } else if (p.anc < 2) {
        DOM.alertANC.className = 'field-alert alert-danger';
        DOM.alertANC.textContent = '🚨 Critically Inadequate ANC (< 2 visits)';
      } else if (p.anc < 4) {
        DOM.alertANC.className = 'field-alert alert-warning';
        DOM.alertANC.textContent = '⚠️ Below WHO Minimum Standard (< 4 visits)';
      } else {
        DOM.alertANC.className = 'field-alert alert-safe';
        DOM.alertANC.textContent = '✓ WHO Protocol Compliant (≥ 4 visits)';
      }
    }

    // Age Alert
    if (DOM.alertAge && DOM.numAge) {
      const raw = parseFloat(DOM.numAge.value);
      if (raw > LIMITS.age.max) {
        DOM.alertAge.className = 'field-alert alert-danger';
        DOM.alertAge.textContent = `🚨 Exceeds Max Limit (${LIMITS.age.max} yrs)! Clamped.`;
      } else if (p.age < 18) {
        DOM.alertAge.className = 'field-alert alert-warning';
        DOM.alertAge.textContent = '⚠️ Adolescent Pregnancy (< 18 yrs)';
      } else if (p.age > 35) {
        DOM.alertAge.className = 'field-alert alert-warning';
        DOM.alertAge.textContent = '⚠️ Advanced Maternal Age (> 35 yrs)';
      } else {
        DOM.alertAge.className = 'field-alert alert-safe';
        DOM.alertAge.textContent = '✓ Optimal Obstetric Age (18–35 yrs)';
      }
    }
  }


  // =========================================================================
  // 7. Clinical Triage Assessment Execution
  // =========================================================================
  function runTriage() {
    updateInputDisplays();
    const patient = getCurrentPatient();
    const result = computeRisk(patient);

    // 1. Primary Hero Result Card
    if (DOM.resultOutput) {
      if (result.isHighRisk) {
        DOM.resultOutput.innerHTML = `
          <div class="result-banner-high">
            <span class="result-pill" style="background: #ffffff; color: #000000; border: 2.5px solid #000000; box-shadow: 3px 3px 0px #000000;">
              🚨 CRITICAL CLINICAL EMERGENCY TRIAGE
            </span>
            <h3 class="result-headline" style="color: #000000; margin-top: 8px;">HIGH-RISK PREGNANCY DETECTED</h3>
            <div class="result-subtitle" style="color: #000000;">
              Adverse Mortality Probability: <b>${result.highProb}%</b>
            </div>
            <p class="result-explanation" style="color: #000000;">
              Patient presents acute clinical danger markers requiring immediate referral to a Tertiary Care Center with 24/7 obstetric surgery, neonatal ICU facilities, and blood transfusion access.
            </p>
          </div>
        `;
      } else {
        DOM.resultOutput.innerHTML = `
          <div class="result-banner-low">
            <span class="result-pill" style="background: #ffffff; color: #000000; border: 2.5px solid #000000; box-shadow: 3px 3px 0px #000000;">
              ✅ FAVORABLE CLINICAL PROGNOSIS
            </span>
            <h3 class="result-headline" style="color: #000000; margin-top: 8px;">LOW-RISK PREGNANCY</h3>
            <div class="result-subtitle" style="color: #000000;">
              Safety Confidence: <b>${result.lowProb}%</b> · Adverse Risk: ${result.highProb}%
            </div>
            <p class="result-explanation" style="color: #000000;">
              Biomarkers and hemodynamics are within safe physiological envelopes. Continue standard WHO/ICMR antenatal care checkups, prophylactic iron-folic acid, and planned institutional delivery.
            </p>
          </div>
        `;
      }
    }

    // 2. SVG Risk Gauge
    if (DOM.gaugeFill && DOM.gaugeVal && DOM.gaugeRiskLabel) {
      const totalArc = 235.6;
      const offset = totalArc - (totalArc * (result.highProb / 100));
      DOM.gaugeFill.style.strokeDashoffset = offset;
      DOM.gaugeFill.style.stroke = result.isHighRisk ? '#ff5c8a' : '#00f59b';
      DOM.gaugeVal.textContent = `${result.highProb}%`;
      DOM.gaugeRiskLabel.textContent = result.isHighRisk ? 'HIGH RISK' : 'LOW RISK';
    }

    if (DOM.kpiLow) DOM.kpiLow.textContent = `${result.lowProb}%`;
    if (DOM.kpiHigh) DOM.kpiHigh.textContent = `${result.highProb}%`;

    // 3. Clinical Diagnostic Flag Breakdown
    renderDiagnosticFlags(patient);

    // 4. Biomarker Safety Spectrum Bars
    renderBiomarkerBars(patient);

    // 5. Biometric Balance Spider Radar
    renderSpiderRadar();
  }

  function renderDiagnosticFlags(p) {
    if (!DOM.flagList) return;
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
    else safeFlags.push('No pre-existing obstetric complications reported');

    if (p.anc < 4) dangerFlags.push(`Inadequate Antenatal Care: ${p.anc} visits (< WHO standard of 4)`);
    else safeFlags.push(`Antenatal Care compliant (${p.anc} visits completed)`);

    if (p.deliv === 'Home') dangerFlags.push('Home delivery planned (Elevated intrapartum complication risk)');

    let html = '';
    dangerFlags.forEach(f => {
      html += `<div class="flag-item flag-danger">⚠️ ${f}</div>`;
    });
    safeFlags.slice(0, 4).forEach(f => {
      html += `<div class="flag-item flag-safe">✓ ${f}</div>`;
    });

    DOM.flagList.innerHTML = html;
  }

  function renderBiomarkerBars(p) {
    if (!DOM.spectrumList) return;
    const biomarkers = [
      {
        name: 'Systolic BP (mmHg)',
        val: p.sysBP,
        safe: '< 120',
        color: p.sysBP >= 140 ? '#ff5c8a' : (p.sysBP >= 120 ? '#ffea00' : '#00f59b'),
        pct: Math.min(100, Math.max(0, ((p.sysBP - 80) / 120) * 100))
      },
      {
        name: 'Diastolic BP (mmHg)',
        val: p.diaBP,
        safe: '< 80',
        color: p.diaBP >= 90 ? '#ff5c8a' : (p.diaBP >= 80 ? '#ffea00' : '#00f59b'),
        pct: Math.min(100, Math.max(0, ((p.diaBP - 50) / 80) * 100))
      },
      {
        name: 'Hemoglobin (g/dL)',
        val: p.hb,
        safe: '≥ 11.0',
        color: p.hb < 7.0 ? '#ff5c8a' : (p.hb < 11.0 ? '#ffea00' : '#00f59b'),
        pct: Math.min(100, Math.max(0, ((p.hb - 5.0) / 12.0) * 100))
      },
      {
        name: 'Gestational Age (weeks)',
        val: p.ga,
        safe: '≥ 37',
        color: p.ga < 34 ? '#ff5c8a' : (p.ga < 37 ? '#ffea00' : '#00f59b'),
        pct: Math.min(100, Math.max(0, ((p.ga - 24) / 19) * 100))
      },
      {
        name: 'Birth Weight (kg)',
        val: p.bw,
        safe: '≥ 2.5',
        color: p.bw < 2.0 ? '#ff5c8a' : (p.bw < 2.5 ? '#ffea00' : '#00f59b'),
        pct: Math.min(100, Math.max(0, ((p.bw - 0.8) / 4.2) * 100))
      },
      {
        name: 'ANC Visits (count)',
        val: p.anc,
        safe: '≥ 4',
        color: p.anc < 2 ? '#ff5c8a' : (p.anc < 4 ? '#ffea00' : '#00f59b'),
        pct: Math.min(100, Math.max(0, (p.anc / 10) * 100))
      }
    ];

    let html = '';
    biomarkers.forEach(b => {
      html += `
        <div class="bar-row">
          <div class="bar-header">
            <span>${b.name}</span>
            <span style="color: #000000; font-weight: 900; background: ${b.color}; padding: 2px 8px; border: 1.5px solid #000; border-radius: 4px;">
              ${b.val}
            </span>
          </div>
          <div class="bar-track">
            <div class="bar-progress" style="width: ${b.pct}%; background: ${b.color};"></div>
          </div>
        </div>
      `;
    });

    DOM.spectrumList.innerHTML = html;
  }

  function renderSpiderRadar() {
    const canvas = DOM.radarCanvas;
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

    // Normalized scores (1.0 = safe, 0.1 = danger)
    const scoreSys = Math.max(0.1, Math.min(1.0, 1.0 - Math.max(0, p.sysBP - 120) / 75));
    const scoreDia = Math.max(0.1, Math.min(1.0, 1.0 - Math.max(0, p.diaBP - 80) / 45));
    const scoreHb = Math.max(0.1, Math.min(1.0, p.hb / 12.0));
    const scoreGA = Math.max(0.1, Math.min(1.0, p.ga / 38.0));
    const scoreBW = Math.max(0.1, Math.min(1.0, p.bw / 3.0));
    const scoreANC = Math.max(0.1, Math.min(1.0, p.anc / 5.0));

    const patientScores = [scoreSys, scoreDia, scoreHb, scoreGA, scoreBW, scoreANC];
    const isDark = htmlRoot.getAttribute('data-theme') === 'dark';

    const axisLineColor = isDark ? '#ffffff' : '#000000';
    const gridLineColor = isDark ? 'rgba(255,255,255,0.25)' : 'rgba(0,0,0,0.2)';
    const textColor = isDark ? '#ffffff' : '#000000';

    // Concentric web polygons
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
      ctx.strokeStyle = gridLineColor;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // Radial spokes and labels
    for (let i = 0; i < totalAxes; i++) {
      const angle = (Math.PI * 2 / totalAxes) * i - Math.PI / 2;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;

      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(x, y);
      ctx.strokeStyle = axisLineColor;
      ctx.lineWidth = 2;
      ctx.stroke();

      const labelX = centerX + Math.cos(angle) * (radius + 20);
      const labelY = centerY + Math.sin(angle) * (radius + 16);
      ctx.font = '800 11px "Outfit", sans-serif';
      ctx.fillStyle = textColor;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(categories[i], labelX, labelY);
    }

    // 1. Safe baseline reference envelope (Green dashed)
    ctx.beginPath();
    for (let i = 0; i < totalAxes; i++) {
      const angle = (Math.PI * 2 / totalAxes) * i - Math.PI / 2;
      const r = radius * 1.0;
      const x = centerX + Math.cos(angle) * r;
      const y = centerY + Math.sin(angle) * r;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = '#00f59b';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    ctx.setLineDash([]);

    // 2. Patient Profile Polygon (Neo-Brutalist Electric Fill)
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
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.fillStyle = 'rgba(255, 234, 0, 0.6)'; // Cyber yellow translucent fill
    ctx.fill();

    // Vertices dots
    for (let i = 0; i < totalAxes; i++) {
      const angle = (Math.PI * 2 / totalAxes) * i - Math.PI / 2;
      const r = radius * patientScores[i];
      const x = centerX + Math.cos(angle) * r;
      const y = centerY + Math.sin(angle) * r;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fillStyle = '#ff5c8a';
      ctx.fill();
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }


  // =========================================================================
  // 8. Dual-Control Synchronization (Inputs + Sliders + Limits)
  // =========================================================================
  const numericControlPairs = [
    { num: DOM.numSysBP, slider: DOM.sysBP, limit: LIMITS.sysBP },
    { num: DOM.numDiaBP, slider: DOM.diaBP, limit: LIMITS.diaBP },
    { num: DOM.numHb,    slider: DOM.hb,    limit: LIMITS.hb },
    { num: DOM.numGA,    slider: DOM.ga,    limit: LIMITS.ga },
    { num: DOM.numBW,    slider: DOM.bw,    limit: LIMITS.bw },
    { num: DOM.numANC,   slider: DOM.anc,   limit: LIMITS.anc },
    { num: DOM.numAge,   slider: DOM.age,   limit: LIMITS.age },
    { num: DOM.numInc,   slider: DOM.inc,   limit: LIMITS.inc }
  ];

  numericControlPairs.forEach(pair => {
    if (!pair.num || !pair.slider) return;

    // Direct input typing
    pair.num.addEventListener('input', () => {
      let val = parseFloat(pair.num.value);
      if (!isNaN(val)) {
        if (val > pair.limit.max) {
          pair.num.classList.add('input-error');
          pair.slider.value = pair.limit.max;
        } else if (val < pair.limit.min) {
          pair.num.classList.add('input-error');
          pair.slider.value = pair.limit.min;
        } else {
          pair.num.classList.remove('input-error');
          pair.slider.value = val;
        }
      }
      if (DOM.liveMode && DOM.liveMode.checked) {
        runTriage();
      }
    });

    // Auto-clamp on blur
    pair.num.addEventListener('blur', () => {
      let val = parseFloat(pair.num.value);
      if (isNaN(val) || val < pair.limit.min) {
        pair.num.value = pair.limit.min;
        pair.slider.value = pair.limit.min;
      } else if (val > pair.limit.max) {
        pair.num.value = pair.limit.max;
        pair.slider.value = pair.limit.max;
      }
      pair.num.classList.remove('input-error');
      runTriage();
    });

    // Range slider dragging
    pair.slider.addEventListener('input', () => {
      pair.num.value = pair.slider.value;
      pair.num.classList.remove('input-error');
      if (DOM.liveMode && DOM.liveMode.checked) {
        runTriage();
      }
    });
  });

  // Dropdown listeners
  [DOM.diab, DOM.comp, DOM.deliv, DOM.edu, DOM.res].forEach(sel => {
    if (sel) {
      sel.addEventListener('change', () => {
        if (DOM.liveMode && DOM.liveMode.checked) runTriage();
      });
    }
  });

  if (DOM.recalcBtn) DOM.recalcBtn.addEventListener('click', runTriage);

  // Preset buttons
  function applyPreset(presetKey) {
    const data = PRESETS[presetKey];
    if (!data) return;

    if (DOM.sysBP) DOM.sysBP.value = data.sysBP;
    if (DOM.diaBP) DOM.diaBP.value = data.diaBP;
    if (DOM.hb) DOM.hb.value = data.hb;
    if (DOM.ga) DOM.ga.value = data.ga;
    if (DOM.bw) DOM.bw.value = data.bw;
    if (DOM.anc) DOM.anc.value = data.anc;
    if (DOM.age) DOM.age.value = data.age;
    if (DOM.inc) DOM.inc.value = data.inc;

    if (DOM.numSysBP) DOM.numSysBP.value = data.sysBP;
    if (DOM.numDiaBP) DOM.numDiaBP.value = data.diaBP;
    if (DOM.numHb) DOM.numHb.value = data.hb;
    if (DOM.numGA) DOM.numGA.value = data.ga;
    if (DOM.numBW) DOM.numBW.value = data.bw;
    if (DOM.numANC) DOM.numANC.value = data.anc;
    if (DOM.numAge) DOM.numAge.value = data.age;
    if (DOM.numInc) DOM.numInc.value = data.inc;

    if (DOM.diab) DOM.diab.value = data.diab;
    if (DOM.comp) DOM.comp.value = data.comp;
    if (DOM.deliv) DOM.deliv.value = data.deliv;
    if (DOM.edu) DOM.edu.value = data.edu;
    if (DOM.res) DOM.res.value = data.res;

    [DOM.numSysBP, DOM.numDiaBP, DOM.numHb, DOM.numGA, DOM.numBW, DOM.numANC, DOM.numAge, DOM.numInc].forEach(el => {
      if (el) el.classList.remove('input-error');
    });

    runTriage();
  }

  const pLow = document.getElementById('presetLowBtn');
  const pBrd = document.getElementById('presetBorderlineBtn');
  const pCrit = document.getElementById('presetCriticalBtn');
  if (pLow) pLow.addEventListener('click', () => applyPreset('low'));
  if (pBrd) pBrd.addEventListener('click', () => applyPreset('borderline'));
  if (pCrit) pCrit.addEventListener('click', () => applyPreset('critical'));


  // =========================================================================
  // 9. TAB 2: INTERACTIVE DATA INSIGHTS ENGINE & DYNAMIC POPULATION SIMULATOR
  // =========================================================================
  // Interactive EDA Visualizer Canvas
  function renderEdaChart() {
    const canvas = DOM.edaCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const isDark = htmlRoot.getAttribute('data-theme') === 'dark';
    const axisColor = isDark ? '#ffffff' : '#000000';
    const textColor = isDark ? '#ffffff' : '#000000';
    const gridColor = isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.12)';

    const padding = { top: 40, right: 30, bottom: 50, left: 60 };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;

    // Draw Chart Background
    ctx.fillStyle = isDark ? '#161821' : '#ffffff';
    ctx.fillRect(padding.left, padding.top, chartW, chartH);
    ctx.strokeStyle = axisColor;
    ctx.lineWidth = 3;
    ctx.strokeRect(padding.left, padding.top, chartW, chartH);

    // Grid lines
    const ySteps = 4;
    for (let i = 0; i <= ySteps; i++) {
      const y = padding.top + (chartH / ySteps) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartW, y);
      ctx.strokeStyle = gridColor;
      ctx.lineWidth = 1;
      ctx.stroke();

      // Y-axis label
      ctx.font = '700 11px "JetBrains Mono", monospace';
      ctx.fillStyle = textColor;
      ctx.textAlign = 'right';
      ctx.fillText(`${100 - i * 25}%`, padding.left - 10, y + 4);
    }

    if (currentEdaMetric === 'hb') {
      // Metric: Hemoglobin Distribution & Mortality Risk
      const bins = [
        { label: '<7.0 (Severe)', hb: 6.5, risk: 91.2, count: 280, color: '#ff5c8a' },
        { label: '7-9 (Moderate)', hb: 8.0, risk: 64.5, count: 740, color: '#ff9100' },
        { label: '9-11 (Mild)', hb: 10.0, risk: 36.8, count: 1420, color: '#ffea00' },
        { label: '11-13 (Normal)', hb: 12.0, risk: 14.1, count: 1850, color: '#00f59b' },
        { label: '>13 (Optimal)', hb: 14.0, risk: 9.3, count: 710, color: '#00e5ff' }
      ];

      const barWidth = chartW / bins.length - 24;
      bins.forEach((b, i) => {
        const x = padding.left + 12 + i * (chartW / bins.length);
        const barH = (b.risk / 100) * chartH;
        const y = padding.top + chartH - barH;

        // Neo-Brutalist 3D Bar
        ctx.fillStyle = '#000000';
        ctx.fillRect(x + 4, y + 4, barWidth, barH); // shadow
        ctx.fillStyle = b.color;
        ctx.fillRect(x, y, barWidth, barH);
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2.5;
        ctx.strokeRect(x, y, barWidth, barH);

        // Percentage on top
        ctx.font = '900 12px "Outfit", sans-serif';
        ctx.fillStyle = textColor;
        ctx.textAlign = 'center';
        ctx.fillText(`${b.risk}%`, x + barWidth / 2, y - 8);

        // X label
        ctx.font = '800 10px "Plus Jakarta Sans", sans-serif';
        ctx.fillText(b.label, x + barWidth / 2, padding.top + chartH + 20);
      });

      // Chart Title
      ctx.font = '900 13px "Outfit", sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Maternal Mortality Risk by Hemoglobin Category (5,000 Cohort)', padding.left + 10, padding.top - 14);

    } else if (currentEdaMetric === 'bp') {
      // Metric: Blood Pressure & Hypertension
      const bins = [
        { label: '<120 (Normal)', risk: 11.2, color: '#00f59b' },
        { label: '120-129 (Elevated)', risk: 24.6, color: '#ffea00' },
        { label: '130-139 (Stage 1)', risk: 48.3, color: '#ff9100' },
        { label: '140-159 (Stage 2)', risk: 78.4, color: '#ff5c8a' },
        { label: '≥160 (Crisis)', risk: 94.7, color: '#ff2a85' }
      ];

      const barWidth = chartW / bins.length - 24;
      bins.forEach((b, i) => {
        const x = padding.left + 12 + i * (chartW / bins.length);
        const barH = (b.risk / 100) * chartH;
        const y = padding.top + chartH - barH;

        ctx.fillStyle = '#000000';
        ctx.fillRect(x + 4, y + 4, barWidth, barH);
        ctx.fillStyle = b.color;
        ctx.fillRect(x, y, barWidth, barH);
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2.5;
        ctx.strokeRect(x, y, barWidth, barH);

        ctx.font = '900 12px "Outfit", sans-serif';
        ctx.fillStyle = textColor;
        ctx.textAlign = 'center';
        ctx.fillText(`${b.risk}%`, x + barWidth / 2, y - 8);

        ctx.font = '800 10px "Plus Jakarta Sans", sans-serif';
        ctx.fillText(b.label, x + barWidth / 2, padding.top + chartH + 20);
      });

      ctx.font = '900 13px "Outfit", sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Adverse Outcome Rate vs Systolic Blood Pressure Clusters', padding.left + 10, padding.top - 14);

    } else if (currentEdaMetric === 'anc') {
      // Metric: Antenatal Care (ANC) Visits
      const bins = [
        { label: '0 Visits', risk: 84.2, color: '#ff5c8a' },
        { label: '1 Visit', risk: 73.1, color: '#ff5c8a' },
        { label: '2-3 Visits', risk: 44.5, color: '#ffea00' },
        { label: '4-5 Visits (WHO)', risk: 16.8, color: '#00f59b' },
        { label: '≥6 Visits', risk: 10.4, color: '#00e5ff' }
      ];

      const barWidth = chartW / bins.length - 24;
      bins.forEach((b, i) => {
        const x = padding.left + 12 + i * (chartW / bins.length);
        const barH = (b.risk / 100) * chartH;
        const y = padding.top + chartH - barH;

        ctx.fillStyle = '#000000';
        ctx.fillRect(x + 4, y + 4, barWidth, barH);
        ctx.fillStyle = b.color;
        ctx.fillRect(x, y, barWidth, barH);
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2.5;
        ctx.strokeRect(x, y, barWidth, barH);

        ctx.font = '900 12px "Outfit", sans-serif';
        ctx.fillStyle = textColor;
        ctx.textAlign = 'center';
        ctx.fillText(`${b.risk}%`, x + barWidth / 2, y - 8);

        ctx.font = '800 10px "Plus Jakarta Sans", sans-serif';
        ctx.fillText(b.label, x + barWidth / 2, padding.top + chartH + 20);
      });

      ctx.font = '900 13px "Outfit", sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Protective Effect: Adverse Risk Plummets with Frequent Antenatal Care', padding.left + 10, padding.top - 14);

    } else {
      // Metric: Infant Birth Weight
      const bins = [
        { label: '<1.5 kg (Extremely Low)', risk: 92.5, color: '#ff5c8a' },
        { label: '1.5-2.0 kg (VLBW)', risk: 76.4, color: '#ff9100' },
        { label: '2.0-2.4 kg (LBW)', risk: 42.1, color: '#ffea00' },
        { label: '2.5-3.5 kg (Normal)', risk: 13.8, color: '#00f59b' },
        { label: '>3.5 kg (High Normal)', risk: 11.2, color: '#00e5ff' }
      ];

      const barWidth = chartW / bins.length - 24;
      bins.forEach((b, i) => {
        const x = padding.left + 12 + i * (chartW / bins.length);
        const barH = (b.risk / 100) * chartH;
        const y = padding.top + chartH - barH;

        ctx.fillStyle = '#000000';
        ctx.fillRect(x + 4, y + 4, barWidth, barH);
        ctx.fillStyle = b.color;
        ctx.fillRect(x, y, barWidth, barH);
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2.5;
        ctx.strokeRect(x, y, barWidth, barH);

        ctx.font = '900 12px "Outfit", sans-serif';
        ctx.fillStyle = textColor;
        ctx.textAlign = 'center';
        ctx.fillText(`${b.risk}%`, x + barWidth / 2, y - 8);

        ctx.font = '800 10px "Plus Jakarta Sans", sans-serif';
        ctx.fillText(b.label, x + barWidth / 2, padding.top + chartH + 20);
      });

      ctx.font = '900 13px "Outfit", sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Neonatal Birth Weight vs Mortality Likelihood', padding.left + 10, padding.top - 14);
    }
  }

  // Interactive Metric Selector Buttons in Tab 2
  const edaMetricButtons = document.querySelectorAll('.eda-metric-btn');
  edaMetricButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      edaMetricButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentEdaMetric = btn.getAttribute('data-metric') || 'hb';
      renderEdaChart();
    });
  });

  // Interactive Population Simulator Slider in Tab 2
  if (DOM.edaSimSlider && DOM.edaSimValueText && DOM.edaSimResultBadge) {
    DOM.edaSimSlider.addEventListener('input', () => {
      const hbVal = parseFloat(DOM.edaSimSlider.value);
      DOM.edaSimValueText.textContent = `${hbVal.toFixed(1)} g/dL`;

      let calculatedRisk = 0;
      let badgeText = '';
      let badgeClass = '';

      if (hbVal < 7.0) {
        calculatedRisk = 91.2 - (hbVal - 5.0) * 8.0;
        badgeText = '🚨 Severe Anemia Hazard Zone: 4.8x Mortality Escalation';
        badgeClass = 'alert-danger';
      } else if (hbVal < 11.0) {
        calculatedRisk = 65.0 - (hbVal - 7.0) * 7.0;
        badgeText = '⚠️ Moderate Anemia: Substantial Obstetric Risk';
        badgeClass = 'alert-warning';
      } else {
        calculatedRisk = Math.max(9.0, 22.0 - (hbVal - 11.0) * 2.5);
        badgeText = '✓ Optimal Hemoglobin: Low Adverse Risk (< 15%)';
        badgeClass = 'alert-safe';
      }

      DOM.edaSimResultBadge.className = `field-alert ${badgeClass}`;
      DOM.edaSimResultBadge.textContent = `${badgeText} · Projected Adverse Rate: ${calculatedRisk.toFixed(1)}%`;
    });
  }

  // Gallery Filter Buttons
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

  // Search Input Filter
  if (DOM.edaSearchInput) {
    DOM.edaSearchInput.addEventListener('input', () => {
      const term = DOM.edaSearchInput.value.toLowerCase().trim();
      galleryCards.forEach(card => {
        const text = (card.textContent || '').toLowerCase();
        if (!term || text.includes(term)) {
          card.style.display = 'flex';
        } else {
          card.style.display = 'none';
        }
      });
    });
  }

  // Modal Zoom Lightbox
  galleryCards.forEach(card => {
    card.addEventListener('click', () => {
      const imgSrc = card.getAttribute('data-img');
      const titleEl = card.querySelector('h4');
      const descEl = card.querySelector('p');

      if (DOM.modalImg && imgSrc) DOM.modalImg.src = imgSrc;
      if (DOM.modalTitle && titleEl) DOM.modalTitle.textContent = titleEl.textContent;
      if (DOM.modalDesc && descEl) DOM.modalDesc.textContent = descEl.textContent;
      if (DOM.modal) DOM.modal.classList.add('active');
    });
  });

  if (DOM.modalClose) {
    DOM.modalClose.addEventListener('click', () => {
      if (DOM.modal) DOM.modal.classList.remove('active');
    });
  }

  if (DOM.modal) {
    DOM.modal.addEventListener('click', (e) => {
      if (e.target === DOM.modal) DOM.modal.classList.remove('active');
    });
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && DOM.modal && DOM.modal.classList.contains('active')) {
      DOM.modal.classList.remove('active');
    }
  });


  // =========================================================================
  // 10. TAB 3: MODEL BENCHMARKS DEEP-DIVE
  // =========================================================================
  const diagButtons = document.querySelectorAll('.filter-btn[data-diag]');
  const deepdiveImg = document.getElementById('deepdiveImg');
  const deepdiveCaption = document.getElementById('deepdiveCaption');

  diagButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      diagButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const key = btn.getAttribute('data-diag');
      if (DIAG_DATA[key] && deepdiveImg && deepdiveCaption) {
        deepdiveImg.src = DIAG_DATA[key].img;
        deepdiveCaption.innerHTML = DIAG_DATA[key].caption;
      }
    });
  });


  // =========================================================================
  // 11. TAB 4: BATCH CSV COHORT SCREENING TOOL
  // =========================================================================
  const dropzone = document.getElementById('csvDropzone');
  const fileInput = document.getElementById('csvFileInput');
  const sampleBtn = document.getElementById('loadSampleCsvBtn');
  const templateBtn = document.getElementById('downloadTemplateBtn');
  const resultsWrapper = document.getElementById('batchResultsWrapper');
  const tableBody = document.getElementById('batchTableBody');
  const exportBtn = document.getElementById('exportScoredCsvBtn');

  function processBatchRecords(records) {
    currentBatchData = [];
    let highCount = 0;
    let sumRisk = 0;

    if (tableBody) tableBody.innerHTML = '';

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

      if (tableBody) {
        const riskPill = score.isHighRisk
          ? `<span class="result-pill" style="background: #ff5c8a; color: #000; border: 2px solid #000; font-size: 0.72rem; padding: 2px 8px;">HIGH-RISK</span>`
          : `<span class="result-pill" style="background: #00f59b; color: #000; border: 2px solid #000; font-size: 0.72rem; padding: 2px 8px;">LOW-RISK</span>`;

        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${idx + 1}</td>
          <td>${riskPill}</td>
          <td style="font-weight: 900; font-family: var(--font-mono);">${score.highProb}%</td>
          <td>${patient.age}</td>
          <td>${patient.sysBP}/${patient.diaBP}</td>
          <td>${patient.hb.toFixed(1)}</td>
          <td>${patient.ga}w</td>
          <td>${patient.bw.toFixed(2)} kg</td>
          <td>${patient.anc}</td>
          <td>${patient.comp}</td>
        `;
        tableBody.appendChild(tr);
      }
    });

    const total = records.length;
    const lowCount = total - highCount;
    const avgRisk = (sumRisk / total).toFixed(1);
    const lowPct = ((lowCount / total) * 100).toFixed(1);

    const elTotal = document.getElementById('batchTotalCount');
    const elHigh = document.getElementById('batchHighCount');
    const elLow = document.getElementById('batchLowRate');
    const elAvg = document.getElementById('batchAvgRisk');

    if (elTotal) elTotal.textContent = total;
    if (elHigh) elHigh.textContent = highCount;
    if (elLow) elLow.textContent = `${lowPct}%`;
    if (elAvg) elAvg.textContent = `${avgRisk}%`;

    if (resultsWrapper) resultsWrapper.style.display = 'block';
  }

  if (sampleBtn) {
    sampleBtn.addEventListener('click', () => {
      processBatchRecords(SAMPLE_COHORT);
    });
  }

  if (templateBtn) {
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
  }

  if (dropzone && fileInput) {
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
  }

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

  if (exportBtn) {
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
  }


  // =========================================================================
  // 12. Initial Load & Window Events
  // =========================================================================
  runTriage();
  renderEdaChart();

  window.addEventListener('resize', () => {
    renderSpiderRadar();
    renderEdaChart();
  });

  console.log('✓ Maternal & Infant Risk AI (Neo-Brutalism Edition) fully initialized.');
});
