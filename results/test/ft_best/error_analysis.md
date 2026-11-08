# Error analysis: results/test/ft_best/predictions.jsonl

Rows: 206; label errors: 63

| Category | Count |
|---|---|
| E2_unsupported_certainty | 38 |
| E3_missed_available_information | 5 |
| E4_polarity_error | 7 |
| E5_applicability_error | 13 |
| E6_definite_label_without_evidence | 115 |
| E7_spurious_citation | 17 |
| E8_correct_label_wrong_evidence | 20 |
| E9_right_evidence_wrong_label | 3 |

## E2_unsupported_certainty (38)
- **sigir-201410 / NCT02097186 (exclusion)** — criterion: _Patients with an estimated pre-operative glomerular filtration rate < 30mls/min/1.73m2_
  - gold: `not enough information` evidence [] | pred: `not excluded` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not excluded"}<|im_end|>`
- **sigir-201410 / NCT02097186 (exclusion)** — criterion: _Patients with previous axillary surgery_
  - gold: `not enough information` evidence [] | pred: `not excluded` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not excluded"}<|im_end|>`
- **sigir-201411 / NCT01526382 (exclusion)** — criterion: _Patients were moribund._
  - gold: `not enough information` evidence [0, 1, 2, 3, 4] | pred: `not excluded` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not excluded"}<|im_end|>`

## E3_missed_available_information (5)
- **sigir-201410 / NCT02097186 (inclusion)** — criterion: _Patients undergoing elective carotid endarterectomy_
  - gold: `not included` evidence [] | pred: `not enough information` evidence [0]
  - cited text: {"0": "A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning."}
  - raw output: `{"evidence_ids": [0], "label": "not enough information"}<|im_end|>`
- **sigir-201410 / NCT02264964 (inclusion)** — criterion: _Maintenance hemodialysis after central vena catheterization._
  - gold: `not included` evidence [] | pred: `not enough information` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not enough information"}<|im_end|>`
- **sigir-201411 / NCT01526382 (inclusion)** — criterion: _Patients were included if they were diagnosed with Shock, Acute respiratory distress syndrome (ARDS), or both._
  - gold: `included` evidence [2, 3] | pred: `not enough information` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not enough information"}<|im_end|>`

## E4_polarity_error (7)
- **sigir-201410 / NCT02097186 (inclusion)** — criterion: _Patients undergoing open abdominal aortic aneurysm repair_
  - gold: `not included` evidence [] | pred: `included` evidence [0, 1, 2]
  - cited text: {"0": "A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.", "1": "She is now complaining of a cool right foot.", "2": "Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered."}
  - raw output: `{"evidence_ids": [0, 1, 2], "label": "included"}<|im_end|>`
- **sigir-201410 / NCT02097186 (inclusion)** — criterion: _Patients undergoing endovascular abdominal aneurysm repair_
  - gold: `not included` evidence [] | pred: `included` evidence [0, 2]
  - cited text: {"0": "A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.", "2": "Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered."}
  - raw output: `{"evidence_ids": [0, 2], "label": "included"}<|im_end|>`
- **sigir-201410 / NCT02097186 (inclusion)** — criterion: _Patients undergoing surgical lower limb revascularisation (suprainguinal or infrainguinal)_
  - gold: `not included` evidence [] | pred: `included` evidence [0, 2]
  - cited text: {"0": "A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.", "2": "Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered."}
  - raw output: `{"evidence_ids": [0, 2], "label": "included"}<|im_end|>`

## E5_applicability_error (13)
- **sigir-201417 / NCT01745731 (exclusion)** — criterion: _Pregnant women or women of childbearing age who do not have adequate contraception._
  - gold: `not excluded` evidence [0] | pred: `not applicable` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not applicable"}<|im_end|>`
- **sigir-201422 / NCT00236912 (inclusion)** — criterion: _If female, using birth control_
  - gold: `not applicable` evidence [3] | pred: `not enough information` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not enough information"}<|im_end|>`
- **sigir-20158 / NCT02562040 (exclusion)** — criterion: _Previous tonsillectomy, including partial tonsillectomy_
  - gold: `not applicable` evidence [] | pred: `not excluded` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not excluded"}<|im_end|>`

## E6_definite_label_without_evidence (115)
- **sigir-201410 / NCT02097186 (exclusion)** — criterion: _Previous history of upper limb deep vein thrombosis_
  - gold: `not excluded` evidence [] | pred: `not excluded` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not excluded"}<|im_end|>`
- **sigir-201410 / NCT02097186 (exclusion)** — criterion: _Patients on glibenclamide or nicorandil (these medications may interfere with RIPC)_
  - gold: `not excluded` evidence [] | pred: `not excluded` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not excluded"}<|im_end|>`
- **sigir-201410 / NCT02097186 (exclusion)** — criterion: _Patients with an estimated pre-operative glomerular filtration rate < 30mls/min/1.73m2_
  - gold: `not enough information` evidence [] | pred: `not excluded` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not excluded"}<|im_end|>`

## E7_spurious_citation (17)
- **sigir-201410 / NCT02097186 (inclusion)** — criterion: _Patients undergoing elective carotid endarterectomy_
  - gold: `not included` evidence [] | pred: `not enough information` evidence [0]
  - cited text: {"0": "A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning."}
  - raw output: `{"evidence_ids": [0], "label": "not enough information"}<|im_end|>`
- **sigir-201410 / NCT02097186 (inclusion)** — criterion: _Patients undergoing open abdominal aortic aneurysm repair_
  - gold: `not included` evidence [] | pred: `included` evidence [0, 1, 2]
  - cited text: {"0": "A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.", "1": "She is now complaining of a cool right foot.", "2": "Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered."}
  - raw output: `{"evidence_ids": [0, 1, 2], "label": "included"}<|im_end|>`
- **sigir-201410 / NCT02097186 (inclusion)** — criterion: _Patients undergoing endovascular abdominal aneurysm repair_
  - gold: `not included` evidence [] | pred: `included` evidence [0, 2]
  - cited text: {"0": "A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.", "2": "Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered."}
  - raw output: `{"evidence_ids": [0, 2], "label": "included"}<|im_end|>`

## E8_correct_label_wrong_evidence (20)
- **sigir-201410 / NCT02264964 (inclusion)** — criterion: _No medical history of central vena catheterization._
  - gold: `included` evidence [0] | pred: `included` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "included"}<|im_end|>`
- **sigir-201411 / NCT01723137 (exclusion)** — criterion: _Less than 18 years of age_
  - gold: `not excluded` evidence [0] | pred: `not excluded` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not excluded"}<|im_end|>`
- **sigir-201411 / NCT01723137 (exclusion)** — criterion: _Decreased level of consciousness_
  - gold: `not excluded` evidence [0, 5] | pred: `not excluded` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not excluded"}<|im_end|>`

## E9_right_evidence_wrong_label (3)
- **sigir-201417 / NCT01745731 (inclusion)** — criterion: _Patients with liver space occupying lesion (LOE) that require extended hepatic resection._
  - gold: `not enough information` evidence [3] | pred: `included` evidence [3]
  - cited text: {"3": "Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained."}
  - raw output: `{"evidence_ids": [3], "label": "included"}<|im_end|>`
- **sigir-201417 / NCT02556359 (inclusion)** — criterion: _Immune Deficiency and early BMF in childhood_
  - gold: `not enough information` evidence [0] | pred: `included` evidence [0]
  - cited text: {"0": "A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40."}
  - raw output: `{"evidence_ids": [0], "label": "included"}<|im_end|>`
- **sigir-20158 / NCT02562040 (inclusion)** — criterion: _Diagnosis of mild sleep-disordered breathing (MSDB) defined as meeting all of the following criteria: Caregiver report of habitual snoring that occurs most of the night on at least three nights per week, and has been present for at least three months (on average occurring > 3 nights per week or more_
  - gold: `not enough information` evidence [0] | pred: `included` evidence [0]
  - cited text: {"0": "A 10 yo boy with nighttime snoring, pauses in breathing, and restlessness with nighttime awakenings."}
  - raw output: `{"evidence_ids": [0], "label": "included"}<|im_end|>`