# Error analysis: results/val/ft_best/predictions.jsonl

Rows: 206; label errors: 63

| Category | Count |
|---|---|
| E2_unsupported_certainty | 40 |
| E3_missed_available_information | 3 |
| E4_polarity_error | 9 |
| E5_applicability_error | 11 |
| E6_definite_label_without_evidence | 96 |
| E7_spurious_citation | 24 |
| E8_correct_label_wrong_evidence | 31 |
| E9_right_evidence_wrong_label | 10 |

## E2_unsupported_certainty (40)
- **sigir-20141 / NCT01397994 (exclusion)** — criterion: _Unstable angina_
  - gold: `not enough information` evidence [0, 1, 2] | pred: `not excluded` evidence [0, 1, 2]
  - cited text: {"0": "A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.", "1": "The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration.", "2": "The latest episode of pain ended half an hour prior
  - raw output: `{"evidence_ids": [0, 1, 2], "label": "not excluded"}<|im_end|>`
- **sigir-20141 / NCT01397994 (exclusion)** — criterion: _Pregnant and nursing women_
  - gold: `not enough information` evidence [] | pred: `not excluded` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not excluded"}<|im_end|>`
- **sigir-20141 / NCT01397994 (exclusion)** — criterion: _Patients not eligible for Tc 99m SPECT_
  - gold: `not enough information` evidence [] | pred: `not excluded` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not excluded"}<|im_end|>`

## E3_missed_available_information (3)
- **sigir-20141 / NCT00149227 (inclusion)** — criterion: _Clinical diagnosis of one or more risk factors, such as diabetes, smoking habit, lipid metabolism abnormality, history of ischemic heart disease (IHD) or cerebrovascular disease, obesity (BMI>25), chronic heart failure (NYHA II-III), and electrocardiogram (ECG) abnormality (LVH)_
  - gold: `included` evidence [3, 4] | pred: `not enough information` evidence [3, 4]
  - cited text: {"3": "She is known to have hypertension and obesity.", "4": "She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease."}
  - raw output: `{"evidence_ids": [3, 4], "label": "not enough information"}<|im_end|>`
- **sigir-201420 / NCT01594385 (inclusion)** — criterion: _Life expectancy longer than 48 hours_
  - gold: `included` evidence [2, 3] | pred: `not enough information` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not enough information"}<|im_end|>`
- **sigir-201427 / NCT01187901 (inclusion)** — criterion: _Patient must discontinue taking any Nonsteroidal anti-inflammatory drugs (NSAIDS) within one month of treatment initiation._
  - gold: `included` evidence [] | pred: `not enough information` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not enough information"}<|im_end|>`

## E4_polarity_error (9)
- **sigir-20142 / NCT02618655 (inclusion)** — criterion: _patients have fever more than one week_
  - gold: `not included` evidence [0] | pred: `included` evidence [0]
  - cited text: {"0": "An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days."}
  - raw output: `{"evidence_ids": [0], "label": "included"}<|im_end|>`
- **sigir-201415 / NCT00277680 (exclusion)** — criterion: _Uterus size exceeding the umbilical level_
  - gold: `not excluded` evidence [1] | pred: `excluded` evidence [4]
  - cited text: {"4": "The ultrasound reports enlarged uterus (12 cm x 9 cm x 7 cms) with multiple cystic areas in the interior."}
  - raw output: `{"evidence_ids": [4], "label": "excluded"}<|im_end|>`
- **sigir-201415 / NCT00180739 (exclusion)** — criterion: _2. Uterine size >20 weeks as evaluated by US or MR._
  - gold: `not excluded` evidence [1] | pred: `excluded` evidence [4]
  - cited text: {"4": "The ultrasound reports enlarged uterus (12 cm x 9 cm x 7 cms) with multiple cystic areas in the interior."}
  - raw output: `{"evidence_ids": [4], "label": "excluded"}<|im_end|>`

## E5_applicability_error (11)
- **sigir-20141 / NCT00149227 (exclusion)** — criterion: _Patients with IHD within 6 months after percutaneous coronary intervention(PCI), and who are stable but are going to implement PCI or coronary artery bypass grafting(CABG)_
  - gold: `not applicable` evidence [] | pred: `not excluded` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not excluded"}<|im_end|>`
- **sigir-20141 / NCT00149227 (exclusion)** — criterion: _Pregnant women and women of childbearing potential_
  - gold: `not enough information` evidence [0] | pred: `not applicable` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not applicable"}<|im_end|>`
- **sigir-20142 / NCT02618655 (inclusion)** — criterion: _full of physical examination and laboratory examination have been carried out after one week，but still cannot make a definite diagnosis_
  - gold: `not applicable` evidence [] | pred: `not enough information` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not enough information"}<|im_end|>`

## E6_definite_label_without_evidence (96)
- **sigir-20141 / NCT01397994 (exclusion)** — criterion: _Valvular heart disease and cardiomyopathy_
  - gold: `not excluded` evidence [] | pred: `not excluded` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not excluded"}<|im_end|>`
- **sigir-20141 / NCT01397994 (exclusion)** — criterion: _Myocardial infarction in < 6 months_
  - gold: `not excluded` evidence [] | pred: `not excluded` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not excluded"}<|im_end|>`
- **sigir-20141 / NCT01397994 (exclusion)** — criterion: _Congestive cardiac failure_
  - gold: `not excluded` evidence [] | pred: `not excluded` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not excluded"}<|im_end|>`

## E7_spurious_citation (24)
- **sigir-20141 / NCT01397994 (exclusion)** — criterion: _Hypertension of > 170/100 mm of Hg_
  - gold: `not excluded` evidence [] | pred: `not excluded` evidence [3]
  - cited text: {"3": "She is known to have hypertension and obesity."}
  - raw output: `{"evidence_ids": [3], "label": "not excluded"}<|im_end|>`
- **sigir-20142 / NCT02618655 (exclusion)** — criterion: _fever for non-infectious diseases such as rheumatic autoimmune disease or with tumor_
  - gold: `not excluded` evidence [] | pred: `not excluded` evidence [0]
  - cited text: {"0": "An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days."}
  - raw output: `{"evidence_ids": [0], "label": "not excluded"}<|im_end|>`
- **sigir-20142 / NCT02618655 (exclusion)** — criterion: _we find that the patient selected does not meet the selection criteria within the observation period_
  - gold: `not enough information` evidence [] | pred: `not excluded` evidence [0, 1, 2, 3, 4, 5]
  - cited text: {"0": "An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days.", "1": "He has just returned from a 5 day vacation in Colorado.", "2": "Parents report that prior to the onset of fever and cough, he had loose stools.", "3": "He denies upper respiratory tract symptoms.", "4": "On examination he is in respiratory distress and has bronchial respiratory sounds
  - raw output: `{"evidence_ids": [0, 1, 2, 3, 4, 5], "label": "not excluded"}<|im_end|>`

## E8_correct_label_wrong_evidence (31)
- **sigir-20141 / NCT01397994 (inclusion)** — criterion: _Patients of chronic stable angina with abnormal Exercise Myocardial Perfusion Spect Scan with reversible and partially reversible ischemic changes._
  - gold: `not enough information` evidence [0, 1, 2] | pred: `not enough information` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not enough information"}<|im_end|>`
- **sigir-20141 / NCT01397994 (exclusion)** — criterion: _IDDM (Type-1 diabetes mellitus)_
  - gold: `not excluded` evidence [4] | pred: `not excluded` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not excluded"}<|im_end|>`
- **sigir-20141 / NCT01397994 (exclusion)** — criterion: _Systolic blood pressure < 100 mm Hg_
  - gold: `not excluded` evidence [3] | pred: `not excluded` evidence []
  - cited text: {}
  - raw output: `{"evidence_ids": [], "label": "not excluded"}<|im_end|>`

## E9_right_evidence_wrong_label (10)
- **sigir-20141 / NCT01397994 (exclusion)** — criterion: _Unstable angina_
  - gold: `not enough information` evidence [0, 1, 2] | pred: `not excluded` evidence [0, 1, 2]
  - cited text: {"0": "A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.", "1": "The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration.", "2": "The latest episode of pain ended half an hour prior
  - raw output: `{"evidence_ids": [0, 1, 2], "label": "not excluded"}<|im_end|>`
- **sigir-20141 / NCT00149227 (inclusion)** — criterion: _Clinical diagnosis of one or more risk factors, such as diabetes, smoking habit, lipid metabolism abnormality, history of ischemic heart disease (IHD) or cerebrovascular disease, obesity (BMI>25), chronic heart failure (NYHA II-III), and electrocardiogram (ECG) abnormality (LVH)_
  - gold: `included` evidence [3, 4] | pred: `not enough information` evidence [3, 4]
  - cited text: {"3": "She is known to have hypertension and obesity.", "4": "She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease."}
  - raw output: `{"evidence_ids": [3, 4], "label": "not enough information"}<|im_end|>`
- **sigir-20142 / NCT02618655 (inclusion)** — criterion: _patients have fever more than one week_
  - gold: `not included` evidence [0] | pred: `included` evidence [0]
  - cited text: {"0": "An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days."}
  - raw output: `{"evidence_ids": [0], "label": "included"}<|im_end|>`