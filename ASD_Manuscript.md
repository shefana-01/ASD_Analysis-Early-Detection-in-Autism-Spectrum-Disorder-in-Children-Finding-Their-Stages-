Integrating Machine Learning Classification with Longitudinal Case Narratives for Early ASD Detection

Abstract

Early detection of Autism Spectrum Disorder (ASD) significantly correlates with improved cognitive outcomes. This research presents a hybrid assessment framework that combines high-accuracy machine learning (ML) models with qualitative case study analysis. By evaluating subjects like Mehek (late-stage regression) and twins Tajun and Tayeeba (early-intervention profiles), we demonstrate how ML feature importance can mirror clinical behavioral milestones.

1. Introduction

The current diagnostic gap in ASD—often two years between concern and diagnosis—limits the window of high neuroplasticity. Following the scoping review by Kohli et al. (2022), this paper explores the effectiveness of automated screening using a dataset of 300 clinical subjects. We specifically examine the "Regression" phenotype and the impact of intervention timing on long-term severity.

2. Longitudinal Case Studies

To ground the ML data in clinical reality, three subjects were tracked:

2.1 Subject 1: Mehek (The Regression Profile)

Case Study 1: Mehek

Mehek was diagnosed with ASD at the age of 2.5 years. She initially showed normal devel-
opmental milestones. She developed early speech and displayed healthy infant behavior.
There was no family history of ASD.
6
However, after her second birthday, regression began. Previously learned words grad-
ually disappeared. She developed sleep disturbances, frequent night crying, and appetite
loss. She began preferring liquids and rejecting solid foods and fruits.
Communication shifted from verbal to gestural methods. For example, she would take
a plate to indicate hunger or physically drag caregivers toward desired objects.
Multiple medical consultations were conducted, including psychiatry, pediatrics, and
neurosurgery. She was diagnosed with mild ASD initially based on observed symptoms.
No modern diagnostic technology specific to ASD was used.
Over time, her symptoms intensified. She developed hyperactivity, self-harming be-
haviors such as head banging and biting, and complete speech loss. She is currently
10 years old and classified under severe ASD. She remains non-verbal and has limited
awareness of personal needs.

2.2 Subjects 2 & 3: Tajun & Tayeeba (The Early Intervention Profile)

Case Study 2: Tajun
Tajun, one of the twin sisters, showed early signs of ASD from infancy. She had no speech
development history. Unlike Mehek, there was no regression; speech was absent from the
beginning.
She was diagnosed at 1.5 years of age. Due to early detection, therapeutic interven-
tions began sooner. She is currently 7.5 years old and classified under moderate ASD.
She has limited speech ability, no history of self-harm, but continues to struggle with
diet regulation and communication.

Case Study 3: Tayeeba
Tayeeba, the twin sister of Tajun, was also diagnosed at 1.5 years. She displayed similar
early signs, primarily absence of speech.
Currently 7.5 years old, she is classified under mild ASD. She has developed partial
speech ability and demonstrates better adaptive behavior compared to her sister.
No self-harming behaviors have been observed. However, dietary challenges remain.

3.Discussion:

These cases suggest that regression-type ASD (normal early development followed by
loss of skills) may be more difficult to detect early. In contrast, absence of developmental
milestones may trigger earlier medical evaluation. The lack of advanced diagnostic tools
in Bangladesh during 2019 likely contributed to delayed severity staging in Mehek’s case.
The data highlights the necessity of AI-assisted screening systems and standardized early
assessment frameworks in developing healthcare systems. And here’s something impor-
tant: Mehek’s case is what researchers call “regressive autism.” It’s rare but documented.
5. Methodology




3.1 Data Preparation

The dataset used (300 subjects) contains 10 behavioral indicators (A1-A10) and demographic markers.

Preprocessing: Categorical data were converted via Label Encoding.

Model Selection: Four architectures were benchmarked: Random Forest, AdaBoost, SVM, and Logistic Regression.

Validation: 5-fold cross-validation was employed to ensure results were not overfitted.

4. Results

The following results were obtained from the benchmarking script:

5. Discussion

The convergence of ML data and case histories suggests that "Class" prediction is only the first step in a diagnostic framework. The case of Mehek reveals that regression-based ASD requires temporal tracking; a child who passes a screening at 12 months may still require monitoring at 24 months. However, the high F1-score of the Random Forest model suggests that behavioral questionnaires (A1-A10) are robust proxies for early risk stratification.

Furthermore, the comparative outcomes of the twins (Tajun and Tayeeba) suggest that while ML can predict the presence of ASD, the severity is influenced by the timing of intervention. Early intervention at 1.5 years allowed for adaptive behavior development that was absent in the delayed diagnosis of Mehek.

6. Conclusion

This study reinforces the role of technology in accelerating ASD screening. By linking automated results to real-world outcomes in twins and regressive cases, we provide a more holistic view of the spectrum than pure data analysis alone.