# Topic labels for the r/AskDocs real-vs-synthetic analysis

Generated 2026-10-11 from `topic_labels.csv`. 6,600 posts.

Join to the analysis dataset on `submission_id`. Same corpus and same seed (42) as the generation run, so the ids line up exactly.

## Columns

| column | meaning |
|---|---|
| `submission_id` | Reddit post id. Join key. |
| `discipline` | **The moderator variable.** The model's label, for every row, one of the 17 disciplines. |
| `cancer_relevant` | Model's flag: does the post involve a malignancy. |
| `cancer_stance` | current_patient / survivor / not_applicable / unclear. Collected but DEFERRED from the analysis -- see below. |
| `model_confidence` | Model's self-reported confidence. Near-useless: it is 1.0 almost everywhere. Do not filter on it. |
| `evidence_quote_verified` | Whether the model's supporting quote was found verbatim in the post. A weak per-row quality flag. |
| `in_validation_sample` | TRUE for the 200 posts in the reference sample. |
| `discipline_gold` | Reference label where it exists, else blank. |
| `cancer_relevant_gold` | Reference cancer flag where it exists. |
| `discipline_contested` | TRUE where the adjudicators recorded the post as genuinely ambiguous. |
| `label_model` | Model id that produced the label. |

## Use `discipline` for all rows, not a mix

`discipline` is the production model's label even on the 200 posts in the reference sample. Substituting the reference label there would make 3% of the corpus measured differently from the rest, and differential measurement error in a moderator distorts interaction estimates unevenly across topics. The reference labels are in `discipline_gold` for comparison, attenuation estimates, or a deliberate override.

## Measurement properties you need

**Read this before treating `discipline_gold` as ground truth.** The 200-post reference sample was labelled by two LLM annotators, not by human coders. A human adjudicated only the posts where the two disagreed. So the reference standard is mostly inter-model consensus, and the production labels come from a third model. That is a consistency check, not an independent validation: the whole chain is LLM judgement with human arbitration at the margin. It was a deliberate trade for time and should be reported as such.

- **How much human judgement is in the reference standard:** 42 of 200 labels were set by a human adjudicating a disagreement (21%). The other 158 are unadjudicated agreement between the two LLM annotators.
- **Inter-annotator agreement** between the two LLM annotators: raw agreement 0.795, Cohen's kappa 0.779 (95% CI 0.718-0.839). Treat it as a reference point for the production model rather than as a human ceiling.
- **Production model agreement** with the adjudicated reference: 0.770 on 200 posts.
- **So the moderator carries real misclassification.** Roughly one label in four disagrees with the reference standard. Misclassification in a moderator attenuates interaction estimates toward zero, so a null topic effect is weaker evidence of no effect than it would be with a clean moderator. Worth either an attenuation correction or an explicit caveat. Note that the reference standard itself carries unmeasured error, so this is a floor on the true misclassification rate, not an estimate of it.
- **`Nonmedical` and `Medical; other` are residual categories**, not topics. They collect posts that fit nothing else. Treating them as substantive levels will produce uninterpretable coefficients; consider dropping or pooling them.
- **`Neoplasms` follows the ICD-10 chapter**, so it covers benign neoplasms as well as malignant, and a malignancy counts only once a clinician or test has raised it. It is not the same variable as `cancer_relevant`.
- **`cancer_stance` is deferred.** current_patient vs survivor was dropped from the analysis as underpowered; the labels are collected so it can be revisited post hoc.

## Realized prevalence and what is testable

Minimum detectable moderation at 80% power, from the pilot-2 Explorations proportions (real 0.191, synthetic 0.073, gap 11.8 pp). A category whose figure exceeds ~5.9 pp can only detect a moderation more than half the size of the main effect.

| discipline | n | share | min. detectable |
|---|---|---|---|
| Dermatology | 715 | 10.8% | 4.3 pp |
| Gastroenterology | 711 | 10.8% | 4.3 pp |
| Genitourinary and reproductive health | 695 | 10.5% | 4.3 pp |
| Psychiatry and mental health | 596 | 9.0% | 4.6 pp |
| Infectious diseases | 572 | 8.7% | 4.7 pp |
| Neurology | 525 | 8.0% | 4.9 pp |
| Cardiovascular | 498 | 7.5% | 5.0 pp |
| Endocrinology, nutrition, and metabolism | 439 | 6.7% | 5.3 pp |
| Musculoskeletal and connective tissue conditions | 362 | 5.5% | 5.8 pp |
| Medical; other | 361 | 5.5% | 5.8 pp |
| Hematology and immunology | 326 | 4.9% | 6.1 pp |
| Otolaryngology | 281 | 4.3% | 6.6 pp |
| Pulmonology | 174 | 2.6% | 8.3 pp |
| Neoplasms | 173 | 2.6% | 8.3 pp |
| Ophthalmology | 69 | 1.0% | 13.1 pp |
| Dentistry | 64 | 1.0% | 13.6 pp |
| Nonmedical | 39 | 0.6% | 17.4 pp |

## Codebook rules

Applied to both the model prompt and the human adjudication, so the two are held to one rubric.

1. NEOPLASMS. ICD-10 Chapter II (C00-D49) covers benign neoplasms as well as malignant ones, so both are coded "Neoplasms", whichever organ is involved. A malignancy counts when it has been diagnosed, or when a clinician or a test has actually raised it (a biopsy, an imaging finding, an abnormal result, a referral); a poster's own worry about cancer is NOT enough, so a question about whether something might be cancer, about screening, or about family-history risk is coded to the organ system. A benign neoplasm counts when it is named as one: a benign tumour, lipoma, fibroid, adenoma or haemangioma. Lesions that are not neoplasms stay with the organ system - a cyst (ovarian cyst is N83, not Chapter II), an undifferentiated lump, a swollen lymph node, or a polyp not described as adenomatous (an unspecified colon polyp is K63.5). Code the organ system also when the post's primary concern is a non-neoplastic problem that happens to co-occur with a neoplasm or a cancer history.

2. CARE-PROCESS QUESTIONS. When the question is about the process of care -- whether it is rude to ask a doctor something, a delayed portal reply, how to communicate with a clinician, whether a discharge was appropriate -- but the post describes a specific complaint, code the complaint's discipline. "Nonmedical" is only for a post with no medical content at all.

3. EXPOSURES AND POISONING. Chemical, drug and environmental exposures -- a caustic ingestion, carbon monoxide, a medication overdose, swallowing a contaminant -- are coded "Medical; other". ICD-10 places poisoning and external causes in their own chapters (XIX and XX), outside the body-system chapters this taxonomy is built from, and the taxonomy has no toxicology category.

Taxonomy source: Understanding Health-Related Discussions on Reddit: Development of a Topic Assignment Method and Exploratory Analysis, JMIR Formative Research 2025;9:e55309, doi:10.2196/55309 (Table 2). Only the 17 discipline names are taken from that paper; the labelling method is ours.
