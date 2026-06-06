# TODO

## Critical fixes (do before any interview / sharing)

- [ ] **Strip chamber names from document text before modeling**
  - Chamber names (`Première chambre civile`, `Chambre commerciale`) appear in decision bodies and leak directly into TF-IDF features (`civile`, `commerciale`, `criminelle`, `sociale` dominate top features)
  - Strip or mask these header lines in `preprocessing.py`, then re-run all models
  - If OOS drops significantly → leakage confirmed; if stable → vocabulary is genuinely discriminative
  - Either outcome is an interesting result; not doing it is indefensible in an interview walk-through

- [ ] **Fix CamemBERT max_length: 128 → 256 or 512**
  - Average document is ~490 words; at max_length=128 (~96 words) BERT sees only ~20% of each document
  - This alone likely explains why frozen BERT underperforms char TF-IDF on OOD
  - The "CLS encodes cassation structure" hypothesis cannot be evaluated until the truncation is fixed
  - CamemBERT's context window supports 512 tokens; use it

## High-value additions

- [ ] **Add error analysis (10–20 misclassified examples)**
  - Pull misclassified docs from the char_logit OOD confusion matrix (worst cells: civ↔com, crim)
  - For each: show predicted vs true label, document excerpt, and a one-line hypothesis for why it's hard
  - This single section demonstrates analytical depth more than any accuracy number

- [ ] **Add UMAP of CamemBERT embeddings colored by court type**
  - Needed to earn the "frozen CLS encodes cassation structure" claim currently made without evidence
  - Plot train (cour_de_cassation) vs OOD (tribunal_judiciaire, cour_d_appel) in the same embedding space
  - Expected: OOD points cluster separately → confirms template overfitting hypothesis

- [ ] **Ablate char n-gram range**
  - Currently hardcoded at (3, 5); no justification shown
  - Try (2, 4), (3, 6) — report OOD F2 delta; even a small table strengthens the claim

## Nice to have

- [ ] **Fine-tune CamemBERT with classification head**
  - Listed as next step but not attempted; a senior ML candidate is expected to at least try it
  - Even a 2-epoch fine-tune on a subset would be meaningful

- [ ] **Split OOD results by court type**
  - Currently `tribunal_judiciaire` and `cour_d_appel` are pooled together
  - Separating them may reveal which court type is harder to generalize to and why

- [ ] **README: acknowledge chamber-name leakage risk**
  - Add one sentence flagging that OOS numbers may be inflated until leakage is confirmed/ruled out
  - Credibility > impressive numbers

## Resume framing (not code — reminder)

Do NOT frame this as "99% accuracy classifier." The correct claim:
> Designed OOD evaluation framework for cross-court generalization on 535k French legal decisions; diagnosed template overfitting in frozen CamemBERT embeddings (−60pp OOD gap vs −39pp for char TF-IDF); char n-gram model selected as production baseline.
