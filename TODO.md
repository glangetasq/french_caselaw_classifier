# TODO

## Critical fixes (do before any interview / sharing)

- [x] **Strip chamber names from document text before modeling**
  - Tested in `strip_chamber_test.ipynb` via `preprocessing.strip_chamber_names()`
  - Result: NB OOS 94.2→93.9%, OOD 49.0→48.6%; Word logit OOS 98.5→98.1%, OOD 55.1→53.8%
  - Slight drop across the board → models were not over-relying on chamber names; stripping reverted
  - Conclusion: vocabulary is genuinely discriminative, not a chamber-name shortcut

- [x] **Fix CamemBERT max_length: 128 → 256 or 512**
  - Rejected 512 due to O(n²) attention cost (16× slower); domain signal is front-loaded so extra context not justified
  - Tested head+tail asymmetric truncation (first 63 + last 63 tokens): OOS 95.8→94.4%, OOD 35.3→34.0% — tail is boilerplate ruling language shared across domains, adds noise; reverted to head-only 128
  - Head-only 128 is the right call: captures jurisdiction header + opening substantive paragraph where domain vocabulary concentrates

## High-value additions

- [ ] **Add error analysis (10–20 misclassified examples)**
  - Pull misclassified docs from the char_logit OOD confusion matrix (worst cells: civ↔com, crim)
  - For each: show predicted vs true label, document excerpt, and a one-line hypothesis for why it's hard
  - This single section demonstrates analytical depth more than any accuracy number

- [ ] **Add UMAP of CamemBERT embeddings colored by court type**
  - Needed to earn the "frozen CLS encodes cassation structure" claim currently made without evidence
  - Plot train (cour_de_cassation) vs OOD (tribunal_judiciaire, cour_d_appel) in the same embedding space
  - Expected: OOD points cluster separately → confirms template overfitting hypothesis

- [x] **Ablate char n-gram range**
  - Currently hardcoded at (3, 5); no justification shown
  - Try (2, 4), (3, 6) — report OOD F2 delta; even a small table strengthens the claim

## Nice to have

- [ ] **Fine-tune CamemBERT with classification head**
  - Listed as next step but not attempted; a senior ML candidate is expected to at least try it
  - Even a 2-epoch fine-tune on a subset would be meaningful

- [ ] **README: Limitations section**
  - Add one sentence flagging that OOS numbers may be inflated until leakage is confirmed/ruled out
  - Credibility > impressive numbers