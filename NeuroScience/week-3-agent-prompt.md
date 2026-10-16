# Prompt for creating the Week 3 neuroscience notes

Copy the prompt below into an AI coding agent with access to this repository, the attached `NS.docx` curriculum, a terminal, and the internet.

---

You are a computational neuroscience educator and scientific Python practitioner working in the `NeuroScience` Obsidian vault. Create the complete Week 3 learning module: five detailed daily working notes, a weekly synthesis report, executable experiments, real-data figures, atomic concept notes, and the necessary knowledge-map connections. Write the files and execute the experiments; an outline or a proposed plan does not satisfy this task.

The audience is a team of five learners completing a four-week foundation phase in AI + Neuroscience. The goal is strong fundamentals, not novel research or state-of-the-art performance. Be rigorous through correct definitions, transparent assumptions, reproducible evidence, and carefully explained mathematics. Do not turn this introductory week into an advanced machine-learning course.

## 1. Read the vault before writing

Read the applicable `AGENTS.md` files completely. Read these reference documents in full:

- `month 1/week 1/week-1-report.md`
- `month 1/week 2/week-2-report.md`
- All five files in `month 1/week 2/days/`, from `w2d1.md` through `w2d5.md`
- `Knowledge-Map.md`
- All existing `notes/concepts/*.md`
- `generate_week1_plots.py`, `generate_week2_plots.py`, `pyproject.toml`, and `uv.lock`
- The Week 3 curriculum and Week 4 handoff in the supplied `NS.docx`

Inspect the actual filesystem rather than assuming every prescribed file exists. At the time this prompt was prepared, Week 1 had a report and exports but no daily-note directory. Do not invent missing reference notes or create unrelated Week 1 material.

Continue the existing argument:

1. Week 1 established the difference between neural activity and measured neural data, the `[Channels × Time]` geometry, units, sensor constraints, frequency bands, and biological task formulations.
2. Week 2 established sampling, artifacts, filtering, timestamped windows, and feature extraction into `[Samples × Features]`. Its final laboratory produced one observation with five columns, `(1, 5)`, not five training examples. Labels must come from the experiment.
3. Week 3 now asks: **How does machine learning turn neural feature matrices into predictions, and how do we test whether those predictions generalize beyond artifacts, recordings, and subjects?**

Preserve the house style: direct technical prose, worked questions with full solutions, numbered sections, bolded rhetorical lead-ins, scientific callouts, explicit units, and Concept → Code → Output → Interpretation. Follow the weekly report's qualifications where older daily notes overstate a claim. In particular, finite filtered windows can have nonzero means; filtering does not certify artifact removal; Alpha/Beta bands are not universal mental-state labels; and high predictive accuracy does not establish a neural mechanism. Do not reproduce unsupported praise, fictional evaluator feedback, or exaggerated certainty.

## 2. Respect the supplied curriculum

Use this exact weekly report title:

**Machine Learning for Neuroscience: From Neural Data to Prediction**

The mandatory topics from `NS.docx` are:

- AI, machine learning, dataset, feature, label, training, testing, and Data → Model → Prediction.
- Classification: EEG → left hand/right hand.
- Detection: EEG → seizure/no seizure.
- Prediction: neural activity → a target variable; distinguish continuous regression targets from discrete classification labels.
- Decoding: brain activity → intended action or stimulus. Explain that decoding describes the scientific task and may use classification or regression.
- Exactly three principal learned algorithms: logistic regression, decision tree, support vector machine. Explain what each does, its intuition, and when it can be useful.
- Training/test data, predictions, accuracy, precision, recall, F1-score, and introductory overfitting.
- Subject-dependent learning, subject-independent learning, and generalization.

Use basic equations where they clarify the mechanism, defining every symbol and shape. Include the sigmoid and a linear score, a tree split and impurity intuition, the SVM margin, and metric formulas. Avoid lengthy optimization proofs, kernel derivations, FFT mathematics, wavelets, CSP, ICA implementation, deep learning, ensembles, domain adaptation, and broad hyperparameter searches. A dummy baseline is an evaluation control, not a fourth principal algorithm. Cross-validation belongs to the practical evaluation protocol; teach it simply.

## 3. Download and work with real EEG data

**Make real recorded EEG the main empirical laboratory.** Do not merely recommend downloading data, paste an unexecuted download example, or build all performance figures from simulated arrays. Download a manageable public subset, inspect it, construct labeled observations, train the three models, and generate plots from the actual recordings and held-out predictions.

The preferred dataset is the **PhysioNet EEG Motor Movement/Imagery Dataset, version 1.0.0**:

- Dataset and license: https://physionet.org/content/eegmmidb/1.0.0/
- Dataset DOI: https://doi.org/10.13026/C28G6P
- MNE download API and run definitions: https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html

Verify the current source documentation before implementation. This dataset supplies 64-channel EEG at 160 Hz in EDF+ files. For left-versus-right hand **imagery**, use runs **4, 8, and 12**. For those runs, `T1` denotes left-fist imagery and `T2` right-fist imagery; `T0` is rest. Other runs assign different meanings to the same codes. Never merge executed movement, hand/foot imagery, and left/right imagery under one assumed label mapping.

Start with one subject to validate loading, annotations, and features. Then use a fixed, predeclared cohort, preferably subjects 1–10, for the subject-generalization exercise. Download only the required runs rather than the whole archive. If resources require a smaller cohort, use at least five usable subjects when possible and state why. Freeze the cohort before looking at model scores; never choose subjects because they produce attractive results.

Use `mne.datasets.eegbci.load_data` or an equivalent documented download method, an explicit local cache path, and reproducible configuration. Avoid changing global MNE settings unnecessarily. Keep original downloaded files unchanged. Record the dataset version, source URLs, license, retrieval date, subject/run IDs, file sizes, and SHA-256 hashes; compare published checksums where available. Distinguish archival provenance from deterministic experiment outputs: timestamps and download progress should not contaminate repeatability checks.

Inspect and print actual sampling rates, channel names/types, reference information when available, duration, units, annotations, event counts, missing data, and exclusions. Do not pretend undocumented metadata is known. MNE represents EEG numerically in volts: convert deliberately to µV for voltage features and plots, and to µV² for powers. Verify consistent channel ordering. Never use annotation channels, subject IDs, run IDs, trial IDs, timestamps, or target codes as predictive feature columns.

Select a small fixed sensorimotor channel set, such as `C3`, `Cz`, and `C4`, if confirmed present after documented name standardization. Preserve channel identity in the features. Use fixed, event-aligned windows wholly inside the corresponding task interval; state onset offset, duration, endpoint convention, actual sample count, and excluded boundary trials. Prefer one observation per trial. If multiple windows are used, keep all windows from a trial together in every split.

Document task-appropriate filtering, referencing, and quality screening. Inspect waveforms and spectra before deciding whether a notch is needed; do not inherit 50 Hz automatically from synthetic Week 2 data. Process separate runs without silently bridging recording breaks. Avoid filtering across a temporal train/test boundary in a way that shares samples; use run-wise splits or a documented guard interval. Mark noncausal processing as an offline analysis.

Carry Week 2's mean, variance, peak-to-peak amplitude, Alpha power, and Beta power into the feature discussion. A modest channel-wise feature set is sufficient. State the PSD estimator, band boundaries, density units, integration convention, and window length if using Welch integration. Explain why this differs from variance of a realistic band-pass-filtered signal. If using log power, define a dimensionless ratio such as `log10(P / P_ref)` with a stated reference power and numerical floor. Keep every feature's definition and ordering fixed.

If a download fails, inspect the error, retry appropriately, and try another documented open labeled EEG dataset that supports multiple subjects. Record an alternative's task and annotation semantics accurately. Never substitute synthetic data silently or invent real-data outputs. If real data remains inaccessible, complete independent theoretical material, identify the blocked experiments, and leave affected completion statuses honest. Do not claim the full module is complete.

## 4. Build a defensible experiment

Create a row-aligned feature matrix `X` of shape `[Trials × Features]`, label vector `y` of shape `[Trials]`, and metadata table containing subject, run, trial, and event timing. Explain that thousands of voltage samples and multiple trials from one person do not constitute thousands of independent people.

Use the same observations, feature definitions, and held-out partitions for all three models:

- Logistic regression: training-only standardization inside a scikit-learn `Pipeline`, with a documented regularization choice and adequate convergence settings.
- Decision tree: document depth and minimum leaf size; compare controlled complexity with a deeper tree in a dedicated overfitting exercise.
- SVM: begin with a linear kernel and training-only standardization in a `Pipeline`. Define a decision score separately from a probability; do not present uncalibrated SVM scores as probabilities.
- `DummyClassifier`: a documented majority-class control evaluated on the same partitions.

Predeclare modest model settings rather than tuning against test results. If any setting, threshold, or feature is selected from data, select it using training/validation data only and respect the grouping at that level. Fit scaling, imputation, and any other learned transform only on each training partition. Predetermined per-trial features do not require fitting, but their design must not be revised to favor held-out results.

Run two clearly named protocols:

1. **Subject-dependent evaluation:** train and test on different runs from the same subject. With the three imagery runs, a simple leave-one-run-out evaluation is appropriate. Show which runs are held out and report per-subject outcomes. This evaluates run transfer within a person; do not call repeated runs separate sessions unless the metadata establishes that.
2. **Subject-independent evaluation:** hold out every recording of one subject while training on the other subjects. Use leave-one-subject-out evaluation on the fixed cohort, with no held-out-subject calibration. Save fold assignments and predictions. Explain the question this protocol answers and why shared-subject random splits answer a different one.

Use scikit-learn's documented grouped splitting mechanisms as appropriate: https://scikit-learn.org/stable/modules/cross_validation.html

Report accuracy, precision, recall, F1, class support, and confusion matrices with an explicit positive class. For binary examples, declare whether `RIGHT_HAND` is positive; for comparisons use macro-F1 as well and state zero-division handling. Report individual held-out-subject scores and an unweighted subject mean, separately from metrics on pooled predictions. Fold spread is descriptive variation, not automatically a confidence interval. Explain why the protocols have different training composition and sizes, so their score difference does not isolate one causal mechanism.

Include a small executed numerical example showing why accuracy can mislead under imbalance, using clearly labeled hypothetical labels rather than claiming the imagery dataset is a seizure dataset. Connect false negatives and false positives to seizure-detection consequences, but do not convert trial classification scores into clinical false alarms/hour or seizure-prediction claims.

Accept modest, near-baseline, or unexpected results. Never promise a particular accuracy, force subject-independent performance to be lower, or keep searching until the narrative looks impressive. Above-baseline predictions support discrimination under the tested protocol; they do not prove a causal neural source, unrestricted thought reading, or clinical readiness.

## 5. Five-day learning sequence

### W3D1 — From neural features to supervised datasets

Begin with 4–8 Week 2 diagnostic questions and complete solutions covering observation rows, feature units, label provenance, filtering limits, temporal compression, and repeated windows. Define AI versus ML, `X`, `y`, dataset, observation, feature, label, training, and testing. Download and inspect the real EEG subset; show event alignment and build the first labeled feature rows. Explain classification, detection, continuous prediction, and decoding using concrete neuroscience examples.

### W3D2 — Logistic regression and the first held-out predictions

Explain weighted feature scores, sigmoid probabilities, the classification threshold, and the conceptual role of the loss and regularization. Implement a training-only preprocessing pipeline and run-held-out prediction for a fixed subject. Compare with the dummy baseline. Show actual features, a two-feature teaching boundary, and errors. Explain why a model's estimated probability is not a biological certainty or automatically calibrated confidence.

### W3D3 — Decision trees and support vector machines

Explain threshold rules and tree depth, then the SVM margin and support vectors. Train both on the same features and partitions used for logistic regression. Compare assumptions, interpretability, scaling needs, and failure modes. Use matched two-feature visualizations for intuition, clearly separating these reduced-feature models from the main feature-set benchmark. Execute a small training/validation tree-depth comparison without selecting depth on a test set.

### W3D4 — Training, testing, metrics, and overfitting

Define TP, TN, FP, FN and derive accuracy, precision, recall, and F1 from a worked confusion matrix. Verify manual calculations against library outputs. Analyze the real held-out predictions, class counts, majority baseline, and per-class errors. Include the hypothetical imbalance example. Explain training/validation/test roles and investigate leakage from shared trials, neighboring windows, globally fitted scaling, and subject identity.

### W3D5 — Subject generalization and complete Week 3 synthesis

Execute the fixed-cohort subject-independent protocol and consolidate the subject-dependent results. Show subject distributions, per-subject scores, protocol comparisons, and concrete misclassified trials. Discuss biological variation, montage/reference differences, recording conditions, artifacts, and uncertainty without asserting an untested cause. Present the complete reproducible pipeline and a comprehensive Week 3 mastery test with conceptual, numerical, code-reading, plot-reading, and methods-critique questions, all fully solved. Hand forward the question of how Week 4 neural networks learn representations while retaining these evaluation obligations.

Every daily note must contain valid YAML frontmatter, reciprocal navigation, overview/goals, 4–8 prerequisite checkpoints with answers, detailed core exposition, at least one fully runnable laboratory with actual output, relevant figures, mastery questions with worked solutions, references, and a summary/bridge. Match or exceed the reference notes' substantive depth through explanations and evidence, not repetition or an arbitrary word count.

## 6. Figures and empirical storytelling

Generate all Week 3 PNG figures through `generate_week3_plots.py`. Use Seaborn over Matplotlib, the repository's required theme, 150 dpi or higher, tight saving, readable panels, and appropriate physical or mathematical units. Inspect every generated image visually and fix clipping, tiny labels, or misleading comparisons.

Produce approximately 10–12 purposeful figures, combining panels where useful:

1. Real event-aligned multichannel EEG with subject/run/trial provenance and task interval.
2. Real raw/preprocessed traces and PSDs, with residual contamination qualified.
3. The real trial-to-feature-matrix transformation, with shape and column definitions.
4. Class counts by subject/run and retained/rejected trial counts.
5. Two actual feature distributions by imagery label, showing overlap.
6. Matched two-feature boundaries for logistic regression, tree, and linear SVM, with held-out points distinguished.
7. Held-out confusion matrices for the principal models, explicitly labeling axis orientation and protocol.
8. A metric comparison against the dummy baseline on identical folds.
9. Tree complexity versus training/validation performance, with selection kept separate from test evaluation.
10. Subject-dependent versus subject-independent partitions, showing the grouping rule.
11. Actual per-subject performance for both protocols, plus the stated aggregate summaries.
12. Representative correctly and incorrectly classified held-out trials, selected by a deterministic documented rule.

Most figures must derive from downloaded EEG, its features, or actual predictions. Small synthetic diagrams or hypothetical metric examples are acceptable when explicitly labeled. A two-dimensional decision boundary demonstrates a two-feature model; it is not a projection of the full model unless implemented and explained correctly. Do not draw interpolated EEG heatmaps as anatomical source maps. Do not cherry-pick unusually clean trials or unusually favorable subjects.

Caption every image immediately with a sequential `**Figure N.**` paragraph, numbered independently within each document. State what is shown, the provenance/protocol where relevant, the inference supported, and the limitation. Reports use `assets/<name>.png`; daily notes use `../assets/<name>.png`.

## 7. Files and knowledge-graph integration

Create:

- `month 1/week 3/week-3-report.md`
- `month 1/week 3/week3-export.md` and its identical mirror `week-3-export.md`
- `month 1/week 3/days/w3d1.md` through `w3d5.md`
- `month 1/week 3/assets/*.png`
- `generate_week3_plots.py`
- Supporting Week 3 download/experiment code if separation improves clarity, with one figure-generation entry point.
- A documented local data cache, provenance manifest, feature table/schema, split assignments, held-out predictions, metrics, and captured laboratory outputs. Keep large raw downloads out of ordinary Git commits and out of the Obsidian note graph; preserve a reproducible acquisition route.
- Useful atomic concept notes, such as `supervised-learning`, `feature-matrix`, `experimental-labels`, `logistic-regression`, `decision-tree`, `support-vector-machine`, `classification-metrics`, `overfitting`, `data-leakage`, and `subject-generalization`. Create only concepts actually used; link related existing concepts.

Preserve the actual imported Week 3 curriculum text in both exports; do not replace the raw brief with a polished rewrite. Record source provenance accurately. Do not invent a `GI.docx` source that was not provided.

The weekly report must contain a 4–6 sentence abstract, keywords, the inherited Week 2 question and five-day linked syllabus, fundamental concepts, the three algorithms, training/evaluation, subject generalization, end-to-end implementation, comparison/discussion, applications, limitations, conclusion, annotated references, Appendix A reproducibility, and Appendix B graph connections. Number its main sections sequentially without gaps. Synthesize the daily laboratories into one argument rather than pasting five disconnected notes together.

Quote YAML wikilinks and use normal YAML title quoting without literal backslash-escaped quote characters. Use `ns/week-03` tags, and status `complete` only after verification. Escape alias pipes in tables: `[[w3d1\|**W3D1**]]`. Use actual creation dates.

Connect W3D1 backward to `w2d5`; link subsequent days reciprocally. W3D5 may lead to the Week 3 report and a plain-text Week 4 handoff. Link the Week 3 report backward to Week 2. Update Week 2's forward report navigation and Appendix B now that Week 3 exists. Update `Knowledge-Map.md` with the report, all five days, the Learning taxonomy, and the extended governing pipeline. Add genuine Week 3 backlinks to reused concept notes.

Do not create links to nonexistent Week 4 reports or daily files. Follow the current hub's convention of describing planned weeks in plain text until their files exist. Do not create empty placeholder reports merely to pass a link check. Preserve unrelated existing work and keep prior-content edits limited to necessary navigation and backlinks.

## 8. Environment and reproducibility

Use `uv` for every Python execution and dependency change. Never use system Python, `pip`, or a second unmanaged environment. Add `mne` and `scikit-learn` through `uv add`; add other dependencies only when needed. Inspect compatibility before changing existing versions: the current project deliberately constrains SciPy to `>=1.14,<1.15` because a later wheel previously failed on this machine. Do not casually remove that constraint. Update the lockfile and report the exact resolved versions.

Use deterministic seed 42 for random generation, shuffled splits, and stochastic estimators; sort subject/run/trial ordering and freeze the configuration. Separate first-time network acquisition from cached reproducible analysis. Make the scripts work from documented project-root commands and resolve output paths relative to the script rather than a personal absolute directory.

Every published Python block must run as written with explicit imports and documented inputs. No ellipses, undefined variables, missing helper functions, notebook-state dependencies, or invented output. When a block consumes saved Week 3 data, specify the file and preceding acquisition command. Organize code so documentation and executed sources remain consistent, following Week 2's source-block approach where useful. Capture the exact console output, including real warnings if emitted; address substantive warnings rather than hiding them.

After a first successful run, rerun the cached experiment and figure pipeline and compare PNG hashes, stable tables, and laboratory stdout. Require byte-identical figures and console outputs in the same locked environment. Keep volatile timestamps, progress bars, and timing measurements outside deterministic outputs. Explain the boundary of reproducibility across different platforms or library versions.

## 9. Literature and scientific interpretation

Verify references online through primary papers, publishers, dataset documentation, and official library documentation. Include full authors, year, title, venue, DOI where one exists, and a one-sentence contribution tied to the notes. Do not invent a DOI for software documentation or textbooks lacking one. Cite the real dataset and its acquisition publication, relevant foundational ML literature, and evidence supporting neuroscience-specific evaluation cautions. Distinguish official software documentation from peer-reviewed scientific evidence.

Use the vault's application field order: **Biological Event → Neural Activity → Neural Data → Target / Label → AI Task → Primary Failure Mode**. Cover motor imagery, seizure detection, and continuous intended-action/stimulus decoding conceptually; distinguish implemented imagery results from unimplemented applications.

Explicitly dismantle these misconceptions with explanations and evidence:

- A feature matrix labels itself.
- Training accuracy estimates unseen-data performance.
- High accuracy always means a good detector.
- A random trial/window split proves unseen-subject generalization.
- Standardizing the whole dataset before splitting is harmless.
- Good subject-dependent performance guarantees good subject-independent performance.
- A decision boundary or feature coefficient proves a causal brain mechanism.
- More data rows automatically mean more independent participants.
- A classifier's output establishes clinical safety or reliable thought reading.

## 10. Completion gate

Before declaring completion, execute and document checks for:

- All canonical reports, days, mirrored exports, scripts, assets, and result/provenance files exist.
- Raw data were actually downloaded and inspected; each result is traceable to the frozen configuration and cohort.
- Event-label mapping is correct for the selected runs; windows do not cross task boundaries.
- Shapes, units, feature schema, row/label/metadata alignment, finite values, and exclusions are verified.
- No annotation or identity fields entered `X`; no trial crosses a partition; no subject crosses a subject-independent partition; learned preprocessing is fitted only on training data.
- Model comparison uses identical partitions; test data never guide model selection; baselines and per-class support are included.
- Figures and tables match saved predictions and metrics; repeated cached runs pass determinism checks.
- Every published Python block executes via `uv run python`; displayed outputs match those executions.
- Frontmatter parses; headings and figure captions are sequential; all image references resolve; syllabus table links use escaped pipes.
- New and modified notes have no dangling wikilinks or orphaned concept nodes. Audit the full vault and distinguish any pre-existing defects from newly introduced ones rather than silently claiming a clean graph.
- References and dataset/software provenance are verified; every image has been visually inspected.

Deliver the complete files. In the final response, briefly list the principal paths, dataset subset, actual evaluation protocols/results, reproducibility commands, and any unresolved limitation. Do not say “complete” if real-data experiments, runnable laboratories, or required verification remain unfinished.
