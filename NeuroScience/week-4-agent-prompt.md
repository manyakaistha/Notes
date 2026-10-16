# Prompt for creating the Week 4 neuroscience notes

Copy the prompt below into an AI coding agent with repository access, the supplied `NS.docx`, a terminal, and internet access.

---

You are a computational neuroscience educator and scientific Python practitioner working in the `NeuroScience` Obsidian vault. Create the complete Week 4 foundation module: five detailed daily working notes, the weekly academic synthesis report, executed teaching laboratories, real EEG plots, atomic concept notes, reproducible scripts, and knowledge-map integration. Write the actual files and execute their code. Delivering only a plan or an outline does not satisfy the task.

The audience is a team of five learners finishing a four-week introduction to AI + Neuroscience. Teach deep-learning fundamentals with clear intuition, precise definitions, explicit shapes, worked examples, and honest empirical evidence. This is not a state-of-the-art benchmarking project. Maintain technical depth without requiring advanced derivations or training a large model for every architecture discussed.

## 1. Read the reference material and continue its argument

Read all applicable `AGENTS.md` instructions completely. Before authoring, read the Week 1 and Week 2 reports in full, all five Week 2 daily notes, the entire Week 3 report and all five Week 3 daily notes, and the Week 4 portion of `NS.docx`. Also inspect:

- `Knowledge-Map.md` and existing atomic concepts.
- `generate_week1_plots.py`, `generate_week2_plots.py`, and `generate_week3_plots.py`.
- `pyproject.toml` and `uv.lock`.
- `data/provenance_manifest.json`, `data/feature_schema.json`, `data/week3_features.csv`, `data/week3_predictions.csv`, `data/week3_metrics.json`, and `data/week3_lab_outputs.json`, if present.
- The actual raw EDF cache and any saved Week 3 split definitions.

Inspect what exists instead of inventing reference files or relying only on the earlier agent prompt. Week 3 currently analyzes subjects S001–S010, runs 04/08/12, and sensorimotor channels `C3`, `Cz`, `C4`. It uses approximately 0.5–3.5 s post-cue windows, 480 samples at 160 Hz, and 15 handcrafted features per trial. Verify these details in code and data before reuse; validate task-interval containment and the half-open sample convention rather than assuming the prose is exact.

Continue the three-week argument:

1. Week 1: neural activity is biological reality; neural data is an indirect sensor measurement represented by numerical tensors.
2. Week 2: preprocessing and feature extraction preserve some information while suppressing or discarding other information.
3. Week 3: classifiers map features to experimental labels, but held-out evaluation, artifacts, overfitting, and subject generalization govern interpretation.
4. Week 4: **How do neural networks learn representations from neural data, and how do CNNs, recurrent models, and transformers use its temporal and channel structure?**

**Resolve the curriculum mismatch explicitly.** Existing Week 3 handoffs and the hub describe Week 4 as “Unsupervised Learning and Neural Networks” and propose PCA, t-SNE, and CSP. The supplied `NS.docx` instead defines Week 4 as deep-learning fundamentals, CNNs, RNN/LSTM, transformers, and neuroscience applications. Follow `NS.docx` for this week's scope. Supervised representation learning is not inherently unsupervised; CSP is commonly supervised. Do not build a PCA/t-SNE/CSP module or omit the requested sequential-model and transformer introductions. When integrating Week 4, make limited corrections to the Week 3 handoff and hub so they accurately point to the completed module.

Preserve the reports' style: short technically precise paragraphs, numbered headings, bolded explanatory lead-ins, scientific callouts, fully worked checkpoint answers, explicit units, and Concept → Code → Output → Interpretation. Preserve scientific qualifications rather than older overstatements. A cue label does not prove the intended neural process occurred; an accuracy above a baseline does not prove causal neural decoding; a post-cue delay does not guarantee sensory activity is absent; filtering does not remove all in-band artifacts. Do not copy fictional reviewer praise or categorical claims that a particular architecture must solve subject transfer.

## 2. Follow the exact Week 4 syllabus

Use this weekly report title:

**Deep Learning for Neuroscience: Neural Networks, CNNs, Sequential Models and Applications**

Cover the complete curriculum:

- **Neural networks:** artificial neuron, weight, bias, layer, activation function, forward propagation, loss, and conceptual backpropagation.
- **CNNs:** convolution, kernel, feature map, pooling, classification, and why local patterns can be useful in neural signals. Explain EEG → CNN → Features → Prediction.
- **RNNs and LSTMs:** sequential data, why EEG is sequential, recurrent state, what an RNN is, and why LSTMs were introduced. No recurrent-gate derivations are required.
- **Transformers:** attention, self-attention, and the idea of learning relationships between different sequence positions. No detailed query/key/value mathematics is required.
- **Applications:** brain-computer interfaces, motor imagery, seizure detection, sleep classification, emotion recognition, neural decoding, and neuroimaging analysis. Explain exactly what is being predicted, where the target comes from, and what can undermine the interpretation.

Keep the main text conceptual. Use the basic affine neuron `z = wᵀx + b`, an activation, a loss example, and a small weight-update demonstration where they help. Define symbols, dimensions, and units. Do not require full backpropagation proofs, LSTM gate equations, attention algebra, positional-encoding derivations, advanced architectures, self-supervised pretraining, or exhaustive optimization searches. A compact executed library example can demonstrate a mechanism without teaching its full derivation.

Distinguish an artificial neuron from a biological neuron. Explain that “deep” concerns learned layer composition, not evidence of biological realism, consciousness, or superior performance. Learned features do not remove the need for appropriate labels, preprocessing, scientific reasoning, and held-out testing.

## 3. Use real EEG as the empirical backbone

Reuse the verified Week 3 PhysioNet cache when it exists. Download any missing recordings through a reproducible script and work on the actual files; do not merely recommend downloading data. Prefer continuing the same task and cohort so learners see the representation change while the scientific target stays fixed.

Primary sources:

- Dataset and current terms: https://physionet.org/content/eegmmidb/1.0.0/
- Dataset DOI: https://doi.org/10.13026/C28G6P
- MNE run definitions and download API: https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html

Verify the source documentation before implementation. For left/right hand motor imagery, use runs 4, 8, and 12; in those runs `T1` is left-hand imagery, `T2` right-hand imagery, and `T0` rest. Do not confuse them with executed movement or hands-versus-feet imagery. Use a fixed cohort chosen before examining Week 4 scores. Do not select “responsive” subjects using Week 3 performance to make deep learning appear effective.

Keep original EDF files unchanged. Verify checksums, sampling rates, channel types/names, units, annotation semantics, and usable task intervals. Preserve source URLs, license, retrieval information, subject/run/trial identifiers, exclusion reasons, and hashes in a Week 4 provenance/configuration manifest. Check the original dataset's license rather than copying the existing manifest blindly.

MNE's numerical voltages are in V. Convert deliberately to µV for plots and any voltage features; power features are in µV². Model inputs may be standardized and dimensionless: distinguish those from physical-voltage traces. Record any reference choice and preprocessing. Process recording runs separately, respect discontinuities and filter boundaries, and identify zero-phase filtering as offline. Do not inherit a 50 Hz notch without examining the spectrum.

Construct and save aligned representations for the same retained trials:

- Handcrafted features: `[Trials, 15]`, or the actual verified Week 3 schema.
- EEG epochs: `[Trials, Channels, Time]`, expected `[N, 3, 480]` if the verified Week 3 design is retained.
- Labels: `[Trials]`, encoded consistently, for example `LEFT_HAND = 0`, `RIGHT_HAND = 1`.
- Metadata: subject, run, trial/event identity, and time bounds; use these for grouping and provenance, never as input features.

State precisely whether the network consumes unfiltered waveforms, preprocessed waveforms, or handcrafted features. “End-to-end” must not imply that acquisition and quality-control decisions disappeared. Avoid creating a second neural dataset solely to produce more figures. Other applications can be explained with cited task diagrams rather than fake empirical experiments.

## 4. Keep the computation small and educational

Use PyTorch for the neural demonstrations, adding it through `uv` only if absent. Use CPU execution by default to make these small experiments practical and easier to reproduce. Do not require a GPU, pretrained weights, cloud training, or paid resources.

Implement these mandatory experiments:

1. **Artificial neuron and one training step:** an executed numerical example showing input, weight, bias, activation, prediction, loss, gradient, and updated weight. Use real feature values where practical; clearly label any constructed teaching input. Explain why a gradient indicates a direction for local loss reduction and why a step is not guaranteed to improve all unseen predictions.
2. **Compact MLP on real features:** a small network with one hidden layer, a clear `[Batch, Features]` input, an activation, and binary or two-class output. Contrast learned nonlinear combinations with Week 3's logistic regression. Plot real training and validation loss and explain model capacity.
3. **Compact 1D CNN on real epochs:** a short temporal convolution stack, activation, pooling, and classification head. Show every intermediate shape and total trainable parameter count. Train and evaluate it on actual labels using a frozen protocol; generate real learned-kernel, feature-map, and held-out-result figures.
4. **RNN/LSTM mechanism demonstration:** pass the same real EEG sequence through small documented recurrent layers; print input, state, and output shapes and plot state evolution. A brief clearly labeled synthetic delayed-memory demonstration is optional, but it must not be represented as real EEG evidence. A full recurrent-model benchmark is not required.
5. **Self-attention mechanism demonstration:** divide a real epoch into a modest number of fixed time patches/tokens, document their duration and feature definition, and execute a small library attention example. Visualize relationships among tokens. State whether the weights are untrained, trained on a teaching objective, or trained on the imagery task. An untrained attention heatmap is a computational illustration, not evidence of physiological relevance. Do not train a full transformer merely to satisfy this introductory topic.

For the CNN, distinguish temporal kernel length in samples from duration in ms. With `Conv1d` input `[Batch, Channels, Time]`, explain whether filters combine recording channels and time jointly or use grouped/depthwise structure. A learned feature-map index is not an electrode channel or anatomical source. A scalp channel list is not a regular pixel grid; do not apply a 2D image analogy as though electrode spacing were uniform.

For RNN/LSTM examples, transpose explicitly to `[Batch, Time, Channels]` when using `batch_first=True`. Explain that ordering is necessary but sample rate and timestamps determine physical timing. Describe LSTM gates verbally as mechanisms controlling state updates and retention; they reduce some optimization difficulties without guaranteeing indefinite memory.

For attention, show token indices alongside their actual EEG time intervals. If sequence position is supplied, explain its purpose conceptually. Explain that attention alone does not inherently know temporal order. State quadratic token-pair memory growth simply; do not form a needlessly large sample-by-sample attention matrix. Distinguish bidirectional/offline context from causal streaming access. Attention weights are neither causal explanations nor confidence scores.

Use one shared configuration for cohort, inputs, random seed, splits, learning rate, batch size, epoch budget, and any regularization. Record the settings before final test evaluation. Prefer fixed small epoch budgets; if selecting a checkpoint, use validation loss only. Reset models, optimizers, and preprocessing independently for each fold. Save the model selected by the declared rule, its predictions, and its training history.

Use losses correctly. With two-class logits `[Batch, 2]`, use `CrossEntropyLoss` and integer class targets `[Batch]`; do not apply softmax before that loss. With one logit per observation, use `BCEWithLogitsLoss` with matching floating targets and apply sigmoid only when producing probabilities. Explain which convention the laboratories use. Demonstrate `train()`, `eval()`, gradient clearing, `backward()`, parameter updates, and inference without gradient tracking. Do not silently update weights or normalization statistics on test data.

## 5. Carry Week 3's evaluation discipline forward

The architecture changes; the grouping obligations remain. Derive explicit train/validation/test assignments and save them before training. Keep all observations from a trial together and respect run/subject boundaries.

Mandatory evaluation:

- A simple run-held-out subject-dependent example with a predeclared subject, not the best Week 3 subject. For example, use Subject 1 Run 4 for training, Run 8 for validation, and Run 12 for final testing. State its small sample size and limits.
- A small subject-independent comparison of the compact CNN, a Week 3 classical baseline, and a dummy baseline on the same retained trial IDs and outer test subjects. Prefer leave-one-subject-out on the existing fixed cohort if the frozen CPU budget permits. With fixed epochs, no inner validation is necessary for checkpoint selection; if early stopping/tuning is used, validation subjects must come only from the outer training subjects. A predeclared train/validation/test subject holdout is acceptable for resource limits, provided its scope is explicit and results are not described as full LOSO.

Evaluate the MLP in the teaching protocol; do not require all architectures under every protocol. Keep the main empirical comparison focused on the compact CNN and classical/dummy controls. Recompute baselines on identical test trials if the Week 4 protocol or retention changed. Existing Week 3 results can only be reused when model inputs, training sets, and test observations truly match.

Fit global/channel-wise normalization statistics on the training partition only, then freeze them for validation/test. Document the aggregation axes and handle constant channels safely. If using predetermined per-epoch centering/scaling, explain that it is a different inference-time transformation that can remove amplitude information; do not present it as equivalent to training-set standardization. Apply any augmentation only to training observations and explain its biological assumption. More generally, avoid augmentation unless it is necessary for this small teaching experiment.

Report accuracy, precision, recall, binary F1, macro-F1, class support, and confusion matrices, defining positive class and zero-division handling. Save trial-level predictions and distinguish per-subject averages from pooled metrics. With multiple outer folds, report descriptive subject variation; do not call standard deviation a confidence interval. Show the relevant dummy baseline rather than universally labeling 50% “chance.”

Do not choose epochs, thresholds, architectures, subjects, or preprocessing based on test scores. Training loss curves do not establish held-out generalization. Do not claim a CNN fixes subject variability, that a deeper model must outperform handcrafted features, or that one small experiment proves deep learning is better or worse overall. Capacity, input representation, data quantity, optimization, and split design all influence performance. Accept and explain near-baseline or unfavorable results without searching for a more flattering experiment.

## 6. Five-day sequence

### W4D1 — Artificial neurons and learning from error

Begin with 4–8 Week 3 diagnostic questions and complete solutions about feature/label shapes, training versus testing, overfitting, classification metrics, subject grouping, and label provenance. Teach neuron, weight, bias, layer, activation, forward pass, loss, and conceptual backpropagation. Execute the artificial-neuron update and a compact MLP on real features. Show input/output shapes and train/validation histories. Explain that artificial neurons are mathematical functions rather than faithful biological cell models.

### W4D2 — CNNs and learned local EEG features

Carry forward the MLP's parameter-learning mechanism. Teach convolution, kernel, feature map, pooling, classification, receptive field intuition, and temporal units. Trace a real EEG trial through a small 1D CNN, then execute its declared training/evaluation protocol. Plot kernels and feature maps with clear provenance. Connect local waveform learning to Week 2's information-compression trade-off and Week 3's evaluation controls.

### W4D3 — Sequential data, RNNs, and LSTMs

Explain sample order, sequential dependence, recurrent state, and why repeated state updates can make long-range learning difficult. Introduce LSTM retention/update control verbally, with an accessible memory analogy that states its limitations. Execute RNN and LSTM passes on real EEG, print exact tensor/state shapes, and visualize state progression. Clearly distinguish a forward-pass illustration from a trained classifier. No gate equations or required recurrent benchmark.

### W4D4 — Attention and transformers

Explain attention as context-dependent weighting, self-attention as relating elements within a sequence, and the transformer as a composition of attention and other learned operations. Introduce positions, masking, and local versus distant context conceptually. Tokenize a real epoch, run the small attention demonstration, and inspect the weight matrix without treating it as a brain connectivity graph. Explain training status and offline/causal limitations. No detailed Q/K/V mathematics or required full transformer training.

### W4D5 — Applications, held-out evidence, and the four-week synthesis

Consolidate the real CNN versus classical/dummy comparison, errors, and subject-generalization results. Cover all seven requested applications with concrete targets, acquisition geometry, label provenance, and primary failure modes. Distinguish seizure detection from prediction, experimental emotion categories from unrestricted emotion reading, and stimulus/intention decoding from unrestricted thought decoding. Discuss MRI inputs separately from EEG: voxel axes are spatial, while EEG sample axes are temporal.

Finish with a Week 4 mastery test and a cumulative four-week test spanning biology → measurement → preprocessing → representation → learning → evaluation → interpretation, with full solutions. End the foundation phase by stating what learners can now formulate and verify and what evidence future advanced work would require. Do not invent a mandatory Week 5 curriculum or link to nonexistent files.

Each daily note must include valid YAML frontmatter, reciprocal navigation, an overview/goals, 4–8 prerequisite questions with complete answers, detailed core exposition, at least one executed laboratory, exact console output, relevant figures, a daily mastery checkpoint with explanatory solutions, references, and a summary/bridge. Maintain the references' substantive depth through worked reasoning rather than filler or repeated definitions.

## 7. Required figures

Generate all Week 4 PNGs through one entry point, `generate_week4_plots.py`, using Seaborn over Matplotlib:

`sns.set_theme(style="whitegrid", context="talk", palette="deep")`, figure dpi at least 150, and tight saving.

Produce approximately 12 purposeful figures; combine panels when helpful:

1. Four-week pipeline and the distinction between handcrafted and learned representations.
2. Artificial-neuron forward pass, activation, and actual weight-update example.
3. Real MLP training/validation curves, with checkpoint-selection rule stated.
4. A real EEG epoch through the CNN, annotated with every tensor shape.
5. Temporal convolution/kernel sliding over a real trace, including kernel duration.
6. Learned CNN kernels, feature maps, and pooling effects on the same documented trial.
7. Actual held-out confusion matrices and classical/dummy/CNN metric comparison.
8. Real per-subject or fixed held-out-subject results with the exact protocol labeled.
9. RNN/LSTM state progression on real EEG, explicitly labeled with training status.
10. EEG time-patch/token construction and sequence order.
11. Self-attention weights over those real tokens, with training status and interpretation limits.
12. Application comparison and cumulative architecture trade-offs, with target/input geometry and limitations.

Most signal/feature/result figures should use actual recorded EEG or executed predictions. Concept diagrams are acceptable and should be labeled as explanatory. Synthetic pedagogical experiments must be identified as such. Never invent learning curves, trained filters, performance numbers, or attention interpretations.

Use physical units for EEG voltage, time, frequency, and power; use explicit dimensionless units for normalized input, activation, loss, probability, and attention weights where appropriate. Label confusion-matrix orientation, split protocol, and class names. A feature-map heatmap is a representation, not an anatomical source reconstruction. Select trial examples by a documented deterministic rule rather than favorable appearance.

Embed report images with `assets/<name>.png` and day-note images with `../assets/<name>.png`. Immediately follow each image with a sequential `**Figure N.**` caption that explains the mechanism or inference and its limits. Number independently within each document. Open and visually inspect every generated image; correct unreadable labels, clipping, inappropriate scales, and text/figure disagreement.

## 8. Deliverables and graph integration

Create:

- `month 1/week 4/week-4-report.md`
- `month 1/week 4/week4-export.md` and identical mirror `week-4-export.md`
- `month 1/week 4/days/w4d1.md` through `w4d5.md`
- `month 1/week 4/assets/*.png`
- `generate_week4_plots.py`, with supporting acquisition/experiment modules if useful.
- Week 4 configuration/provenance, retained-trial manifest, explicit splits, input/normalization schemas, training histories, predictions, metrics, captured laboratory outputs, and small model checkpoints needed to reproduce the figures.
- Atomic notes for concepts actually used: for example `artificial-neuron`, `activation-function`, `forward-propagation`, `backpropagation`, `convolution`, `pooling`, `recurrent-neural-network`, `long-short-term-memory`, `self-attention`, `transformer`, and `representation-learning`. Reuse existing notes for overfitting, leakage, experimental labels, and subject generalization.

Keep raw downloads in a documented cache and out of routine Git commits and Obsidian note indexing. Preserve the exact imported Week 4 curriculum in both exports, accurately identifying its source. Do not fabricate a missing `GI.docx` brief.

The report must have complete frontmatter, exact title, 4–6 sentence abstract, keywords, inherited Week 3 question and linked five-day syllabus, numbered sections covering all curriculum topics, executed implementations, empirical comparison, all requested applications, limitations, conclusion, verified annotated references, Appendix A reproducibility, and Appendix B graph connections. Follow the golden reports' structure and tone; synthesize the week into one argument rather than concatenating the daily notes. Keep main-section numbering gapless.

Use `ns/week-04` tags, actual creation dates, normally quoted YAML strings/wikilinks, and status `complete` only after verification. Escape alias pipes inside tables, for example `[[w4d1\|**W4D1**]]`. Connect W4D1 backward to `w3d5`, subsequent days reciprocally, and W4D5 to the Week 4 synthesis/hub. The report links backward to Week 3 and to the hub. There is no required `next` link to Week 5.

Update Week 3's forward navigation and curricular handoff to the actual Week 4 report; replace the inaccurate unsupervised-learning trajectory with the supplied deep-learning scope. Update `Knowledge-Map.md` with the completed four-week spine, all five Week 4 days, concept taxonomy, and cumulative governing pipeline. Add genuine backlinks from reused concepts and an **Appears in:** line plus `## Related` links to every new atomic note. Match the atomic-note definition length requirements in applicable instructions.

Preserve unrelated work. Limit prior-note edits to the required links, backlinks, and targeted handoff correction. If prior numerical/scientific claims cannot be verified, qualify them in new material and report the discrepancy rather than rewriting previous experiments as part of this task.

## 9. Environment, determinism, and honest code

Use `uv` for all Python execution and dependency changes. Inspect the current project before installation; `mne` and `scikit-learn` are already declared. Add PyTorch through `uv add torch` if needed, respecting the current Python/platform requirements and the intentional `scipy>=1.14,<1.15` constraint. Add any directly imported dependencies explicitly rather than depending on accidental transitive installation. Keep `uv.lock` authoritative and report versions from the live environment, not copied prose.

Seed NumPy and PyTorch with 42, use deterministic algorithms where supported, seed minibatch shuffling explicitly, use stable subject/trial ordering, and document CPU/thread settings. Consult the installed version's PyTorch reproducibility documentation: https://docs.pytorch.org/docs/stable/notes/randomness.html. A seed alone does not guarantee identical results across devices or versions.

Separate network acquisition and timestamped provenance from cached deterministic analysis. Resolve paths relative to the project/script, avoiding personal absolute paths in reproducibility commands. Keep checkpoints separate from image generation if useful, but provide an exact command sequence that reproduces both training and figures from raw cached data. Print a stable figure manifest.

Run every Python block as written through `uv run python`; include explicit imports, definitions, inputs, and prerequisite commands. No ellipses, undeclared variables, hidden notebook state, pseudocode presented as executable code, or manually estimated outputs. Short recurrent/attention demonstrations are valid complete laboratories when their purpose and training status are transparent. Capture actual stdout and address substantive warnings.

Rerun the cached training/analysis pipeline twice in the same locked CPU environment. Verify byte-identical PNGs and laboratory stdout and agreement of saved predictions, metrics, and training histories. Exclude volatile timestamps, download progress, elapsed times, and checkpoint serialization metadata from byte-level comparison unless deliberately normalized. If determinism cannot be achieved, diagnose it and report the specific limitation; never falsely claim a passed check.

## 10. Literature, misconceptions, and applications

Verify academic references through original papers, publishers, and official documentation. Cite foundational backpropagation, convolutional networks, LSTM, and transformer work as appropriate, with full bibliographic details, a verified DOI where one exists, and a one-sentence contribution. Do not invent a DOI for a paper or software page without one. Cite the actual dataset and acquisition publication. Include official PyTorch/MNE documentation with access dates for implementation facts; distinguish it from peer-reviewed physiological evidence.

Use the application field order established in the vault:

**Biological Event → Neural Activity → Neural Data → Target / Label → AI Task → Primary Failure Mode**.

For each of the seven applications, state the prediction unit and target provenance: intended movement per trial, seizure occurrence over a time interval, sleep stage per scoring epoch, protocol/self-report emotion label, kinematic or stimulus target, or neuroimaging diagnosis/segmentation label. BCI is a system context that may include motor imagery or other control paradigms, not necessarily a separate disjoint classification task. Separate implemented imagery results from conceptual applications.

Explicitly explain and correct these misconceptions:

- Artificial neurons faithfully simulate biological neurons.
- Deep learning learns useful features without labels or task assumptions.
- A larger/deeper network necessarily performs better on small EEG cohorts.
- CNN feature maps correspond directly to anatomical brain regions.
- Pooling preserves all event timing.
- RNNs/LSTMs automatically remember arbitrarily long sequences.
- Attention weights explain the causal source of a prediction.
- Transformers inherently know sample order and physical time.
- Raw EEG removes the need for preprocessing or leakage controls.
- High softmax confidence establishes calibration, correctness, or clinical safety.
- Better training loss proves better unseen-subject generalization.
- Motor-imagery results validate seizure, sleep, emotion, or diagnostic performance.

## 11. Completion gate

Before declaring the module complete, execute checks confirming:

- All canonical files, mirrored exports, scripts, assets, and empirical artifacts exist.
- Real EEG was actually loaded or downloaded; provenance, license, checksums, labels, task intervals, trial IDs, units, and exclusions are accurate.
- Feature and epoch representations align row-for-row with labels and metadata; shapes and finite values are validated.
- Forward-pass dimensions, parameter counts, loss/target conventions, and optimizer updates are correct.
- Every learned normalization step uses training data only; no trial/run/subject violates the stated split; no test data influenced selection.
- Training/inference mode and gradient behavior are correct; weights and test-time statistics remain frozen.
- Classical/dummy comparisons use matching observations and partitions; reported metrics match saved held-out predictions.
- Recurrent and attention diagrams disclose training status; physiological interpretations are appropriately limited.
- Every published code block executed through `uv`, its displayed output is genuine, and repeatability checks pass or clearly identify a remaining defect.
- YAML parses, sections/captions are sequential, image references resolve, table pipes are escaped, and new/modified wikilinks resolve.
- New concepts have backlinks and connect to the hub; no nonexistent Week 5 links were introduced. Distinguish pre-existing vault defects from new ones.
- All images were visually inspected and all academic citations/provenance checked.
- All Week 4 syllabus topics and all seven applications are present, and the four-week cumulative mastery test has complete solutions.

Deliver the completed files. End with a concise summary of principal paths, real-data subset, what was actually trained versus illustrated, actual held-out results, reproducibility commands, and any unresolved limitation. Do not declare completion if mandatory real-data training, figures, runnable notes, or verification remains unfinished.
