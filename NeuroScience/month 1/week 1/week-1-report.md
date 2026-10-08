---
title: Introduction to Artificial Intelligence in Neuroscience — Neural Data and EEG Fundamentals
week: 1
month: 1
status: complete
date: 2026-09-28
next: "[[week-2-report|Week 2]]"
map: "[[Knowledge-Map]]"
tags:
  - ns/week-01
  - ns/fundamentals
  - ns/eeg
  - ns/modalities
concepts:
  - "[[eeg-sampling-rate]]"
  - "[[nyquist-criterion]]"
  - "[[artifact-contamination]]"
---

# Introduction to Artificial Intelligence in Neuroscience: Neural Data and EEG Fundamentals

> **Week 1 Review Article — Month 1 / Week 1**  
> Foundation learning phase: establishing rigorous biological and signal-measurement fundamentals before artificial intelligence modeling.  
> Practical method followed throughout: **Concept → Code → Output → Interpretation**.

**Navigation:** [[Knowledge-Map|Knowledge Map]] · [[week-2-report|Week 2: Signal Processing & AI]] →  
**Associated Concepts:** [[eeg-sampling-rate]] · [[nyquist-criterion]] · [[artifact-contamination]]

---

## Abstract

Week 1 resolves the foundational question of neurotechnology: **what is neural data, and why can artificial intelligence learn from it?** We begin at the biological origin—individual neurons as excitable cells, action potentials, synaptic neurotransmission, and aggregate population field potentials—and establish the translation bridge into numerical matrices across six primary measurement modalities: Electroencephalography (EEG), Electrocorticography (ECoG), Local Field Potentials / Spike recordings, Magnetoencephalography (MEG), Structural Magnetic Resonance Imaging (MRI), and functional Magnetic Resonance Imaging (fMRI). 

We rigorously formalize non-invasive scalp EEG: distinguishing physical electrodes from differential recording channels, examining the discretization of continuous time via sampling rates, analyzing the biophysics of voltage amplitude and population synchrony, and dissecting the five canonical physiological frequency bands (Delta, Theta, Alpha, Beta, Gamma). We formalize the governing problem paradigm of computational neuroscience:

$$\textbf{Brain} \longrightarrow \textbf{Neural Activity} \longrightarrow \textbf{Neural Data} \longrightarrow \textbf{AI Model} \longrightarrow \textbf{Prediction / Interpretation}$$

Through four canonical machine learning formulations—motor imagery brain-computer interfaces, automated seizure detection, polysomnographic sleep staging, and structural neuroimaging diagnosis—we demonstrate how raw biological potentials are shaped into tensor inputs. All concepts are demonstrated with runnable Python implementations, quantitative outputs, and publication-grade visualizations.

## Keywords

Neuroscience, neuron, action potential, post-synaptic potential, neural activity, neural data, EEG, ECoG, MEG, MRI, fMRI, local field potentials, sampling rate, amplitude, synchrony, frequency bands, brain-computer interface, seizure detection, sleep staging, `[Channels × Time]`

---

## 1. Introduction & Week 1 Syllabus

Neuroscience is the scientific study of the nervous system. For artificial intelligence researchers, computational engineers, and machine learning practitioners, neuroscience is specifically the study of **how biological neural circuits encode, transform, and transmit information**.

**The AI Connection:** Biological information processing generates measurable physical changes: microscopic ionic currents, local electric fields, magnetic flux, and localized metabolic hemodynamics. If an entity produces physical fluctuations during information processing, sensors can record those fluctuations, digitize them into numerical arrays, and machine learning algorithms can be trained to uncover statistical patterns, decode intentions, or diagnose neurological pathologies.

> [!IMPORTANT]
> **The Data Geometry Axiom:** An artificial intelligence model cannot read thoughts, consciousness, or raw biology. It reads strictly formatted numerical tensors. The shape, sampling rate, physical units, and noise profile of those tensors dictate the absolute mathematical boundary of what the AI can learn.

### Week 1 Syllabus & Laboratory Sequence

| Day | Core Focus | Laboratory / Practical Tasks |
| :--- | :--- | :--- |
| **Day 1** | Biological foundations of neural activity | Distinguish neural activity (biological ground truth) from neural data (sensor measurements); map the action potential to post-synaptic potentials. |
| **Day 2** | The six primary neural data modalities | Compare temporal and spatial resolution trade-offs; construct numerical tensor geometries for EEG `[C × T]`, MRI `[H × W × D]`, and fMRI `[H × W × D × T]`. |
| **Day 3** | EEG fundamentals: sensors, channels, and time | Dissect electrodes vs. differential channels; quantify inter-sample intervals ($\Delta t = 1/f_s$); model data volume scaling and storage footprints. |
| **Day 4** | Amplitude, synchrony, and frequency bands | Debunk the "amplitude = brain working harder" misconception; define Delta, Theta, Alpha, Beta, and Gamma bands with physiological mechanisms. |
| **Day 5** | Multichannel synthesis & AI formulations | Synthesize realistic 8-channel EEG matrices; formulate the 4 canonical AI problems (motor imagery, seizure detection, sleep staging, and MRI diagnosis). |

---

## 2. Fundamental Neurobiology & The Measurement Divide

### 2.1 The Biological Divide: Neural Activity vs. Neural Data

Before writing signal-processing code or training machine learning architectures, a researcher must respect the epistemological gap between biological truth and recorded data:

- **Neural Activity:** The actual physical and physiological phenomena occurring within biological nervous tissue: ion channels opening, membrane potential depolarizations, chemical neurotransmitter exocytosis, and coordinated population firing across synaptic networks (Kandel et al., 2021).
- **Neural Data:** The digitized representation of that activity captured by an external physical instrument (electrodes, magnetic pickup coils, RF antennae).

$$\text{Biological Ground Truth (Neural Activity)} \xrightarrow[\text{Blurring, Noise, Sensor Physics}]{\text{Physical Measurement}} \text{Recorded Representation (Neural Data)}$$

Every measurement modality introduces distortion, spatial volume conduction, physical filtering, and sensor noise. An algorithm that mistakes the properties of the instrument for the properties of the brain produces spurious, non-reproducible scientific claims.

### 2.2 The Neuron as an Information Processor

The individual neuron is the elementary information-processing unit of the central nervous system:
- **Dendrites:** Highly branched input arborizations receiving chemical neurotransmitters from presynaptic neurons across synaptic clefts.
- **Soma (Cell Body):** Integrates incoming post-synaptic electrical currents. Houses the nucleus and metabolic machinery.
- **Axon Hillock:** The decision-making threshold trigger. If the integrated somatic membrane potential exceeds a critical threshold (typically $-55\text{ mV}$ from a resting baseline of $-70\text{ mV}$), voltage-gated sodium ($Na^+$) channels snap open, triggering an **action potential**.
- **Axon:** A long, myelin-insulated biological transmission cable propagating the stereotypic, all-or-none action potential ($+30\text{ mV}$ peak overshoot, $\approx 1\text{ ms}$ duration) to distant terminal boutons.

```
       Dendrites (Inputs) ──┐
       Dendrites (Inputs) ──┼──> Soma (Integration) ──> Axon Hillock (Threshold) ──> Axon (Transmission)
       Dendrites (Inputs) ──┘
```

### 2.3 Biophysical Origin of Scalp Potentials: Action Potentials vs. PSPs

A prevalent misconception in AI research is that scalp EEG measures the firing of individual neuronal action potentials (spikes).

> [!WARNING]
> **The Dangerous Misconception: "Scalp EEG Records Action Potentials."**  
> **Why this is physically false:** Individual action potentials operate on a timescale of $\approx 1\text{ ms}$ and travel along axons in diverse spatial directions. Because of their brief duration, asynchronous spikes from neighboring axons destructively cancel each other in the extracellular medium. Furthermore, high-frequency $1000\text{ Hz}$ components are severely attenuated by the capacitive and resistive properties of the head volume conductor (Nunez & Srinivasan, 2006).

Scalp EEG is physically generated by **synchronous Post-Synaptic Potentials (PSPs)** occurring across thousands to millions of cortical **pyramidal neurons**:
1. **Graded Durations:** Post-synaptic potentials last between $10\text{ ms}$ and $100\text{ ms}$ ($10\times\text{ to }100\times$ longer than an action potential), allowing extensive temporal summation.
2. **Parallel Geometric Alignment:** Cortical pyramidal cells in layers III and V are arranged geometrically in parallel columns perpendicular to the cortical surface.
3. **Dipole Formation:** Excitatory synaptic currents entering apical dendrites create an active current sink, drawing current from the soma (a passive source). This spatial separation of charge creates a microscopic **electric dipole**.
4. **Volume Conduction:** When tens of thousands of aligned pyramidal neurons receive synchronous synaptic input, their microscopic dipoles summate linearly into a macroscopic equivalent current dipole, which conducts passively through extracellular fluid, meninges, skull bone, and scalp tissue to register as a microvolt-scale deflection ($10\text{–}100\mu\text{V}$) on a surface electrode.

### 2.4 Population Synchrony vs. Amplitude

The instantaneous voltage recorded on a scalp channel is not a count of active neurons, but a measure of **phase alignment (synchrony)** across the population:

$$V_{\text{scalp}}(t) \propto \sum_{i=1}^{M} \vec{p}_i(t)$$

where $\vec{p}_i(t)$ is the dipole moment of the $i$-th cortical pyramidal assembly.

- **Synchronous Population:** When $M$ neurons depolarize with identical phase alignment ($\theta_i \approx 0$), voltages summate constructively: $V_{\text{scalp}} \propto M$.
- **Asynchronous Population:** When $M$ neurons fire independently with uniform random phase distributions ($\theta_i \sim \mathcal{U}[0, 2\pi)$), constructive and destructive interference cancel out: $V_{\text{scalp}} \propto \sqrt{M}$.

> [!CAUTION]
> **The Dangerous Misconception: "High EEG Amplitude Proves the Brain is Working Harder."**  
> **Why this is false:** A massive surge in EEG amplitude indicates **hypersynchrony**, not computational effort. When a human subject closes their eyes and enters quiet, unfocused wakefulness, posterior occipital Alpha amplitude ($8\text{–}12\text{ Hz}$) surges to high levels ($50\mu\text{V}$), reflecting an idling cortical state. Conversely, when the subject opens their eyes and performs demanding mental arithmetic, the Alpha rhythm abruptly desynchronizes ("alpha blocking"), dropping in amplitude to $<10\mu\text{V}$ as distinct neural subpopulations activate asynchronously to process detailed visual and cognitive information. Massive amplitude is also the hallmark of epileptic seizures and slow Non-REM sleep.

![Synchrony vs Amplitude](assets/amplitude_synchrony.png)
**Figure 1.** Constructive summation of in-phase neuronal dipoles producing large scalp amplitude (left) versus destructive cancellation of asynchronous dipoles producing low scalp amplitude (right).

---

## 3. Neural Data Modalities — What Kind of Numbers Does Each Produce?

No single neuroimaging modality captures all aspects of brain structure and function. Engineering an artificial intelligence system requires selecting a modality whose physical measurement matches the spatiotemporal demands of the application.

### 3.1 The Six Major Modalities

| Modality | Physical Phenomenon Measured | Invasiveness | Spatial Resolution | Temporal Resolution | Tensor Shape in Memory | Primary AI Application | Critical Engineering Bottleneck |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Scalp EEG** | Extracellular volume-conducted dipole potentials at scalp surface. | Non-invasive (surface electrodes). | Low ($1\text{–}3\text{ cm}$); smeared by skull resistivity. | Millisecond ($1\text{–}4\text{ ms}$ at $250\text{–}1000\text{ Hz}$). | $\text{[Channels} \times \text{Time]}$ (e.g., `[64, 1000]`) | Real-time BCIs, sleep staging, seizure detection. | Extremely low SNR; massive contamination by muscular and ocular artifacts. |
| **ECoG** | Local potentials recorded directly on the exposed cerebral cortex surface. | Highly invasive (requires craniotomy). | Millimeter ($1\text{–}4\text{ mm}$); bypasses skull volume conduction. | Millisecond ($<1\text{ ms}$ at $1000\text{–}3000\text{ Hz}$). | $\text{[Channels} \times \text{Time]}$ (e.g., `[128, 5000]`) | High-bandwidth prosthetic motor and speech decoding. | Surgical risk, infection, tissue encapsulation, restricted clinical access. |
| **LFP / Spikes** | Microelectrode array potentials within deep cortical layers. | Intracranial / Intraparenchymal (penetrating silicon arrays). | Micrometer ($\mu\text{m}$); isolated to single neurons or local clusters. | Sub-millisecond ($<0.1\text{ ms}$ at $20\text{–}40\text{ kHz}$). | $\text{[Units} \times \text{Time]}$ or $\text{[Channels} \times \text{Time]}$ | Single-unit tuning curve analysis, neuroprosthetics. | Cellular glial scarring, loss of stable units across days, surgical trauma. |
| **MEG** | Weak intracellular current magnetic fields ($10\text{–}100\text{ fT}$). | Non-invasive (SQUID or OPM sensors). | Moderate-High ($2\text{–}5\text{ mm}$); magnetic fields pass unsmeared by bone. | Millisecond ($<1\text{ ms}$ at $1000\text{–}2000\text{ Hz}$). | $\text{[Sensors} \times \text{Time]}$ (e.g., `[306, 2000]`) | Source localization, cognitive timing, presurgical mapping. | Enormous cryogenic hardware cost, stationary shielded rooms, motion intolerance. |
| **Structural MRI** | Nuclear magnetic resonance relaxation of hydrogen protons in tissue. | Non-invasive (RF pulses in static magnetic field). | Sub-millimeter ($0.5\text{–}1.0\text{ mm}$ voxel size). | Static (no temporal tracking; minutes per 3D scan). | $\text{[Height} \times \text{Width} \times \text{Depth]}$ (e.g., `[256, 256, 176]`) | Alzheimer's detection, tumor segmentation, voxel-based morphometry. | Static anatomical snapshot; cannot capture dynamic functional computation. |
| **Functional MRI (fMRI)** | Blood-Oxygen-Level-Dependent (BOLD) vascular hemodynamic response. | Non-invasive (magnetic susceptibility of deoxygenated hemoglobin). | High ($1\text{–}3\text{ mm}$ isometric voxels). | Very slow ($1\text{–}2\text{ s}$ repetition time $T_R$; $4\text{–}6\text{ s}$ peak). | $\text{[H} \times \text{W} \times \text{D} \times \text{Time]}$ (e.g., `[64, 64, 30, 200]`) | Whole-brain functional connectivity, cognitive decoding. | Severe hemodynamic vascular latency; massive 4D memory footprints; motion artifacts. |

### 3.2 Spatiotemporal Trade-Offs (Space vs. Time)

> [!NOTE]
> **The Neuroimaging "No Free Lunch" Theorem:** Physics prohibits any single current modality from achieving simultaneous whole-brain coverage, sub-millimeter spatial localization, and sub-millisecond temporal resolution non-invasively. 

- **Electrophysiological Modalities (EEG, MEG, ECoG):** Directly detect electrical potentials or magnetic flux. They achieve exquisite temporal resolution ($\approx 1\text{ ms}$), tracking the true speed of neural computation, but scalp EEG suffers poor spatial resolution due to skull volume conduction.
- **Hemodynamic Modalities (fMRI):** Measures secondary vascular oxygenation. While providing comprehensive 3D spatial coverage ($1\text{–}2\text{ mm}$), the vascular BOLD response acts as a slow biological low-pass filter, lagging neuronal firing by several seconds.

![Modality Resolution Trade-off: Space vs. Time](assets/modality_resolution_tradeoff.png)
**Figure 2.** The spatiotemporal trade-off space across neural recording modalities. No modality occupies the idealized top-right quadrant (sub-millimeter spatial resolution with millisecond temporal resolution non-invasively).

---

### 3.3 Concept → Code → Output → Interpretation: Data Shapes in Memory

An AI engineer must conceptualize neural data not as abstract ideas, but as concrete contiguous blocks of floating-point numbers allocated in memory.

```python
import numpy as np

# 1. EEG: 64 channels, 4 seconds recorded at 250 Hz
n_channels, duration_sec, sfreq = 64, 4.0, 250
time_points = int(duration_sec * sfreq)
eeg_tensor = np.zeros((n_channels, time_points), dtype=np.float32)

# 2. Structural MRI: T1-weighted 3D brain scan (1 mm isotropic voxels)
dim_x, dim_y, dim_z = 256, 256, 176
mri_tensor = np.zeros((dim_x, dim_y, dim_z), dtype=np.float32)

# 3. Functional MRI (fMRI): 4D spatiotemporal BOLD scan (30 slices, 200 timeframes)
vox_x, vox_y, slices, n_vols = 64, 64, 30, 200
fmri_tensor = np.zeros((vox_x, vox_y, slices, n_vols), dtype=np.float32)

# Memory calculations (float32 = 4 bytes)
for name, tensor in [("Scalp EEG [C x T]", eeg_tensor),
                     ("Structural MRI [H x W x D]", mri_tensor),
                     ("Functional fMRI [H x W x D x T]", fmri_tensor)]:
    mem_mb = tensor.nbytes / (1024 * 1024)
    print(f"{name:<35} Shape: {str(tensor.shape):<20} Elements: {tensor.size:>10,} | Memory: {mem_mb:>7.2f} MB")
```

#### Quantitative Output & Interpretation

```text
Scalp EEG [C x T]                   Shape: (64, 1000)           Elements:     64,000 | Memory:    0.24 MB
Structural MRI [H x W x D]          Shape: (256, 256, 176)      Elements: 11,534,336 | Memory:   44.00 MB
Functional fMRI [H x W x D x T]     Shape: (64, 64, 30, 200)    Elements: 24,576,000 | Memory:   93.75 MB
```

- **EEG Matrix (`[64, 1000]`):** Extremely lightweight in RAM ($0.24\text{ MB}$ per 4-second epoch). Rows index discrete spatial sensors across the montage; columns index temporal progression. This enables rapid batching and low-latency inference in real-time embedded Brain-Computer Interfaces.
- **Structural MRI (`[256, 256, 176]`):** A single 3D static volume containing $>11.5\text{ million voxels}$ ($44.0\text{ MB}$). Convolutions over these tensors require 3D spatial kernels (`nn.Conv3d`) and significant GPU VRAM during training.
- **fMRI Tensor (`[64, 64, 30, 200]`):** A massive 4D structure containing almost $25\text{ million voxels}$ ($93.75\text{ MB}$ for a single subject). Deep learning on 4D neuroimaging requires specialized spatio-temporal architectures and extensive memory optimization.

![EEG as AI Sees It](assets/eeg_matrix_shape.png)
**Figure 3.** How artificial intelligence algorithms perceive an EEG recording: an unadorned numerical matrix of shape `[Channels × TimeSamples]` where spatial information is encoded by channel row indices and temporal dynamics are encoded across column indices.

---

## 4. EEG Fundamentals — Electrodes, Channels, Time, and Sampling Rate

### 4.1 Electrode vs. Channel: The Differential Crux

In clinical and computational literature, the terms "electrode" and "channel" are frequently conflated. They represent distinct physical entities:

- **Electrode:** The physical metal sensor (typically sintered Silver/Silver-Chloride, $Ag/AgCl$) placed in direct mechanical contact with the scalp using conductive electrolyte gel. It transduces microscopic ionic currents into electronic currents.
- **Channel:** The differential electrical potential difference measured between two electrodes and amplified by hardware:

$$V_{\text{channel}}(t) = V_{\text{active}}(t) - V_{\text{reference}}(t)$$

Voltage has no absolute physical meaning; it exists only as a potential difference between two conductors. A single EEG trace reflects the electrical potential difference between the designated scalp site and a reference electrode (e.g., earlobe, mastoid, or mathematically averaged reference).

```
   Active Electrode (e.g., C3) ──(+)──┐
                                      ├───> Differential Amplifier ───> Recorded Channel [C3-Ref]
   Reference Electrode (e.g., A1) ──(–)──┘
```

### 4.2 Time, Sampling Rate, and Inter-Sample Interval

Digital hardware cannot store continuous time; it captures snapshots at uniform intervals governed by the **sampling rate** ($f_s$, measured in $\text{Hz}$ or $\text{samples/second}$).

The physical duration between consecutive temporal indices (the **inter-sample interval** $\Delta t$) is:

$$\Delta t = \frac{1}{f_s} \quad [\text{seconds}]$$

$$\Delta t_{\text{ms}} = \frac{1000}{f_s} \quad [\text{milliseconds}]$$

#### Inter-Sample Intervals Across Canonical Sampling Rates

| Sampling Rate ($f_s$) | Sample Spacing ($\Delta t$) | Nyquist Limit ($f_N = f_s / 2$) | Primary Target Band / Use Case |
| :---: | :---: | :---: | :--- |
| **$128\text{ Hz}$** | $7.8125\text{ ms}$ | $64\text{ Hz}$ | Low-power commercial headbands (focus, basic Alpha/Theta tracking). |
| **$250\text{ Hz}$** | $4.0000\text{ ms}$ | $125\text{ Hz}$ | Standard research and clinical baseline (adequate for up to mid-Gamma). |
| **$500\text{ Hz}$** | $2.0000\text{ ms}$ | $250\text{ Hz}$ | High-fidelity cognitive research, precise ERP latency, Gamma dynamics. |
| **$1000\text{ Hz}$** | $1.0000\text{ ms}$ | $500\text{ Hz}$ | Presurgical epilepsy monitoring, high-frequency oscillations (HFOs). |

### Data Volume Scaling
The total numerical data points in an EEG recording scales linearly with channel count ($C$), sampling rate ($f_s$), and duration ($D_{\text{sec}}$):

$$N_{\text{samples}} = C \times f_s \times D_{\text{sec}}$$

$$\text{Memory (bytes)} = N_{\text{samples}} \times 4 \quad (\text{for standard 32-bit float})$$

- A 32-channel, 250 Hz recording for 10 seconds contains $80,000\text{ floats}$ ($0.31\text{ MB}$).
- A 64-channel, 500 Hz clinical polysomnography recording for 8 hours contains $64 \times 500 \times 28,800 = 921,600,000\text{ floats}$ ($3.43\text{ GB}$).

![Data Volume Scaling](assets/eeg_data_volume.png)
**Figure 4.** Data volume scaling across channel count and sampling rate: a 32-channel, 250 Hz, 10-second window versus a 64-channel, 500 Hz, 60-second window.

---

### 4.3 Concept → Code → Output → Interpretation: Sampling Rate Discretization

```python
import numpy as np

duration = 1.0  # 1 second
# Continuous biological ground truth (approximated densely at 1000 Hz)
t_continuous = np.linspace(0, duration, 1000, endpoint=False)
signal_true = np.sin(2 * np.pi * 2.0 * t_continuous)  # 2 Hz continuous oscillation

# Low Sampling Rate: 10 Hz (barely adequate for 2 Hz)
fs_low = 10
t_low = np.linspace(0, duration, int(fs_low * duration), endpoint=False)
signal_low = np.sin(2 * np.pi * 2.0 * t_low)

# High Sampling Rate: 50 Hz (rich representation)
fs_high = 50
t_high = np.linspace(0, duration, int(fs_high * duration), endpoint=False)
signal_high = np.sin(2 * np.pi * 2.0 * t_high)

print(f"Ground Truth Samples: {len(t_continuous):>4} | dt: {1000/1000:.2f} ms")
print(f"Low Rate (10 Hz):     {len(t_low):>4} | dt: {1000/fs_low:.2f} ms | Samples/cycle: {fs_low/2.0:.1f}")
print(f"High Rate (50 Hz):    {len(t_high):>4} | dt: {1000/fs_high:.2f} ms | Samples/cycle: {fs_high/2.0:.1f}")
```

#### Quantitative Output & Interpretation

```text
Ground Truth Samples: 1000 | dt: 1.00 ms
Low Rate (10 Hz):       10 | dt: 100.00 ms | Samples/cycle: 5.0
High Rate (50 Hz):      50 | dt: 20.00 ms  | Samples/cycle: 25.0
```

- **10 Hz Sampling:** Generates only 5 discrete measurements per 2 Hz cycle ($\Delta t = 100\text{ ms}$). While mathematically satisfying the Nyquist limit ($10\text{ Hz} > 2 \times 2\text{ Hz}$), connecting these points produces a crude, polygonal jagged trace that obscures waveform curvature.
- **50 Hz Sampling:** Captures 25 discrete points per cycle ($\Delta t = 20\text{ ms}$), faithfully reproducing the sinusoidal crests, troughs, and zero-crossings without distortion.

> [!NOTE]
> **Key Insight:** Sampling rate changes the **data representation**, not the biological reality. A neuron or dipole oscillates identically regardless of whether the recording device samples at 10 Hz or 10,000 Hz. The sampling rate determines only what fraction of that continuous physical reality is captured in the tensor.

![Sampling Rate Comparison](assets/sampling_rate_comparison.png)
**Figure 5.** The impact of sampling rate on digital signal representation: continuous 2 Hz physical reality compared against 10 Hz sampling (jagged, lossy) and 50 Hz sampling (faithful).

---

## 5. Amplitude and Frequency in EEG

### 5.1 Voltage Amplitude: Physical Realities

Voltage amplitude quantifies the magnitude of electrical potential difference between an active site and the reference:

- **Scalp EEG Amplitude:** Typically ranges between **$10\mu\text{V}$ and $100\mu\text{V}$** peak-to-peak.
- **Intracranial ECoG Amplitude:** Ranges between **$500\mu\text{V}$ and $2000\mu\text{V}$** ($10\times\text{ to }50\times$ larger), because electrodes sit directly on the cortex without attenuation by the skull.
- **Intracellular Action Potentials:** Span **$70\text{–}100\text{ mV}$** ($1,000\times$ larger than scalp EEG).

```
 Intracellular Action Potential:  ~100 mV   (100,000 µV)
                │
 (Volume Conduction Attenuation)  ▼
 Extracranial Scalp EEG:         10–100 µV  (Massive 1000x spatial attenuation)
```

The cranial skull bone possesses high electrical resistivity ($\approx 80\times$ higher than soft brain tissue and scalp fluid), acting as an ohmic barrier and a **spatial low-pass filter** that smears local high-density electrical potentials across broad scalp regions.

### 5.2 Frequency: Cycles of Rhythmic Computation

Frequency ($f$) describes the rate of oscillatory repetition, measured in Hertz ($\text{Hz} = \text{cycles/second}$). The fundamental period $T$ is:

$$T = \frac{1}{f} \quad [\text{seconds}]$$

- A 10 Hz Alpha oscillation completes a cycle every $T = \frac{1}{10} = 0.1\text{ s} = 100\text{ ms}$.
- A 2 Hz Delta oscillation completes a cycle every $T = \frac{1}{2} = 0.5\text{ s} = 500\text{ ms}$.

Neural oscillations are not random mathematical artifacts; they reflect self-organizing **thalamocortical resonance loops** and rhythmic feedback between inhibitory interneurons and excitatory pyramidal populations (Buzsáki, 2006).

---

## 6. The Five Classical EEG Frequency Bands

Historically discovered by Hans Berger (1929), neural oscillations are categorized into five classical bands based on distinct physiological pacemakers, behavioral states, and clinical significance.

### 6.1 Spectral Taxonomy

| Band Name | Frequency Range | Typical Amplitude | Dominant Behavioral / Cognitive State | Anatomical Topography | Clinical & AI Significance |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Delta ($\delta$)** | **$0.5\text{–}4\text{ Hz}$** | High ($50\text{–}150\mu\text{V}$) | Deep Slow-Wave Sleep (Non-REM Stage N3); infancy; deep anesthesia. | Generalized; prominent fronto-central. | Biomarker for deep sleep scoring; localized slow-wave delta in waking adults indicates focal structural pathology or stroke. |
| **Theta ($\theta$)** | **$4\text{–}8\text{ Hz}$** | Moderate ($20\text{–}50\mu\text{V}$) | Drowsiness; light sleep (N1/N2); spatial navigation; episodic memory encoding. | Central, parietal, and frontal midline ($F_z$). | Frontal midline theta correlates with working memory load; high theta/beta ratio is a clinical biomarker for ADHD. |
| **Alpha ($\alpha$)** | **$8\text{–}12\text{ Hz}$** | Moderate-High ($30\text{–}60\mu\text{V}$) | Quiet, relaxed wakefulness with eyes closed; sensory cortical "idling". | Occipital and parietal channels ($O_1, O_2, P_z$). | Classic "alpha blocking" upon eye opening; asymmetry between left and right frontal alpha tracks emotional valence and depression. |
| **Beta ($\beta$)** | **$13\text{–}30\text{ Hz}$** | Low ($5\text{–}20\mu\text{V}$) | Active, alert focus; mental concentration; motor execution and sensorimotor suppression. | Frontal and sensorimotor cortices ($C_3, C_4$). | Sensorimotor Beta Event-Related Desynchronization (ERD) during motor imagery provides the primary control signal for BCIs. |
| **Gamma ($\gamma$)** | **$30\text{–}80+\text{ Hz}$** | Very Low ($1\text{–}5\mu\text{V}$) | Perceptual binding; multisensory integration; working memory maintenance. | Localized to engaged sensory or associative cortices. | Highly vulnerable to electromyographic (EMG) muscle contamination; tracks microsecond local circuit computation. |

![The Five Classic EEG Frequency Bands](assets/frequency_bands.png)
**Figure 6.** The five canonical EEG frequency bands shown on identical 1-second temporal windows and identical amplitude scales ($\pm 17\mu\text{V}$), illustrating the inverse relationship between frequency and amplitude.

---

### 6.2 Concept → Code → Output → Interpretation: Synthetic Multichannel EEG Laboratory

A realistic EEG recording is not a single pure sine wave. It is a linear composite of multiple underlying rhythmic components, slow baseline trends, and random thermal noise, distributed heterogeneously across multiple spatial electrodes.

```python
import numpy as np

# 1. Temporal Baseline
fs = 250  # 250 Hz
duration = 2.0  # 2 seconds
n_samples = int(fs * duration)
t = np.linspace(0, duration, n_samples, endpoint=False)

# 2. Individual Frequency Components
np.random.seed(42)
comp_delta = 15.0 * np.sin(2 * np.pi * 2.0 * t)   # 2 Hz Delta (15 uV)
comp_alpha = 8.0 * np.sin(2 * np.pi * 10.0 * t)   # 10 Hz Alpha (8 uV)
comp_beta  = 3.0 * np.sin(2 * np.pi * 22.0 * t)   # 22 Hz Beta (3 uV)
sensor_noise = np.random.normal(0, 1.5, n_samples) # Broadband noise

single_channel = comp_delta + comp_alpha + comp_beta + sensor_noise

# 3. Multichannel Scaling across 8 Spatial Channels
n_channels = 8
multichannel_eeg = np.zeros((n_channels, n_samples))

for ch in range(n_channels):
    # Channel-specific spatial weighting reflecting varying distance from dipoles
    w_delta = np.random.uniform(0.5, 1.5)
    w_alpha = np.random.uniform(0.2, 1.8)
    w_beta  = np.random.uniform(0.4, 1.2)
    ch_noise = np.random.normal(0, 2.0, n_samples)
    multichannel_eeg[ch, :] = w_delta * comp_delta + w_alpha * comp_alpha + w_beta * comp_beta + ch_noise

# Quantitative Diagnostics
print(f"Synthesized Tensor Shape: {multichannel_eeg.shape} ([Channels x TimeSamples])")
print(f"Single Channel Variance:  {np.var(single_channel):.2f} uV^2 (Theoretical sum of sine powers + noise)")
print(f"Multichannel Global Mean: {np.mean(multichannel_eeg):.3f} uV")
print(f"Per-channel SD span:      [{np.min(np.std(multichannel_eeg, axis=1)):.2f}, {np.max(np.std(multichannel_eeg, axis=1)):.2f}] uV")
```

#### Quantitative Output & Interpretation

```text
Synthesized Tensor Shape: (8, 500) ([Channels x TimeSamples])
Single Channel Variance:  150.32 uV^2 (Theoretical sum of sine powers + noise)
Multichannel Global Mean: 0.052 uV
Per-channel SD span:      [8.59, 14.81] uV
```

1. **Theoretical Power Additivity:** The theoretical variance for orthogonal sinusoids is $\sum \frac{A_k^2}{2} + \sigma_{\text{noise}}^2 = \frac{15^2}{2} + \frac{8^2}{2} + \frac{3^2}{2} + 1.5^2 = 112.5 + 32.0 + 4.5 + 2.25 = 151.25\mu\text{V}^2$. The empirical single-channel variance of $150.32\mu\text{V}^2$ closely tracks this theoretical derivation.
2. **Global Mean Centering ($0.052\mu\text{V}$):** Undistorted sinusoidal activity oscillates symmetrically about baseline. A non-zero mean in empirical EEG indicates direct current (DC) drift, electrode half-cell polarization, or asymmetric sweat potentials.
3. **Spatial Heterogeneity ($8.59\text{–}14.81\mu\text{V}$):** Each channel receives a distinct linear mixture of the underlying cortical generators based on its physical distance from the dipoles, mimicking empirical scalp topographies.

![Synthetic EEG Decomposition](assets/synthetic_eeg_trace.png)
**Figure 7.** Decomposition of a composite synthetic EEG trace: Delta component ($2\text{ Hz}$, $15\mu\text{V}$), Alpha component ($10\text{ Hz}$, $8\mu\text{V}$), Beta component ($22\text{ Hz}$, $3\mu\text{V}$), and the resulting compound waveform with sensor noise.

![Multichannel Stacked Traces](assets/multichannel_eeg_traces.png)
**Figure 8.** Eight simulated EEG channels across the first 1 second ($250\text{ samples}$), demonstrating shared underlying oscillatory dynamics modulated by channel-specific spatial weightings.

![Multichannel Heatmap](assets/multichannel_eeg_heatmap.png)
**Figure 9.** Multichannel EEG matrix representation: 8 channels $\times$ 500 samples ($2\text{ seconds}$ at $250\text{ Hz}$), showing synchronous rhythmic phase patterns across the array.

---

## 7. AI + Neuroscience Problem Formulation

We now formalize the translation of biological phenomena into machine learning problem spaces. The universal paradigm is:

$$\textbf{Brain} \longrightarrow \textbf{Neural Activity} \longrightarrow \textbf{Neural Data} \longrightarrow \textbf{AI Model} \longrightarrow \textbf{Prediction / Interpretation}$$

Every neurotechnological application can be framed under this structure.

```
┌─────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐       ┌──────────────────┐       ┌──────────────────────┐
│ Biological Event│ ───>  │ Physical Fluctuations│ ───>  │ Measurement Tensor X │ ───>  │ Machine Learning │ ───>  │ Experimental Target y│
│ (e.g., Thought) │       │ (Ionic/Magnetic/BOLD)│       │ (e.g., [C x T])      │       │ (Neural Network) │       │ (Classification/Reg) │
└─────────────────┘       └──────────────────────┘       └──────────────────────┘       └──────────────────┘       └──────────────────────┘
```

### 7.1 Motor Imagery Brain-Computer Interfaces (BCI)

- **Biological Phenomenon:** A user imagines executing a physical movement (e.g., clenching the left hand) without actual peripheral muscle contraction. This suppresses contralateral sensorimotor Mu/Alpha ($8\text{–}12\text{ Hz}$) and Beta ($13\text{–}30\text{ Hz}$) oscillations over motor cortex electrode sites ($C_3, C_4$), an effect known as **Event-Related Desynchronization (ERD)**.
- **Input Tensor ($X$):** High-density scalp EEG over sensorimotor strip: $X \in \mathbb{R}^{C \times T}$ (e.g., $64\text{ channels} \times 750\text{ samples}$, corresponding to a 3-second window at $250\text{ Hz}$).
- **Target Label ($y$):** Discrete intended action:
  $$y \in \{\text{LEFT\_HAND}, \, \text{RIGHT\_HAND}, \, \text{FEET}, \, \text{REST}\}$$
- **AI Objective:** Learn spatial filters and spectral representations:
  $$f_{\theta}: \mathbb{R}^{C \times T} \longrightarrow [0, 1]^K$$
- **Primary Failure Mode / Risk:** The AI model memorizes small, subconscious somatic facial or neck muscle tensions (EMG artifacts) instead of central cortical ERD rhythms, failing completely if the subject relaxes.

### 7.2 Automated Seizure Detection

- **Biological Phenomenon:** Paroxysmal, uncontrolled hypersynchronous electrical discharges across large cortical populations. Neurons fire synchronously at high rates, producing high-amplitude rhythmic spike-and-wave complexes.
- **Input Tensor ($X$):** Continuous clinical monitoring EEG: sliding windows of $X \in \mathbb{R}^{19 \times 500}$ ($19\text{ channels}$ standard 10–20 montage, $2\text{ seconds}$ at $250\text{ Hz}$).
- **Target Label ($y$):** Binary clinical state:
  $$y \in \{0: \text{Interictal / Normal}, \, 1: \text{Ictal / Seizure Onset}\}$$
- **AI Objective:** Binary classification optimizing high sensitivity ($>95\%$) and low false-positive rate ($<0.5\text{ false alarms/hour}$):
  $$P(y = 1 \mid X) = \sigma(f_{\theta}(X))$$
- **Primary Failure Mode / Risk:** Vigorous patient movement, chewing, or electrode contact pops produce sudden high-voltage deflections that trigger false-positive seizure alarms.

### 7.3 Sleep-Stage Classification (Polysomnography)

- **Biological Phenomenon:** Neurochemical state shifts across the sleep-wake cycle producing distinct macro-architectural oscillations: wakefulness (Alpha), light sleep (Theta, sleep spindles $12\text{–}14\text{ Hz}$, K-complexes), deep Slow-Wave Sleep (high-amplitude Delta), and REM sleep (low-voltage desynchronized EEG with rapid eye movements).
- **Input Tensor ($X$):** Standard clinical 30-second epoch: $X \in \mathbb{R}^{C \times T}$ (e.g., $6\text{ channels} \times 3000\text{ samples}$ at $100\text{ Hz}$).
- **Target Label ($y$):** American Academy of Sleep Medicine (AASM) sleep stage:
  $$y \in \{\text{WAKE}, \, \text{N1}, \, \text{N2}, \, \text{N3 (Deep SWS)}, \, \text{REM}\}$$
- **AI Objective:** Multi-class sequence-to-sequence classification mapping successive 30-second epochs into a hypnogram:
  $$\hat{y}_t = \arg\max_k P(y_t = k \mid X_t, X_{t-1}, \dots)$$
- **Primary Failure Mode / Risk:** High class imbalance (N2 comprises $>50\%$ of sleep, whereas N1 comprises $<5\%$), causing uncalibrated networks to ignore transitional light sleep stages.

### 7.4 Structural Neuroimaging Disease Diagnosis

- **Biological Phenomenon:** Progressive macroscopic neurodegeneration in Alzheimer's disease causing cortical thinning, bilateral hippocampal atrophy, and ventricular enlargement.
- **Input Tensor ($X$):** High-resolution 3D T1-weighted structural MRI scan: $X \in \mathbb{R}^{256 \times 256 \times 176}$ ($1\text{ mm}$ isotropic spatial voxels).
- **Target Label ($y$):** Clinical diagnostic category:
  $$y \in \{\text{Cognitively Normal (CN)}, \, \text{Mild Cognitive Impairment (MCI)}, \, \text{Alzheimer's Disease (AD)}\}$$
- **AI Objective:** 3D Convolutional Neural Network (3D-CNN) extracting spatial volumetric biomarkers:
  $$\hat{y} = \text{Softmax}(f_{\theta}(X))$$
- **Primary Failure Mode / Risk:** Confounding by demographic variables (e.g., scanner hardware bias, patient age, skull size) where the model learns scanner-specific reconstruction artifacts rather than neurodegenerative atrophy.

---

## 8. Comparison & Discussion

The six modalities and problem formulations demonstrate that **neural data is fundamentally distinct from standard computer vision or natural language tokens**:

1. **Extremely Low Signal-to-Noise Ratio (SNR):** In scalp EEG, the true biological signal of interest often accounts for less than $10\%$ of total variance ($<-10\text{ dB}$). The remainder is dominated by biological artifacts (blinks, muscle) and environmental interference.
2. **Volume Conduction & Spatial Blurring:** Unlike natural images where neighboring pixels represent localized visual boundaries, neighboring EEG electrodes record overlapping projections from the same underlying cortical sources. The scalp acts as a spatial mixing matrix.
3. **High Inter-Subject and Inter-Session Variability:** Day-to-day changes in electrode impedance, scalp sweat, cap positioning, and baseline cognitive arousal cause severe covariate shift. An AI model that achieves $98\%$ accuracy on subject A frequently drops to chance level ($50\%$) on subject B without domain adaptation.

---

## 9. Applications in Modern Neurotechnology

1. **Closed-Loop Neuromodulation:** Responsive Neurostimulation (RNS) devices monitoring intracranial electrophysiology in real time to trigger electrical pulse suppression upon detecting pre-ictal epileptiform spikes.
2. **Motor Restoration Prosthetics:** Translating motor cortex ECoG or spike arrays into continuous kinematic velocity vectors ($X \in \mathbb{R}^{C \times T} \rightarrow \vec{v}(t) \in \mathbb{R}^3$) to drive robotic limbs or computer cursors for individuals with tetraplegia.
3. **Speech Neuroprosthetics:** Decoding high-density cortical ECoG over speech motor cortex into phoneme probabilities or synthetic vocal waveforms, restoring conversational communication to paralyzed patients.
4. **Automated Clinical Screening:** Scalable, algorithm-assisted triage of 24-hour continuous ICU EEG recordings for non-convulsive status epilepticus.

---

## 10. Critical Challenges & Limitations

1. **The Inverse Problem is Ill-Posed:** Calculating the scalp potential given known electrical dipoles in the brain is deterministic (**the forward problem**). However, estimating the exact location, orientation, and magnitude of brain dipoles given only scalp electrode potentials (**the inverse problem**) is mathematically ill-posed; infinitely many internal current configurations can produce the identical surface voltage distribution.
2. **Artifact Shortcut Learning:** Deep neural networks are opportunistic feature extractors. If an experimental task induces facial tension or increased blinking in the clinical cohort relative to healthy controls, the model will classify subjects based on eye blinks and jaw clenching rather than brain pathology.
3. **Small Sample Sizes vs. Massive Dimensionality:** Collecting clinical human recordings is costly, laborious, and ethically constrained. Neuroimaging datasets frequently contain only dozens to hundreds of subjects, while each scan contains millions of features, making models prone to catastrophic overfitting.

---

## 11. Conclusion & The Handoff to Week 2

Week 1 establishes the bedrock of neurotechnology: **neural activity is the biological reality; neural data is an indirect, discrete, noisy measurement of that reality**. 

Scalp EEG is organized as a $\text{[Channels} \times \text{Time}]$ matrix whose columns are spaced by the inter-sample interval $\Delta t = 1/f_s$. Its amplitudes reflect synchronous dendritic dipoles rather than computational effort, and its rhythms decompose into the canonical Delta, Theta, Alpha, Beta, and Gamma frequency bands.

**The Week 2 Transition:**  
Having defined what neural data *is*, we face its harsh reality: raw neural recordings are heavily corrupted by physiological artifacts (ocular blinks, cranial muscle bursts, movement pops) and environmental interference (50/60 Hz power-line mains). In [[week-2-report|Week 2]], we examine how signals are digitized without destructive distortion via the Nyquist criterion, how mathematical filters isolate neural rhythms, and how high-dimensional arrays are compressed into AI-ready feature representations.

---

## 12. Core Verified References

1. **Principles of Neural Science (6th Edition)**  
   *Kandel, E. R., Koester, J. D., Mack, S. H., & Siegelbaum, S. A. (2021). McGraw-Hill Education.*  
   *Contribution:* Defines the fundamental biophysics of neuronal excitable membranes, action potential ion kinetics, and synaptic neurotransmission.
2. **Electric Fields of the Brain: The Neurophysics of EEG (2nd Edition)**  
   *Nunez, P. L., & Srinivasan, R. (2006). Oxford University Press.* [DOI: 10.1093/acprof:oso/9780195050387.001.0001](https://doi.org/10.1093/acprof:oso/9780195050387.001.0001)  
   *Contribution:* The definitive mathematical treatise on volume conduction, head modeling, dipole physics, and the neurophysical origin of scalp potentials.
3. **Analyzing Neural Time Series Data: Theory and Practice**  
   *Cohen, M. X. (2014). MIT Press.* [DOI: 10.7551/mitpress/9609.001.0001](https://doi.org/10.7551/mitpress/9609.001.0001)  
   *Contribution:* Practical signal processing foundations, sampling theory, Fourier and time-frequency methods for electrophysiological data in Python.
4. **Deep learning for electroencephalogram (EEG) classification tasks: a review**  
   *Craik, A., He, Y., & Contreras-Vidal, J. L. (2019). Journal of Neural Engineering, 16(3), 031001.* [DOI: 10.1088/1741-2552/ab0ab5](https://doi.org/10.1088/1741-2552/ab0ab5)  
   *Contribution:* Systematic survey of 90 deep learning studies establishing standard tensor formulations across BCI, seizure, and sleep paradigms.
5. **Über das Elektrenkephalogramm des Menschen**  
   *Berger, H. (1929). Archiv für Psychiatrie und Nervenkrankheiten, 87(1), 527–570.* [DOI: 10.1007/BF01797193](https://doi.org/10.1007/BF01797193)  
   *Contribution:* The landmark discovery of human scalp electroencephalography and the identification of the 10 Hz Alpha rhythm ("Berger wave").
6. **Rhythms of the Brain**  
   *Buzsáki, G. (2006). Oxford University Press.* [DOI: 10.1093/acprof:oso/9780195301069.001.0001](https://doi.org/10.1093/acprof:oso/9780195301069.001.0001)  
   *Contribution:* The theoretical framework for self-organizing neuronal oscillations, cross-frequency coupling, and their functional roles in neural computation.
7. **What we can do and what we cannot do with fMRI**  
   *Logothetis, N. K. (2008). Nature, 453(7197), 869–878.* [DOI: 10.1038/nature06976](https://doi.org/10.1038/nature06976)  
   *Contribution:* Critical evaluation of the BOLD signal, proving fMRI reflects local post-synaptic dendritic activity rather than spiking output, and detailing hemodynamic temporal limits.

---

## Appendix A — Reproducibility (UV + Python)

To regenerate all nine publication-grade figures embedded throughout this review article and verify the numerical outputs, run the repository generation script using `uv`:

```bash
# Restore exact project virtual environment from uv.lock
uv sync

# Run the Week 1 figure generation script
uv run python generate_week1_plots.py
```

Generated image assets are saved directly into [`month 1/week 1/assets/`](file:///Volumes/1700%20APFS/PERSONAL_PROGRAMMING_PROJECTS/NeuroScience/month%201/week%201/assets).

---

## Appendix B — Knowledge Graph Connections

This review article serves as the foundational root of the repository's knowledge graph. It connects to the following Obsidian notes:

- **Central MOC Hub:** [[Knowledge-Map]]
- **Forward Weekly Review:** [[week-2-report|Week 2: Fundamentals of Neural Signal Processing]]
- **Atomic Concept Notes:**
  - [[eeg-sampling-rate]] — Discretization, $\Delta t$, and Nyquist bounds.
  - [[nyquist-criterion]] — Mathematical condition preventing aliasing distortion.
  - [[artifact-contamination]] — Non-neural ocular, muscular, and electrical noise.
