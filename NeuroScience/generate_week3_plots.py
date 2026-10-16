"""Generate all Week 3 figures, execute laboratories, and export empirical artifacts.

This script executes the complete Week 3 AI + Neuroscience learning pipeline:
1. Verifies cached PhysioNet EEGBCI data (Subjects 1-10, Runs 4, 8, 12).
2. Generates provenance manifest and SHA-256 validation table.
3. Preprocesses raw continuous EEG (1-35 Hz band-pass) per run.
4. Extracts event-aligned trials (t = 0.5s to 3.5s post-onset) for C3, Cz, C4.
5. Computes 15 features per trial (mean, var, p2p, alpha power, beta power).
6. Runs Subject-Dependent (LORO) and Subject-Independent (LOSO) evaluations.
7. Evaluates Dummy, Logistic Regression, Decision Tree, and Linear SVM models.
8. Produces all 12 publication-quality Seaborn figures at 180 DPI in month 1/week 3/assets/.
9. Executes standalone teaching laboratories and verifies reproducible stdout.
"""
from pathlib import Path
import contextlib
import io
import json
import hashlib
import urllib.request
import warnings
from importlib.resources import files
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, welch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, KFold
import mne

# Repository styling and paths
sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
    'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'grid.alpha': 0.22, 'axes.spines.top': False,
    'axes.spines.right': False,
})

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'month 1' / 'week 3' / 'assets'
DATA_DIR = ROOT / 'data'
CACHE_DIR = DATA_DIR / 'cache'
EEGMI_BASE_URL = 'https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0/'

SUBJECTS = list(range(1, 11))
RUNS = [4, 8, 12]
CHANNELS = ['C3', 'Cz', 'C4']
FS = 160.0
RANDOM_STATE = 42

def bandpass_filter(data, low_cut=1.0, high_cut=35.0, fs=160.0, order=4):
    """Zero-phase Butterworth band-pass filter applied per run."""
    nyq = 0.5 * fs
    b, a = butter(order, [low_cut / nyq, high_cut / nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)

def ensure_data_cached():
    """Verify all 30 EDF files exist and match official PhysioNet SHA-256 checksums."""
    target_dir = CACHE_DIR / 'MNE-eegbci-data' / 'files' / 'eegmmidb' / '1.0.0'
    registry = files('mne').joinpath('data', 'eegbci_checksums.txt')
    with open(registry) as f:
        checksums = dict(line.strip().split() for line in f if line.strip())
        
    manifest_records = []
    for sub in SUBJECTS:
        for run in RUNS:
            file_part = f'S{sub:03d}/S{sub:03d}R{run:02d}.edf'
            dest = target_dir / file_part
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                url = EEGMI_BASE_URL + file_part
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
                dest.write_bytes(data)
            
            content = dest.read_bytes()
            h = hashlib.sha256(content).hexdigest()
            expected_h = checksums.get(file_part)
            assert h == expected_h, f"Hash mismatch for {file_part}"
            
            manifest_records.append({
                'subject': sub,
                'run': run,
                'filename': dest.name,
                'relative_path': file_part,
                'size_bytes': len(content),
                'sha256': h,
                'checksum_verified': True,
                'source_url': EEGMI_BASE_URL + file_part,
                'sampling_rate_hz': FS,
                'task': 'left_vs_right_hand_motor_imagery'
            })
            
    manifest_path = DATA_DIR / 'provenance_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump({
            'dataset_name': 'PhysioNet EEG Motor Movement/Imagery Dataset',
            'version': '1.0.0',
            'doi': '10.13026/C28G6P',
            'license': 'Open Data Commons Attribution License v1.0 (ODC-By)',
            'cohort_subjects': SUBJECTS,
            'runs': RUNS,
            'channels_selected': CHANNELS,
            'sampling_rate_hz': FS,
            'files_count': len(manifest_records),
            'files': manifest_records
        }, f, indent=2)
    return target_dir

def extract_trials_and_features():
    """Extract 450 row-aligned trials and compute 15 features per trial."""
    target_dir = CACHE_DIR / 'MNE-eegbci-data' / 'files' / 'eegmmidb' / '1.0.0'
    trial_records = []
    
    for sub in SUBJECTS:
        for run in RUNS:
            edf_path = target_dir / f'S{sub:03d}' / f'S{sub:03d}R{run:02d}.edf'
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            mne.datasets.eegbci.standardize(raw)
            raw.pick(CHANNELS)
            
            assert raw.info['sfreq'] == FS
            data_uv = raw.get_data() * 1e6  # Convert V to uV
            filtered_uv = bandpass_filter(data_uv, 1.0, 35.0, FS, order=4)
            
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            t1_id = event_id.get('T1')
            t2_id = event_id.get('T2')
            
            for onset, _, eid in events:
                if eid in [t1_id, t2_id]:
                    lbl_str = 'LEFT_HAND' if eid == t1_id else 'RIGHT_HAND'
                    lbl_int = 0 if eid == t1_id else 1  # 1 = RIGHT_HAND (positive class)
                    
                    # Analysis window: 0.5s to 3.5s post-onset (duration = 3.0s, 480 samples)
                    start_samp = int(onset + 0.5 * FS)
                    end_samp = int(onset + 3.5 * FS)
                    assert 0 <= start_samp < end_samp <= raw.n_times, 'Truncated trial window'
                    epoch_filt = filtered_uv[:, start_samp:end_samp]
                    assert epoch_filt.shape == (len(CHANNELS), 480)
                    
                    rec = {
                        'subject': sub,
                        'run': run,
                        'onset_sample': int(onset),
                        'onset_seconds': float(onset / FS),
                        'window_start_sec': float(onset / FS + 0.5),
                        'window_end_sec': float(onset / FS + 3.5),
                        'duration_sec': 3.0,
                        'n_samples': 480,
                        'label': lbl_str,
                        'target': lbl_int
                    }
                    
                    for ch_idx, ch_name in enumerate(CHANNELS):
                        sig = epoch_filt[ch_idx]
                        rec[f'{ch_name}_mean'] = float(np.mean(sig))
                        rec[f'{ch_name}_var'] = float(np.var(sig))
                        rec[f'{ch_name}_p2p'] = float(np.ptp(sig))
                        
                        f, psd = welch(sig, fs=FS, nperseg=int(1.5 * FS), noverlap=int(0.75 * FS))
                        df_freq = f[1] - f[0]
                        alpha_pwr = np.sum(psd[(f >= 8.0) & (f <= 12.0)]) * df_freq
                        beta_pwr = np.sum(psd[(f >= 13.0) & (f <= 30.0)]) * df_freq
                        rec[f'{ch_name}_alpha'] = float(alpha_pwr)
                        rec[f'{ch_name}_beta'] = float(beta_pwr)
                        
                    trial_records.append(rec)
                    
    df = pd.DataFrame(trial_records)
    csv_path = DATA_DIR / 'week3_features.csv'
    df.to_csv(csv_path, index=False)
    
    # Feature schema definition
    feature_cols = [c for c in df.columns if any(c.startswith(ch) for ch in CHANNELS)]
    schema = {
        'total_trials': len(df),
        'positive_class': 'RIGHT_HAND (target=1)',
        'negative_class': 'LEFT_HAND (target=0)',
        'channels': CHANNELS,
        'feature_count': len(feature_cols),
        'feature_names': feature_cols,
        'feature_units': {
            'mean': 'microvolts (uV)',
            'var': 'microvolts squared (uV^2)',
            'p2p': 'peak-to-peak amplitude (uV)',
            'alpha': 'integrated Welch PSD 8-12 Hz (uV^2)',
            'beta': 'integrated Welch PSD 13-30 Hz (uV^2)'
        }
    }
    with open(DATA_DIR / 'feature_schema.json', 'w') as f:
        json.dump(schema, f, indent=2)
        
    return df

def run_ml_experiments(df):
    """Run Subject-Dependent and Subject-Independent evaluations and save predictions."""
    feature_cols = [c for c in df.columns if any(c.startswith(ch) for ch in CHANNELS)]
    logo = LeaveOneGroupOut()
    
    def get_models():
        return {
            'Dummy': DummyClassifier(strategy='most_frequent'),
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(solver='liblinear', C=1.0, random_state=RANDOM_STATE))
            ]),
            'Decision Tree': DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE),
            'Linear SVM': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', SVC(kernel='linear', C=1.0, random_state=RANDOM_STATE))
            ])
        }
        
    # 1. Subject-Dependent (Leave-One-Run-Out per subject)
    sdep_results = []
    sdep_preds_list = []
    for sub in SUBJECTS:
        sub_df = df[df['subject'] == sub].reset_index(drop=True)
        X_sub = sub_df[feature_cols].values
        y_sub = sub_df['target'].values
        runs_sub = sub_df['run'].values
        
        models = get_models()
        fold_preds = {m: np.zeros(len(y_sub), dtype=int) for m in models}
        
        for tr, te in logo.split(X_sub, y_sub, runs_sub):
            for m_name, model in models.items():
                model.fit(X_sub[tr], y_sub[tr])
                fold_preds[m_name][te] = model.predict(X_sub[te])
                
        for i in range(len(y_sub)):
            sdep_preds_list.append({
                'protocol': 'subject_dependent_loro',
                'subject': sub,
                'run': int(runs_sub[i]),
                'trial_idx': i,
                'y_true': int(y_sub[i]),
                'pred_dummy': int(fold_preds['Dummy'][i]),
                'pred_lr': int(fold_preds['Logistic Regression'][i]),
                'pred_dt': int(fold_preds['Decision Tree'][i]),
                'pred_svm': int(fold_preds['Linear SVM'][i])
            })
            
        for m_name in models:
            acc = accuracy_score(y_sub, fold_preds[m_name])
            f1 = f1_score(y_sub, fold_preds[m_name], zero_division=0)
            sdep_results.append({
                'protocol': 'subject_dependent',
                'subject': sub,
                'model': m_name,
                'accuracy': float(acc),
                'f1_score': float(f1)
            })

    # 2. Subject-Independent (Leave-One-Subject-Out across 10 subjects)
    X_all = df[feature_cols].values
    y_all = df['target'].values
    subs_all = df['subject'].values
    
    models = get_models()
    sind_preds = {m: np.zeros(len(y_all), dtype=int) for m in models}
    sind_scores_per_sub = {m: [] for m in models}
    
    for tr, te in logo.split(X_all, y_all, subs_all):
        test_sub = subs_all[te][0]
        for m_name, model in models.items():
            model.fit(X_all[tr], y_all[tr])
            preds = model.predict(X_all[te])
            sind_preds[m_name][te] = preds
            sind_scores_per_sub[m_name].append({
                'subject': int(test_sub),
                'accuracy': float(accuracy_score(y_all[te], preds)),
                'f1_score': float(f1_score(y_all[te], preds, zero_division=0))
            })
            
    for i in range(len(y_all)):
        sdep_preds_list.append({
            'protocol': 'subject_independent_loso',
            'subject': int(subs_all[i]),
            'run': int(df.iloc[i]['run']),
            'trial_idx': i,
            'y_true': int(y_all[i]),
            'pred_dummy': int(sind_preds['Dummy'][i]),
            'pred_lr': int(sind_preds['Logistic Regression'][i]),
            'pred_dt': int(sind_preds['Decision Tree'][i]),
            'pred_svm': int(sind_preds['Linear SVM'][i])
        })
        
    df_preds = pd.DataFrame(sdep_preds_list)
    df_preds.to_csv(DATA_DIR / 'week3_predictions.csv', index=False)
    
    # Consolidate metrics summary
    summary_metrics = {
        'subject_dependent': {},
        'subject_independent': {}
    }
    
    # Subject-dependent summary
    df_sdep = pd.DataFrame(sdep_results)
    for m_name in get_models():
        sub_accs = df_sdep[df_sdep['model'] == m_name]['accuracy'].values
        summary_metrics['subject_dependent'][m_name] = {
            'mean_subject_accuracy': float(np.mean(sub_accs)),
            'std_subject_accuracy': float(np.std(sub_accs)),
            'per_subject_accuracy': [float(x) for x in sub_accs]
        }
        
    # Subject-independent summary
    for m_name in get_models():
        preds = sind_preds[m_name]
        acc = accuracy_score(y_all, preds)
        prec = precision_score(y_all, preds, zero_division=0)
        rec = recall_score(y_all, preds, zero_division=0)
        f1 = f1_score(y_all, preds, zero_division=0)
        cm = confusion_matrix(y_all, preds).tolist()
        per_sub_accs = [s['accuracy'] for s in sind_scores_per_sub[m_name]]
        
        summary_metrics['subject_independent'][m_name] = {
            'pooled_accuracy': float(acc),
            'mean_subject_accuracy': float(np.mean(per_sub_accs)),
            'std_subject_accuracy': float(np.std(per_sub_accs)),
            'pooled_precision': float(prec),
            'pooled_recall': float(rec),
            'pooled_f1': float(f1),
            'confusion_matrix': cm,
            'per_subject_accuracy': per_sub_accs
        }
        
    with open(DATA_DIR / 'week3_metrics.json', 'w') as f:
        json.dump(summary_metrics, f, indent=2)
        
    return df, df_preds, summary_metrics

def save_fig(fig, filename, manifest):
    """Save figure with strict DPI and tight layout, recording to manifest."""
    fig.tight_layout(pad=1.5, rect=(0, 0, 1, 0.96))
    path = ASSETS / filename
    fig.savefig(path, dpi=180, metadata={'Software': 'NeuroScience Week 3'})
    plt.close(fig)
    manifest.append(path.relative_to(ROOT).as_posix())

def generate_all_plots(df, df_preds, summary_metrics):
    """Generate all 12 publication-grade figures for Week 3."""
    manifest = []
    target_dir = CACHE_DIR / 'MNE-eegbci-data' / 'files' / 'eegmmidb' / '1.0.0'
    
    # Event-aligned multichannel EEG
    edf_s1_r4 = target_dir / 'S001' / 'S001R04.edf'
    raw_s1 = mne.io.read_raw_edf(edf_s1_r4, preload=True, verbose=False)
    mne.datasets.eegbci.standardize(raw_s1)
    raw_s1.pick(CHANNELS)
    events, event_id = mne.events_from_annotations(raw_s1, verbose=False)
    t1_onsets = events[events[:, 2] == event_id['T1'], 0]
    t2_onsets = events[events[:, 2] == event_id['T2'], 0]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    t_axis = np.linspace(-0.5, 4.5, int(5.0 * FS), endpoint=False)
    colors = sns.color_palette("deep", 3)
    
    for ax_idx, (onsets, label_name, cue_name) in enumerate([
        (t1_onsets, 'Trial 0 (T1: Left Hand Motor Imagery)', 'T1 Cue'),
        (t2_onsets, 'Trial 1 (T2: Right Hand Motor Imagery)', 'T2 Cue')
    ]):
        ax = axes[ax_idx]
        onset = onsets[0]
        start_samp = int(onset - 0.5 * FS)
        end_samp = int(onset + 4.5 * FS)
        epoch_raw = raw_s1.get_data()[:, start_samp:end_samp] * 1e6
        
        for ch_idx, ch_name in enumerate(CHANNELS):
            ax.plot(t_axis, epoch_raw[ch_idx], label=ch_name, color=colors[ch_idx], lw=1.2)
            
        ax.axvline(0.0, color='red', linestyle='--', lw=1.5, label='Visual Cue (t=0.0s)')
        ax.axvspan(0.5, 3.5, color='orange', alpha=0.18, label='Analysis window [0.5, 3.5) s')
        duration = raw_s1.annotations.duration[np.isclose(raw_s1.annotations.onset, onset / FS)][0]
        ax.axvline(duration, color='gray', linestyle=':', lw=1.5, label=f'Annotation end ({duration:.1f} s)')
        ax.set(ylabel='Voltage (µV)', title=f'Subject 001, Run 04 — {label_name}')
        if ax_idx == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(0, -0.20), fontsize=9, ncol=3)
    axes[1].set_xlabel('Time Relative to Cue Onset (seconds)')
    fig.suptitle('Event-Aligned Multichannel Sensorimotor EEG', y=1.02, fontsize=15, weight='bold')
    save_fig(fig, 'event_aligned_eeg.png', manifest)
    
    # Real Raw vs Preprocessed Traces and PSDs
    s1_r4_data = raw_s1.get_data()[0] * 1e6  # C3 channel
    s1_r4_filt = bandpass_filter(s1_r4_data, 1.0, 35.0, FS, order=4)
    t_sec = np.linspace(0, 10, int(10 * FS), endpoint=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    axes[0].plot(t_sec, s1_r4_data[:len(t_sec)], color='gray', lw=1.0, label='Raw C3 (DC drift + wideband)')
    axes[0].plot(t_sec, s1_r4_filt[:len(t_sec)], color=colors[0], lw=1.3, label='Filtered C3 (1–35 Hz band-pass)')
    axes[0].set(xlabel='Continuous Time (seconds)', ylabel='Voltage (µV)', title='Time Domain Waveform (10s segment)')
    axes[0].legend(fontsize=9, loc='upper right')
    
    f_raw, p_raw = welch(s1_r4_data, fs=FS, nperseg=int(2 * FS), noverlap=int(FS))
    f_filt, p_filt = welch(s1_r4_filt, fs=FS, nperseg=int(2 * FS), noverlap=int(FS))
    axes[1].semilogy(f_raw, p_raw, color='gray', lw=1.2, label='Raw C3 PSD')
    axes[1].semilogy(f_filt, p_filt, color=colors[0], lw=1.5, label='Filtered C3 PSD')
    axes[1].axvspan(8, 12, color='green', alpha=0.15, label='Alpha (8–12 Hz)')
    axes[1].axvspan(13, 30, color='purple', alpha=0.15, label='Beta (13–30 Hz)')
    axes[1].set(xlabel='Frequency (Hz)', ylabel='PSD (µV²/Hz)', xlim=(0, 60), title='Power Spectral Density (0–60 Hz)')
    axes[1].legend(fontsize=9, loc='upper right')
    fig.suptitle('Real Preprocessing Transformation (Subject 1, Run 04, Electrode C3)', y=1.02, fontsize=15, weight='bold')
    save_fig(fig, 'raw_vs_preprocessed.png', manifest)

    # Trial-to-Feature-Matrix Transformation
    feature_cols = [c for c in df.columns if any(c.startswith(ch) for ch in CHANNELS)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={'width_ratios': [1.2, 1, 1.8]})
    
    trial_snip = df.iloc[0]
    axes[0].plot(np.arange(480) / FS, bandpass_filter(raw_s1.get_data() * 1e6)[:, int(trial_snip['onset_sample']+0.5*FS):int(trial_snip['onset_sample']+3.5*FS)].T)
    axes[0].set(xlabel='Window Time (s)', ylabel='Voltage (µV)', title='Filtered trial [3 channels × 480 samples]')
    axes[0].legend(CHANNELS, loc='upper right', fontsize=9)
    
    feats_15 = df.iloc[0][feature_cols].values
    # Scale each column across training trials, never across mixed-unit features.
    display_scaler = StandardScaler().fit(df.loc[(df['subject'] == 1) & df['run'].isin([4, 8]), feature_cols])
    norm_feats_15 = display_scaler.transform(df.iloc[[0]][feature_cols])[0]
    axes[1].barh(range(15), norm_feats_15, color=colors[2])
    axes[1].set_yticks(range(15))
    axes[1].set_yticklabels(feature_cols, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set(xlabel='Training-column z-score (unitless)', title=r'Feature Vector $x_i \in \mathbb{R}^{15}$')
    
    X_snip = df[feature_cols].iloc[:20].values
    X_snip_std = display_scaler.transform(df.iloc[:20][feature_cols])
    sns.heatmap(X_snip_std, ax=axes[2], cmap='vlag', center=0, vmin=-3, vmax=3, cbar_kws={'label': 'Training-column z-score (clipped ±3)'}, yticklabels=[f'Tr {i}' for i in range(20)])
    axes[2].set_xticks(np.arange(15) + 0.5)
    axes[2].set_xticklabels(feature_cols, rotation=90, fontsize=8)
    axes[2].set(ylabel='Observation Index', title='First 20 rows of X [450 × 15]')
    fig.suptitle('Representation Bridge — From Continuous Voltage to Supervised Matrix', y=1.02, fontsize=15, weight='bold')
    save_fig(fig, 'trial_to_feature_matrix.png', manifest)

    # Class Counts and Retained Counts
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    counts = df.groupby(['subject', 'label']).size().unstack()
    counts.plot(kind='bar', stacked=True, ax=axes[0], color=['#4c72b0', '#c44e52'])
    axes[0].set(xlabel='Subject Identifier', ylabel='Trial Count', title='Imagery Trials per Subject (Runs 4, 8, 12)')
    axes[0].legend(['LEFT_HAND (T1)', 'RIGHT_HAND (T2)'], fontsize=9)
    axes[0].axhline(45, color='gray', linestyle=':')
    
    retained_data = pd.DataFrame({
        'Status': ['Valid Imagery Trials', 'Rejected Boundary Trials'],
        'Count': [450, 0]
    })
    sns.barplot(data=retained_data, x='Status', y='Count', hue='Status', ax=axes[1], palette=['#55a868', '#dd8452'], legend=False)
    axes[1].set(ylabel='Total Observation Count', title='Trial Retention Audit (10-Subject Cohort)')
    for p in axes[1].patches:
        axes[1].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() + 5),
                         ha='center', va='bottom', fontsize=11, weight='bold')
    axes[1].set_ylim(0, 500)
    fig.suptitle('Experimental Label Distribution and Cohort Trial Balance', y=1.02, fontsize=15, weight='bold')
    save_fig(fig, 'class_counts_and_splits.png', manifest)

    # ECDF retains every observation and avoids negative-power KDE tails.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, channel in zip(axes, ['C3', 'C4']):
        sns.ecdfplot(data=df, x=f'{channel}_alpha', hue='label',
                     palette={'LEFT_HAND': '#4c72b0', 'RIGHT_HAND': '#c44e52'}, ax=ax)
        ax.set(xscale='log', xlabel=f'{channel} alpha power (µV²; log scale)',
               ylabel='Cumulative fraction of trials (unitless)',
               title=f'{channel}: class overlap across 450 trials', ylim=(0, 1.02))
        legend = ax.get_legend()
        if legend:
            legend.set_title('Cue label')
    fig.suptitle('Alpha power distributions: separation is limited', fontsize=16, weight='bold')
    save_fig(fig, 'feature_distributions.png', manifest)

    # Matched 2-Feature Decision Boundaries
    s1_df = df[df['subject'] == 1].reset_index(drop=True)
    X_2d = np.column_stack([
        np.log10(np.maximum(s1_df['C3_alpha'].values, 1e-3)),
        np.log10(np.maximum(s1_df['C4_alpha'].values, 1e-3))
    ])
    y_2d = s1_df['target'].values
    train_mask = s1_df['run'].isin([4, 8]).values
    test_mask = s1_df['run'] == 12
    
    X_tr_2d, y_tr_2d = X_2d[train_mask], y_2d[train_mask]
    X_te_2d, y_te_2d = X_2d[test_mask], y_2d[test_mask]
    
    lr_2d = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(solver='liblinear', C=1.0, random_state=RANDOM_STATE))])
    dt_2d = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=RANDOM_STATE)
    svm_2d = Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear', C=1.0, random_state=RANDOM_STATE))])
    
    lr_2d.fit(X_tr_2d, y_tr_2d)
    dt_2d.fit(X_tr_2d, y_tr_2d)
    svm_2d.fit(X_tr_2d, y_tr_2d)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    x_min, x_max = X_2d[:, 0].min() - 0.2, X_2d[:, 0].max() + 0.2
    y_min, y_max = X_2d[:, 1].min() - 0.2, X_2d[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    for ax, (model_name, model) in zip(axes, [
        ('Logistic Regression', lr_2d),
        ('Decision Tree (Depth=3)', dt_2d),
        ('Linear Support Vector Machine', svm_2d)
    ]):
        with np.errstate(all='ignore'):
            Z = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], alpha=0.20, colors=['#4c72b0', '#c44e52'])
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=1.5)
        
        for cls_val, m, col, lbl in [(0, 'o', '#4c72b0', 'Train Left (R4/R8)'), (1, 's', '#c44e52', 'Train Right (R4/R8)')]:
            idx = (y_tr_2d == cls_val)
            ax.scatter(X_tr_2d[idx, 0], X_tr_2d[idx, 1], marker=m, c=col, edgecolors='k', s=55, label=lbl)
            
        for cls_val, m, col, lbl in [(0, 'P', '#2b5c8f', 'Held-out Left (R12)'), (1, '*', '#8b1e22', 'Held-out Right (R12)')]:
            idx = (y_te_2d == cls_val)
            ax.scatter(X_te_2d[idx, 0], X_te_2d[idx, 1], marker=m, c=col, edgecolors='white', s=110, lw=1.2, label=lbl)
            
        acc_te = accuracy_score(y_te_2d, model.predict(X_te_2d))
        ax.set(xlabel=r'$\log_{10}(\mathrm{C3\ \alpha\ Power} / 1\ \mu\mathrm{V}^2)$', title=f'{model_name}\nHeld-Out Acc: {acc_te:.2f}')
        if ax == axes[0]:
            ax.set_ylabel(r'$\log_{10}(\mathrm{C4\ \alpha\ Power} / 1\ \mu\mathrm{V}^2)$')
            ax.legend(fontsize=7, loc='lower left')
            
    fig.suptitle('Matched Two-Feature Decision Boundaries (Subject 1, Train: Runs 4/8, Test: Run 12)', y=1.02, fontsize=15, weight='bold')
    save_fig(fig, 'matched_decision_boundaries.png', manifest)

    # Held-Out Confusion Matrices (LOSO Protocol)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    model_keys = ['Dummy', 'Logistic Regression', 'Decision Tree', 'Linear SVM']
    
    for ax, m_key in zip(axes.flat, model_keys):
        cm = np.array(summary_metrics['subject_independent'][m_key]['confusion_matrix'])
        acc = summary_metrics['subject_independent'][m_key]['pooled_accuracy']
        f1 = summary_metrics['subject_independent'][m_key]['pooled_f1']
        
        annot = np.array([[f"{cm[0,0]}\n({cm[0,0]/230:.1%})", f"{cm[0,1]}\n({cm[0,1]/230:.1%})"],
                          [f"{cm[1,0]}\n({cm[1,0]/220:.1%})", f"{cm[1,1]}\n({cm[1,1]/220:.1%})"]])
        sns.heatmap(cm / cm.sum(axis=1, keepdims=True), vmin=0, vmax=1, annot=annot, fmt='', cmap='Blues', cbar=False, ax=ax, annot_kws={'fontsize': 13})
        ax.set(title=f'{m_key}\nAcc = {acc:.3f}, F1 = {f1:.3f}',
               xlabel='Predicted Label (0 = Left, 1 = Right)',
               ylabel='True Label (0 = Left, 1 = Right)')
        ax.set_xticklabels(['LEFT', 'RIGHT'])
        ax.set_yticklabels(['LEFT', 'RIGHT'])
        
    fig.suptitle('Subject-Independent Held-Out Confusion Matrices (LOSO, N = 450 Trials)', y=1.02, fontsize=15, weight='bold')
    save_fig(fig, 'confusion_matrices.png', manifest)

    # Metric Comparison Against Dummy Baseline
    metrics_records = []
    for m_key in model_keys:
        m_dict = summary_metrics['subject_independent'][m_key]
        metrics_records.append({'Model': m_key, 'Metric': 'Accuracy', 'Score': m_dict['pooled_accuracy']})
        metrics_records.append({'Model': m_key, 'Metric': 'Precision', 'Score': m_dict['pooled_precision']})
        metrics_records.append({'Model': m_key, 'Metric': 'Recall', 'Score': m_dict['pooled_recall']})
        metrics_records.append({'Model': m_key, 'Metric': 'F1-Score', 'Score': m_dict['pooled_f1']})
        
    df_plot_metrics = pd.DataFrame(metrics_records)
    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
    model_palette = dict(zip(model_keys, sns.color_palette('deep', 4)))
    for ax, metric in zip(axes, ['Accuracy', 'Precision', 'Recall', 'F1-Score']):
        subset = df_plot_metrics[df_plot_metrics['Metric'] == metric]
        sns.barplot(data=subset, x='Model', y='Score', hue='Model', palette=model_palette,
                    legend=False, ax=ax)
        ax.set(title=metric, xlabel='', ylabel='Score (unitless)', ylim=(0, 1))
        ax.set_xticks(range(4), ['Dummy', 'Logistic', 'Tree', 'SVM'], rotation=40)
        for i, score in enumerate(subset['Score']):
            ax.text(i, score + .025, f'{score:.3f}', ha='center', fontsize=9)
        if metric == 'Accuracy':
            baseline = summary_metrics['subject_independent']['Dummy']['pooled_accuracy']
            ax.axhline(baseline, color='#555555', linestyle=':', linewidth=1.5)
            ax.text(0.02, .94, f'Dummy accuracy: {baseline:.1%}', transform=ax.transAxes, fontsize=9)
    fig.suptitle('LOSO benchmark: inspect accuracy and positive-class errors together', fontsize=16, weight='bold')
    save_fig(fig, 'metric_comparison.png', manifest)

    # Inner validation holds out subjects, matching the outer LOSO question.
    depths = np.arange(1, 11)
    outer = df[df['subject'] != 1]
    X_outer, y_outer = outer[feature_cols].values, outer['target'].values
    groups = outer['subject'].values
    train_scores, validation_scores = [], []
    for depth in depths:
        tr_scores, va_scores = [], []
        for tr, va in LeaveOneGroupOut().split(X_outer, y_outer, groups):
            tree = DecisionTreeClassifier(max_depth=int(depth), min_samples_leaf=2, random_state=RANDOM_STATE)
            tree.fit(X_outer[tr], y_outer[tr])
            tr_scores.append(accuracy_score(y_outer[tr], tree.predict(X_outer[tr])))
            va_scores.append(accuracy_score(y_outer[va], tree.predict(X_outer[va])))
        train_scores.append(tr_scores)
        validation_scores.append(va_scores)
    tr_mean = np.mean(train_scores, axis=1)
    va_mean = np.mean(validation_scores, axis=1)
    va_sd = np.std(validation_scores, axis=1)
    best = int(np.argmax(va_mean))
    curve = {'depths': depths.tolist(), 'mean_training_accuracy': tr_mean.tolist(),
             'mean_validation_accuracy': va_mean.tolist(), 'std_validation_accuracy': va_sd.tolist(),
             'best_depth': int(depths[best]), 'protocol': 'inner_LOSO_S002_to_S010',
             'min_samples_leaf': 2}
    (DATA_DIR / 'week3_tree_validation.json').write_text(json.dumps(curve, indent=2) + '\n')
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(depths, tr_mean, 'o-', color='#4c72b0', label='Training folds: mean')
    ax.plot(depths, va_mean, 's--', color='#c44e52', label='Held-out subjects: mean')
    ax.fill_between(depths, va_mean-va_sd, va_mean+va_sd, color='#c44e52', alpha=.12,
                    label='Validation ±1 subject SD (not a CI)')
    ax.axvline(depths[best], color='#555555', linestyle=':', label=f'Highest inner mean: depth {depths[best]}')
    ax.set(xlabel='Maximum tree depth (unitless)', ylabel='Accuracy (unitless)',
           title='Depth study: S002–S010 only; S001 excluded', xticks=depths, ylim=(0, 1.02))
    ax.legend(loc='upper left', fontsize=9)
    fig.suptitle('Fitting the training data does not guarantee subject transfer', fontsize=16, weight='bold')
    save_fig(fig, 'tree_depth_overfitting.png', manifest)

    # Colors, annotations and folds all share one source of truth.
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2.5]})
    for ax, size, columns, title in [
        (axes[0], 3, ['Run 04', 'Run 08', 'Run 12'], 'A. LORO: train on 30 trials; test on 15 trials from the same person'),
        (axes[1], 10, [f'S{i:03d}' for i in SUBJECTS], 'B. LOSO: train on 405 trials; test on 45 trials from a new person')
    ]:
        grid = np.eye(size, dtype=int)
        annot = np.where(grid, 'TEST', 'train')
        sns.heatmap(grid, cmap=ListedColormap(['#dce7f3', '#c44e52']), vmin=0, vmax=1,
                    ax=ax, annot=annot, fmt='', cbar=False, linewidths=2, linecolor='white',
                    annot_kws={'fontsize': 10}, xticklabels=columns,
                    yticklabels=[f'Fold {i+1}' for i in range(size)])
        ax.set(title=title, xlabel='Held-out grouping unit', ylabel='Evaluation fold')
        ax.tick_params(axis='y', rotation=0)
    fig.legend(handles=[Patch(facecolor='#dce7f3', label='Training group'),
                        Patch(facecolor='#c44e52', label='Held-out group')],
               loc='upper center', bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False)
    save_fig(fig, 'evaluation_protocols.png', manifest)

    # Per-Subject Performance Distribution
    sdep_svm_accs = [summary_metrics['subject_dependent']['Linear SVM']['per_subject_accuracy'][s-1] for s in SUBJECTS]
    sind_svm_accs = summary_metrics['subject_independent']['Linear SVM']['per_subject_accuracy']
    
    df_sub_perf = pd.DataFrame({
        'Subject': [f'S{s:02d}' for s in SUBJECTS] * 2,
        'Accuracy': sdep_svm_accs + sind_svm_accs,
        'Protocol': ['Subject-Dependent (LORO)'] * 10 + ['Subject-Independent (LOSO)'] * 10
    })
    
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df_sub_perf, x='Subject', y='Accuracy', hue='Protocol', ax=ax, palette=['#4c72b0', '#dd8452'])
    ax.axhline(0.50, color='gray', linestyle=':', label='Fair-guess expectation (50%)')
    ax.axhline(np.mean(sdep_svm_accs), color='#4c72b0', linestyle='--', lw=1.2, label=f'Mean LORO: {np.mean(sdep_svm_accs):.3f}')
    ax.axhline(np.mean(sind_svm_accs), color='#dd8452', linestyle='--', lw=1.2, label=f'Mean LOSO: {np.mean(sind_svm_accs):.3f}')
    ax.set(ylim=(0, 1), ylabel='Held-Out Accuracy', title='Linear SVM Generalization across Fixed 10-Subject Cohort')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    fig.suptitle('Biological Heterogeneity across Individuals (N = 10 Subjects)', y=1.02, fontsize=15, weight='bold')
    save_fig(fig, 'subject_generalization.png', manifest)

    # Representative Correctly and Incorrectly Classified Trials
    sub7_df = df_preds[(df_preds['protocol'] == 'subject_independent_loso') & (df_preds['subject'] == 7)].copy()
    
    tp_idx = int(sub7_df[(sub7_df['y_true'] == 1) & (sub7_df['pred_svm'] == 1)].iloc[0]['trial_idx'])
    tn_idx = int(sub7_df[(sub7_df['y_true'] == 0) & (sub7_df['pred_svm'] == 0)].iloc[0]['trial_idx'])
    fp_idx = int(sub7_df[(sub7_df['y_true'] == 0) & (sub7_df['pred_svm'] == 1)].iloc[0]['trial_idx'])
    fn_idx = int(sub7_df[(sub7_df['y_true'] == 1) & (sub7_df['pred_svm'] == 0)].iloc[0]['trial_idx'])
    
    exemplars = [
        ('True Positive (Right Hand Correct)', tp_idx, '#55a868'),
        ('True Negative (Left Hand Correct)', tn_idx, '#4c72b0'),
        ('False Positive (Left Classified as Right)', fp_idx, '#c44e52'),
        ('False Negative (Right Classified as Left)', fn_idx, '#dd8452')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    t_win = 0.5 + np.arange(480) / FS
    
    for ax, (title, tr_idx, col) in zip(axes.flat, exemplars):
        tr_info = df.iloc[tr_idx]
        sub = int(tr_info['subject'])
        run = int(tr_info['run'])
        onset = int(tr_info['onset_sample'])
        c3_a = float(tr_info['C3_alpha'])
        c4_a = float(tr_info['C4_alpha'])
        
        edf_path = target_dir / f'S{sub:03d}' / f'S{sub:03d}R{run:02d}.edf'
        raw_ex = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        mne.datasets.eegbci.standardize(raw_ex)
        raw_ex.pick(CHANNELS)
        filt_ex = bandpass_filter(raw_ex.get_data() * 1e6, 1.0, 35.0, FS)
        
        start_samp = int(onset + 0.5 * FS)
        end_samp = int(onset + 3.5 * FS)
        epoch = filt_ex[:, start_samp:end_samp]
        
        ax.plot(t_win, epoch[0], label='C3 (Left)', lw=1.2, color='#4c72b0')
        ax.plot(t_win, epoch[2], label='C4 (Right)', lw=1.2, color='#c44e52')
        ax.set(title=f'{title}\nS{sub:02d} R{run:02d} (C3 Alpha={c3_a:.1f}, C4 Alpha={c4_a:.1f} µV²)',
               ylabel='Voltage (µV)')
        if ax == axes[0, 0]:
            ax.legend(fontsize=9, loc='upper right')
    axes[1, 0].set_xlabel('Time within Trial Window (seconds)')
    axes[1, 1].set_xlabel('Time within Trial Window (seconds)')
    fig.suptitle('Empirical Error Analysis — Held-Out Single-Trial Exemplars (Subject 07)', y=1.02, fontsize=15, weight='bold')
    save_fig(fig, 'classified_trial_exemplars.png', manifest)
    
    print(f"Generated {len(manifest)} figures successfully in {ASSETS}")
    for filename in manifest:
        print(filename)
    return manifest

LABS = {
'w3d1_features': '''import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, welch

ROOT = Path.cwd()
edf_path = ROOT / 'data/cache/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R04.edf'
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
mne.datasets.eegbci.standardize(raw)
raw.pick(['C3', 'Cz', 'C4'])
fs = raw.info['sfreq']

# Zero-phase Butterworth 1.0-35.0 Hz filter applied to continuous run
nyq = 0.5 * fs
b, a = butter(4, [1.0/nyq, 35.0/nyq], btype='band')
data_filt = filtfilt(b, a, raw.get_data() * 1e6, axis=-1)

# Extract event annotations
events, event_id = mne.events_from_annotations(raw, verbose=False)
t1_id, t2_id = event_id['T1'], event_id['T2']

rows = []
for onset, _, eid in events:
    if eid in [t1_id, t2_id]:
        lbl = 'LEFT_HAND' if eid == t1_id else 'RIGHT_HAND'
        target = 0 if eid == t1_id else 1
        # Window: 0.5s to 3.5s post-onset (3.0s, 480 samples)
        start_samp = int(onset + 0.5 * fs)
        end_samp = int(onset + 3.5 * fs)
        epoch = data_filt[:, start_samp:end_samp]
        
        row = {'label': lbl, 'target': target}
        for ch_idx, ch_name in enumerate(['C3', 'Cz', 'C4']):
            sig = epoch[ch_idx]
            row[f'{ch_name}_mean'] = float(np.mean(sig))
            row[f'{ch_name}_var'] = float(np.var(sig))
            row[f'{ch_name}_p2p'] = float(np.ptp(sig))
            f, psd = welch(sig, fs=fs, nperseg=int(1.5*fs), noverlap=int(0.75*fs))
            df = f[1] - f[0]
            row[f'{ch_name}_alpha'] = float(np.sum(psd[(f >= 8) & (f <= 12)]) * df)
            row[f'{ch_name}_beta'] = float(np.sum(psd[(f >= 13) & (f <= 30)]) * df)
        rows.append(row)

df_run4 = pd.DataFrame(rows)
feat_cols = [c for c in df_run4.columns if c not in ['label', 'target']]
X = df_run4[feat_cols].values
y = df_run4['target'].values
print(f"Extracted Run 4 trials: {len(df_run4)}")
print(f"Feature matrix X shape: {X.shape}; Label vector y shape: {y.shape}")
print(f"Class counts: Left={np.sum(y == 0)}, Right={np.sum(y == 1)}")
print(f"First trial C3 features [mean, var, p2p, alpha, beta]: {X[0, :5].round(2)}")
''',

'w3d2_logistic_regression': '''import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path.cwd()
df = pd.read_csv(ROOT / 'data/week3_features.csv')
s1_df = df[df['subject'] == 1].reset_index(drop=True)

feat_cols = [c for c in s1_df.columns if any(c.startswith(ch) for ch in ['C3', 'Cz', 'C4'])]
train_mask = s1_df['run'].isin([4, 8]).values
test_mask = (s1_df['run'] == 12).values

X_train, y_train = s1_df.loc[train_mask, feat_cols].values, s1_df.loc[train_mask, 'target'].values
X_test, y_test = s1_df.loc[test_mask, feat_cols].values, s1_df.loc[test_mask, 'target'].values

# Baseline: Dummy majority classifier
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)

# Pipeline: Training-only StandardScaler + L2-regularized Logistic Regression
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(solver='liblinear', C=1.0, random_state=42))
])
lr_pipe.fit(X_train, y_train)
lr_pred = lr_pipe.predict(X_test)
lr_probs = lr_pipe.predict_proba(X_test)[:, 1]

print(f"Subject 1 (Train: Runs 4/8 [N={len(y_train)}], Test: Run 12 [N={len(y_test)}])")
print(f"Dummy Classifier  — Accuracy: {accuracy_score(y_test, dummy_pred):.3f}")
print(f"Logistic Reg      — Accuracy: {accuracy_score(y_test, lr_pred):.3f}, Precision: {precision_score(y_test, lr_pred, zero_division=0):.3f}, Recall: {recall_score(y_test, lr_pred, zero_division=0):.3f}, F1: {f1_score(y_test, lr_pred, zero_division=0):.3f}")
print("Confusion Matrix (y_true on rows, y_pred on cols):")
print(confusion_matrix(y_test, lr_pred))
print(f"Sample held-out probabilities P(Right): {lr_probs[:5].round(3)}")
''',

'w3d3_trees_and_svm': '''import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

ROOT = Path.cwd()
df = pd.read_csv(ROOT / 'data/week3_features.csv')
s1_df = df[df['subject'] == 1].reset_index(drop=True)
feat_cols = [c for c in s1_df.columns if any(c.startswith(ch) for ch in ['C3', 'Cz', 'C4'])]

train_mask = s1_df['run'].isin([4, 8]).values
test_mask = (s1_df['run'] == 12).values
X_train, y_train = s1_df.loc[train_mask, feat_cols].values, s1_df.loc[train_mask, 'target'].values
X_test, y_test = s1_df.loc[test_mask, feat_cols].values, s1_df.loc[test_mask, 'target'].values

# Decision Tree (bounded depth)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)

# Linear SVM Pipeline
svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='linear', C=1.0, random_state=42))
])
svm_pipe.fit(X_train, y_train)
svm_pred = svm_pipe.predict(X_test)

print(f"Decision Tree (Depth=3) — Test Acc: {accuracy_score(y_test, tree_pred):.3f}, F1: {f1_score(y_test, tree_pred, zero_division=0):.3f}")
print(f"Linear SVM (C=1.0)      — Test Acc: {accuracy_score(y_test, svm_pred):.3f}, F1: {f1_score(y_test, svm_pred, zero_division=0):.3f}")
print(f"SVM number of support vectors: {svm_pipe.named_steps['clf'].n_support_.sum()} / {len(y_train)} training trials")

# Internal 5-fold CV on training runs to evaluate tree depth (1 to 5) without test leakage
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print("\\nInternal 5-fold CV on Training Folds:")
for d in range(1, 6):
    tr_accs, val_accs = [], []
    for k_tr, k_va in kf.split(X_train, y_train):
        m = DecisionTreeClassifier(max_depth=d, min_samples_leaf=2, random_state=42)
        m.fit(X_train[k_tr], y_train[k_tr])
        tr_accs.append(accuracy_score(y_train[k_tr], m.predict(X_train[k_tr])))
        val_accs.append(accuracy_score(y_train[k_va], m.predict(X_train[k_va])))
    print(f"  Depth {d}: Train Acc = {np.mean(tr_accs):.3f}, Val Acc = {np.mean(val_accs):.3f}")
''',

'w3d4_metrics_and_leakage': '''import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Part 1: Worked Confusion Matrix Derivation
# Constructed teaching example (15 trials, 7 Left, 8 Right); not model predictions
y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1]) # 6 of 8 Right correct, 4 of 7 Left correct

tn = np.sum((y_true == 0) & (y_pred == 0))
fp = np.sum((y_true == 0) & (y_pred == 1))
fn = np.sum((y_true == 1) & (y_pred == 0))
tp = np.sum((y_true == 1) & (y_pred == 1))

acc_manual = (tp + tn) / (tp + tn + fp + fn)
prec_manual = tp / (tp + fp)
rec_manual = tp / (tp + fn)
f1_manual = 2 * (prec_manual * rec_manual) / (prec_manual + rec_manual)

print("Manual Metric Derivation:")
print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"  Accuracy:  {acc_manual:.4f} (Library: {accuracy_score(y_true, y_pred):.4f})")
print(f"  Precision: {prec_manual:.4f} (Library: {precision_score(y_true, y_pred):.4f})")
print(f"  Recall:    {rec_manual:.4f} (Library: {recall_score(y_true, y_pred):.4f})")
print(f"  F1-Score:  {f1_manual:.4f} (Library: {f1_score(y_true, y_pred):.4f})")

# Part 2: Hypothetical Seizure Imbalance Demonstration (Accuracy Paradox)
N = 1000
y_imbalanced_true = np.zeros(N, dtype=int)
y_imbalanced_true[:10] = 1 # 10 seizure windows (1%), 990 non-seizure windows (99%)

# Trivial majority classifier: predicts all non-seizure (0)
y_trivial_pred = np.zeros(N, dtype=int)
cm_imb = confusion_matrix(y_imbalanced_true, y_trivial_pred)
acc_imb = accuracy_score(y_imbalanced_true, y_trivial_pred)
rec_imb = recall_score(y_imbalanced_true, y_trivial_pred, zero_division=0)
f1_imb = f1_score(y_imbalanced_true, y_trivial_pred, zero_division=0)

print("\\nHypothetical Seizure Imbalance Demonstration (10 seizures, 990 background):")
print(f"  Trivial All-Negative Accuracy: {acc_imb:.1%}  <-- Deceptively high!")
print(f"  Clinical Recall (Sensitivity): {rec_imb:.1%}  <-- Completely useless (missed 100% of seizures)!")
print(f"  Clinical F1-Score:             {f1_imb:.4f}")
print(f"  Confusion Matrix:\\n{cm_imb}")
''',

'w3d5_subject_generalization': '''import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path.cwd()
df = pd.read_csv(ROOT / 'data/week3_features.csv')
feat_cols = [c for c in df.columns if any(c.startswith(ch) for ch in ['C3', 'Cz', 'C4'])]

logo = LeaveOneGroupOut()
X_all = df[feat_cols].values
y_all = df['target'].values
groups_sub = df['subject'].values

# Subject-Independent Leave-One-Subject-Out (LOSO) across all 10 subjects
svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='linear', C=1.0, random_state=42))
])
dummy = DummyClassifier(strategy='most_frequent')

svm_preds = np.zeros(len(y_all), dtype=int)
dummy_preds = np.zeros(len(y_all), dtype=int)
sub_accs_svm = []
sub_accs_dum = []

for tr, te in logo.split(X_all, y_all, groups_sub):
    sub_id = groups_sub[te][0]
    svm_pipe.fit(X_all[tr], y_all[tr])
    dummy.fit(X_all[tr], y_all[tr])
    
    p_svm = svm_pipe.predict(X_all[te])
    p_dum = dummy.predict(X_all[te])
    
    svm_preds[te] = p_svm
    dummy_preds[te] = p_dum
    
    acc_s = accuracy_score(y_all[te], p_svm)
    acc_d = accuracy_score(y_all[te], p_dum)
    sub_accs_svm.append(acc_s)
    sub_accs_dum.append(acc_d)
    print(f"  Subject S{sub_id:02d} Held-Out: Dummy Acc = {acc_d:.3f}, Linear SVM Acc = {acc_s:.3f}")

print("\\n--- Subject-Independent (LOSO) 10-Subject Benchmark Summary ---")
print(f"Dummy Classifier : Pooled Acc = {accuracy_score(y_all, dummy_preds):.3f}, Mean Sub Acc = {np.mean(sub_accs_dum):.3f}")
print(f"Linear SVM       : Pooled Acc = {accuracy_score(y_all, svm_preds):.3f}, Mean Sub Acc = {np.mean(sub_accs_svm):.3f}, F1 = {f1_score(y_all, svm_preds):.3f}")
print(f"Per-Subject Accuracies: {[round(x, 3) for x in sub_accs_svm]}")
'''
}

def run_labs():
    """Execute all laboratory blocks and capture their exact output."""
    results = {}
    print("\n--- Executing Verifiable Teaching Laboratories ---")
    for name, source in LABS.items():
        namespace = {}
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            exec(compile(source, f'<{name}>', 'exec'), namespace)
        out = capture.getvalue().strip()
        results[name] = (namespace, out)
        print(f"\n[{name}] captured successfully ({len(out)} chars stdout).")
    return results

def main():
    print("Step 1: Verifying cached dataset...")
    ensure_data_cached()
    print("Step 2: Extracting row-aligned trials and computing features...")
    df = extract_trials_and_features()
    print(f"Extracted {len(df)} trials across {df['subject'].nunique()} subjects.")
    print("Step 3: Running machine learning protocols...")
    df, df_preds, summary_metrics = run_ml_experiments(df)
    print("Step 4: Generating publication figures...")
    manifest = generate_all_plots(df, df_preds, summary_metrics)
    print("Step 5: Executing teaching laboratories...")
    lab_results = run_labs()
    
    # Save captured lab outputs to JSON
    lab_outputs = {name: out for name, (_, out) in lab_results.items()}
    with open(DATA_DIR / 'week3_lab_outputs.json', 'w') as f:
        json.dump(lab_outputs, f, indent=2)
        
    print("\nAll tasks completed successfully!")

if __name__ == '__main__':
    main()
