"""Regenerate all Week 2 figures and execute the report's exact teaching examples."""
from pathlib import Path
import contextlib
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, freqz, welch

sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams.update({'figure.dpi': 150, 'savefig.bbox': 'tight'})
ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'month 1/week 2/assets'

# These standalone source blocks are embedded verbatim in the report.
LABS = {
'signals': '''import numpy as np
import matplotlib.pyplot as plt

# 1. Define the time axis: 2 seconds at 250 Hz
fs = 250
t = np.linspace(0, 2, 2 * fs, endpoint=False)

# 2. Signal A: 10 Hz Alpha, 5 uV peak
signal_a = 5 * np.sin(2 * np.pi * 10 * t)

# 3. Signal B: add random measurement noise (2.5 uV SD)
np.random.seed(42)
random_noise = np.random.normal(0, 2.5, len(t))
signal_b = signal_a + random_noise

# 4. Signal C: add 1 Hz drift and 50 Hz electrical interference
slow_drift = 8 * np.sin(2 * np.pi * 1 * t)
line_noise = 3 * np.sin(2 * np.pi * 50 * t)
signal_c = signal_a + slow_drift + line_noise + random_noise
print(f"Shape: {signal_c.shape}; sample interval: {1000/fs:.1f} ms")
print(f"Variance A/B/C (uV^2): {np.var(signal_a):.2f}, {np.var(signal_b):.2f}, {np.var(signal_c):.2f}")
''',
'sampling': '''import numpy as np
import matplotlib.pyplot as plt

# 1. Dense approximation of continuous reality: 10 Hz Alpha
f_signal = 10
duration = 1.0
t_continuous = np.linspace(0, duration, 1000)
signal_continuous = np.sin(2 * np.pi * f_signal * t_continuous)

# 2. Adequate sampling: 50 Hz, above 20 Hz
fs_good = 50
t_good = np.linspace(0, duration, int(fs_good * duration), endpoint=False)
signal_good = np.sin(2 * np.pi * f_signal * t_good)

# 3. Inadequate sampling: 12 Hz, below 20 Hz
fs_bad = 12
t_bad = np.linspace(0, duration, int(fs_bad * duration), endpoint=False)
signal_bad = np.sin(2 * np.pi * f_signal * t_bad)
print(f"Good/bad samples: {len(t_good)}/{len(t_bad)}")
print(f"Good samples per cycle: {fs_good/f_signal:.1f}")
print(f"Alias frequency: {abs(f_signal - fs_bad)} Hz")
print(f"Gamma to 70 Hz: Nyquist boundary = {2*70} Hz")
print(f"128 channels, 300 s, 200 Hz: {(128, 300*200)}")
''',
'artifacts': '''import numpy as np
import matplotlib.pyplot as plt

# 1. Underlying EEG: 4 seconds, 250 Hz; 10 Hz Alpha, 1 uV peak
np.random.seed(42)
fs = 250
t = np.linspace(0, 4, fs * 4, endpoint=False)
clean_eeg = np.sin(2 * np.pi * 10 * t)

# 2. Blink: low-frequency surge centered at 1.5 s
blink_artifact = np.zeros_like(t)
blink_artifact += 25 * np.exp(-((t - 1.5)**2) / 0.02)

# 3. Jaw-muscle contamination: broadband random burst, 2.5-3.0 s
muscle_artifact = np.zeros_like(t)
jaw_clench_indices = (t >= 2.5) & (t <= 3.0)
muscle_artifact[jaw_clench_indices] = np.random.normal(0, 8, np.sum(jaw_clench_indices))

# 4. Electrode movement: +20 uV baseline step at 3.5 s
movement_artifact = np.zeros_like(t)
movement_artifact[t >= 3.5] = 20

# 5. Electrical interference: 50 Hz, 5 uV peak
line_noise = 5 * np.sin(2 * np.pi * 50 * t)
observed_eeg = clean_eeg + blink_artifact + muscle_artifact + movement_artifact + line_noise
print(f"Blink peak (uV): {blink_artifact.max():.2f}")
print(f"Movement offset (uV): {movement_artifact[-1]:.2f}")
print(f"Observed shape: {observed_eeg.shape}")
''',
'filtering': '''import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 1. Contaminated teaching signal: Alpha + drift + line hum + noise
np.random.seed(42)
fs = 250
t = np.linspace(0, 4, fs * 4, endpoint=False)
clean_alpha = np.sin(2 * np.pi * 10 * t)
slow_drift = 5 * np.sin(2 * np.pi * 0.5 * t)
line_noise = 3 * np.sin(2 * np.pi * 50 * t)
high_freq_noise = np.random.normal(0, 1.5, len(t))
observed_eeg = clean_alpha + slow_drift + line_noise + high_freq_noise

# 2. Butterworth band-stop: 49-51 Hz (finite-width notch)
b_notch, a_notch = butter(4, [49/(fs/2), 51/(fs/2)], btype='bandstop')
eeg_notch = filtfilt(b_notch, a_notch, observed_eeg)

# 3. High-pass at 1 Hz: suppress the 0.5 Hz drift
b_hp, a_hp = butter(4, 1/(fs/2), btype='high')
eeg_hp = filtfilt(b_hp, a_hp, eeg_notch)

# 4. Low-pass at 30 Hz: suppress faster fluctuations
b_lp, a_lp = butter(4, 30/(fs/2), btype='low')
eeg_clean = filtfilt(b_lp, a_lp, eeg_hp)

# 5. Alternative: retain only Alpha, 8-12 Hz
b_bp, a_bp = butter(4, [8/(fs/2), 12/(fs/2)], btype='bandpass')
eeg_bandpass = filtfilt(b_bp, a_bp, observed_eeg)

# 6. Quantify recovery in the interior, 1-3 s; edges excluded
interior = (t >= 1) & (t < 3)
for name, data in [('Raw', observed_eeg), ('Notch', eeg_notch),
                   ('High-pass', eeg_hp), ('Low-pass', eeg_clean),
                   ('Alpha-only', eeg_bandpass)]:
    rmse = np.sqrt(np.mean((data[interior] - clean_alpha[interior])**2))
    print(f"{name} interior RMSE (uV): {rmse:.3f}")
''',
'features': '''import numpy as np
from scipy.signal import butter, filtfilt

# 1. Four seconds of synthetic EEG: 10 Hz Alpha + 20 Hz Beta
fs = 250
t = np.linspace(0, 4, fs * 4, endpoint=False)
clean_signal = 10 * np.sin(2 * np.pi * 10 * t) + 3 * np.sin(2 * np.pi * 20 * t)

# 2. Time-domain features; voltage in uV, variance in uV^2
feature_mean = np.mean(clean_signal)
feature_variance = np.var(clean_signal)
feature_p2p_amp = np.max(clean_signal) - np.min(clean_signal)

# 3. Estimate band power using variance after band-pass filtering
# Finite-window estimate: filter response and edges affect the result
def get_band_power(signal, sfreq, low_cut, high_cut):
    b, a = butter(4, [low_cut/(sfreq/2), high_cut/(sfreq/2)], btype='bandpass')
    band_passed_signal = filtfilt(b, a, signal)
    return np.var(band_passed_signal)

power_alpha = get_band_power(clean_signal, fs, 8, 12)
power_beta = get_band_power(clean_signal, fs, 13, 30)

# 4. Compress 1000 samples into 5 ordered features
ai_feature_vector = np.array([
    feature_mean, feature_variance, feature_p2p_amp, power_alpha, power_beta
])
print("Features [mean, variance, peak-to-peak, Alpha power, Beta power]:")
print(np.array2string(ai_feature_vector, precision=3, suppress_small=True))
print(f"Raw/vector shapes: {clean_signal.shape} -> {ai_feature_vector.shape}")
''',
'pipeline': '''import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 1. Biological simulation and digitization: 4 s at 250 Hz
np.random.seed(42)
fs = 250
duration = 4.0
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
# Alpha-dominant simulation, not a validated mental-state label
true_neural_signal = 12 * np.sin(2 * np.pi * 10 * t) + 2 * np.sin(2 * np.pi * 20 * t)

# 2. Contamination: drift, line hum, blink, and sensor noise
sweat_drift = 20 * np.sin(2 * np.pi * 0.2 * t)
line_noise = 8 * np.sin(2 * np.pi * 50 * t)
blink = 80 * np.exp(-((t - 2.0)**2) / 0.05)
sensor_noise = np.random.normal(0, 2, len(t))
raw_observed_eeg = true_neural_signal + sweat_drift + line_noise + blink + sensor_noise

# 3. Offline Butterworth filters from the supplied laboratory
def apply_bandpass(data, low_cut, high_cut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low_cut/nyq, high_cut/nyq], btype='band')
    return filtfilt(b, a, data)

def apply_notch(data, freq, fs):
    nyq = 0.5 * fs
    b, a = butter(4, [(freq-1)/nyq, (freq+1)/nyq], btype='bandstop')
    return filtfilt(b, a, data)

eeg_notch = apply_notch(raw_observed_eeg, 50, fs)
eeg_clean = apply_bandpass(eeg_notch, 1, 30, fs)

# 4. Time-domain features and estimated band powers
feat_mean = np.mean(eeg_clean)
feat_var = np.var(eeg_clean)
feat_amp = np.max(eeg_clean) - np.min(eeg_clean)
alpha_band = apply_bandpass(eeg_clean, 8, 12, fs)
feat_alpha_power = np.var(alpha_band)
beta_band = apply_bandpass(eeg_clean, 13, 30, fs)
feat_beta_power = np.var(beta_band)
ai_input_vector = np.array([feat_mean, feat_var, feat_amp, feat_alpha_power, feat_beta_power])

# 5. One window is one training row, not five training examples
ai_input_matrix = ai_input_vector.reshape(1, -1)
print("Features [mean, variance, peak-to-peak, Alpha power, Beta power]:")
print(np.array2string(ai_input_vector, precision=3, suppress_small=True))
print(f"Vector/matrix shapes: {ai_input_vector.shape}/{ai_input_matrix.shape}")
print(f"Filtered mean (uV): {feat_mean:.3f}")
'''
}

def run_labs():
    results = {}
    for name, source in LABS.items():
        namespace = {}
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            exec(compile(source, f'<{name}>', 'exec'), namespace)
        results[name] = (namespace, capture.getvalue().strip())
    return results

def save(fig, name, manifest):
    fig.tight_layout()
    path = ASSETS / f'{name}.png'
    fig.savefig(path, metadata={'Software': 'NeuroScience Week 2'})
    plt.close(fig)
    manifest.append(path.relative_to(ROOT).as_posix())

def traces(t, entries, title, manifest, name):
    fig, axes = plt.subplots(len(entries), 1, figsize=(12, 2.2*len(entries)), sharex=True, squeeze=False)
    for index, (ax, (label, values)) in enumerate(zip(axes[:, 0], entries)):
        sns.lineplot(x=t, y=values, ax=ax, linewidth=1, color=sns.color_palette()[index % 6])
        ax.set(title=label, ylabel='Voltage (µV)')
    axes[-1, 0].set_xlabel('Time (s)')
    fig.suptitle(title, y=1.01)
    save(fig, name, manifest)

def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    results = run_labs()
    manifest = []
    d = results['signals'][0]
    traces(d['t'], [(n, d[v]) for n, v in [('A: 10 Hz Alpha','signal_a'), ('B: Alpha + random noise','signal_b'), ('C: Alpha + drift + line hum + noise','signal_c')]], 'Signal depends on the scientific question', manifest, 'signal_contamination')
    # Temporal ordering, same values and distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, title, y in zip(axes, ['Original time order', 'Sorted values: original timestamps lost'], [d['signal_a'], np.sort(d['signal_a'])]):
        sns.lineplot(x=d['t'], y=y, ax=ax)
        ax.set(title=title, xlabel='Assigned time (s)', ylabel='Voltage (µV)')
    save(fig, 'temporal_order', manifest)
    d = results['sampling'][0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, label, tt, yy in zip(axes, ['50 Hz sampling: 5 points/cycle', '12 Hz sampling: 2 Hz alias'], [d['t_good'], d['t_bad']], [d['signal_good'], d['signal_bad']]):
        sns.lineplot(x=d['t_continuous'], y=d['signal_continuous'], ax=ax, color='0.7', label='10 Hz reference')
        ax.plot(tt, yy, 'o-', label='Samples')
        if len(tt)==12:
            ax.plot(d['t_continuous'], -np.sin(2*np.pi*2*d['t_continuous']), '--', label='Equivalent −2 Hz sine')
        ax.set(title=label, xlabel='Time (s)', ylabel='Voltage (µV)')
        ax.legend(fontsize=10)
    save(fig, 'aliasing', manifest)
    d = results['artifacts'][0]
    traces(d['t'], [(n,d[v]) for n,v in [('Clean 10 Hz Alpha (1 µV peak)','clean_eeg'),('Eye blink (25 µV peak)','blink_artifact'),('Muscle burst (broadband approximation)','muscle_artifact'),('Electrode movement (+20 µV step)','movement_artifact'),('50 Hz electrical interference','line_noise'),('Observed sum','observed_eeg')]], 'Four distinct sources of EEG contamination', manifest, 'artifact_components')
    d = results['filtering'][0]
    traces(d['t'], [(n,d[v]) for n,v in [('Raw: Alpha + 0.5 Hz drift + 50 Hz hum + noise','observed_eeg'),('After 49–51 Hz band-stop','eeg_notch'),('Then 1 Hz high-pass','eeg_hp'),('Then 30 Hz low-pass','eeg_clean'),('Alternative: 8–12 Hz Alpha only','eeg_bandpass')]], 'Filtering changes the surviving information', manifest, 'filter_stages')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    responses = [('Low-pass: 30 Hz',30,'low'),('High-pass: 1 Hz',1,'high'),('Band-pass: 8–12 Hz',[8,12],'bandpass'),('Band-stop: 49–51 Hz',[49,51],'bandstop')]
    for ax,(title,cut,kind) in zip(axes.flat,responses):
        b,a = butter(4,cut,btype=kind,fs=250)
        f,h = freqz(b,a,worN=8192,fs=250)
        sns.lineplot(x=f,y=np.abs(h)**2,ax=ax)
        ax.set(title=title, xlabel='Frequency (Hz)',ylabel='Amplitude gain',xlim=(0,65),ylim=(-.03,1.05))
    fig.suptitle('Forward-backward Butterworth filter responses', y=1.02)
    save(fig, 'filter_responses', manifest)
    # A 45 Hz cutoff is not proof that 50 Hz interference is negligible.
    fig,ax=plt.subplots(figsize=(10,4))
    b,a=butter(4,[.5,45],btype='bandpass',fs=250)
    f,h=freqz(b,a,worN=8192,fs=250)
    ax.plot(f,20*np.log10(np.maximum(np.abs(h)**2,1e-12)))
    f50,h50=freqz(b,a,worN=np.array([50.]),fs=250)
    gain50=float(np.abs(h50[0])**2)
    ax.axvline(45,color='0.5',linestyle='--',label='45 Hz cutoff')
    ax.scatter([50],[20*np.log10(gain50)],color='red',zorder=5,label=f'50 Hz: {gain50:.3f} amplitude gain')
    ax.set(title='0.5–45 Hz Butterworth, forward + backward',xlabel='Frequency (Hz)',ylabel='Amplitude gain (dB)',xlim=(30,65),ylim=(-35,1))
    ax.legend(fontsize=11)
    save(fig, 'mains_attenuation', manifest)
    d=results['features'][0]
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    mask=d['t']<.5
    sns.lineplot(x=d['t'][mask],y=d['clean_signal'][mask],ax=axes[0])
    axes[0].set(title='Time domain: summed Alpha and Beta',xlabel='Time (s)',ylabel='Voltage (µV)')
    f,p=welch(d['clean_signal'],fs=250,nperseg=1000,window='boxcar',detrend=False)
    sns.lineplot(x=f,y=p,ax=axes[1])
    axes[1].set(title='Frequency domain: power spectral density',xlabel='Frequency (Hz)',ylabel='PSD (µV²/Hz)',xlim=(0,35))
    save(fig,'time_frequency',manifest)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    labels=['Alpha (8–12 Hz)','Beta (13–30 Hz)']
    vals=[d['power_alpha'],d['power_beta']]
    sns.barplot(x=labels,y=vals,ax=axes[0],color=sns.color_palette()[0])
    axes[0].set(title='Finite-window band-pass variance',ylabel='Estimated band power (µV²)')
    for i,v in enumerate(vals): axes[0].text(i,v+.7,f'{v:.3f}',ha='center',fontsize=12)
    axes[0].set_ylim(0,58)
    sns.barplot(x=['Raw window','Feature vector'],y=[1000,5],ax=axes[1],color=sns.color_palette()[1])
    axes[1].set(title='Compression discards time order',ylabel='Number of values')
    for i,v in enumerate([1000,5]): axes[1].text(i,v+15,str(v),ha='center',fontsize=12)
    save(fig,'feature_compression',manifest)
    d=results['pipeline'][0]
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for ax,title,key in zip(axes,['Known synthetic target','Raw observed EEG','Filtered EEG: inspect residuals'],'true_neural_signal raw_observed_eeg eeg_clean'.split()):
        sns.lineplot(x=d['t'],y=d[key],ax=ax,linewidth=1)
        ax.set(title=title,ylabel='Voltage (µV)')
        ax.axvspan(1.6,2.4,color='orange',alpha=.15)
    axes[2].plot(d['t'],d['true_neural_signal'],color='0.5',alpha=.6,linewidth=.8,label='Known target')
    axes[2].legend(fontsize=10)
    axes[-1].set_xlabel('Time (s)')
    save(fig,'end_to_end',manifest)
    for name,(_,output) in results.items():
        print(f'\n[{name}]\n{output}')
    print(f'\n50 Hz gain for §7 methods reading: {gain50:.6f} ({20*np.log10(gain50):.3f} dB)')
    print('\nSaved figure manifest:')
    for path in manifest: print(path)

if __name__ == '__main__':
    main()
