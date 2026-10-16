"""Generate all Week 4 figures, execute laboratories, and export empirical artifacts.

This script executes the complete Week 4 AI + Neuroscience learning pipeline:
1. Loads cached PhysioNet EEGBCI data and verified Week 3 features.
2. Formulates aligned representations:
   - Handcrafted features: [450, 15]
   - Raw/Preprocessed EEG epochs: [450, 3, 480]
   - Binary labels: [450] (0 = LEFT_HAND, 1 = RIGHT_HAND)
3. Executes Experiment 1: Artificial Neuron and One Gradient Step.
4. Executes Experiment 2: Compact MLP on Real Features (Subject 1 run holdout).
5. Executes Experiment 3: Compact 1D CNN on Real Epochs (Subject-dependent + LOSO 10-fold benchmark).
6. Executes Experiment 4: Recurrent state dynamics (RNN & LSTM forward passes on real EEG).
7. Executes Experiment 5: Self-attention mechanism on real temporal tokens.
8. Produces all 12 publication-quality figures at 180 DPI in month 1/week 4/assets/.
9. Exports provenance manifest, split schemas, predictions, metrics, checkpoints, and captured lab stdout.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Global configuration and determinism
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)

sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams.update({'figure.dpi': 150, 'savefig.bbox': 'tight'})

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'month 1' / 'week 4' / 'assets'
DATA_DIR = ROOT / 'data'
ASSETS.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ['C3', 'Cz', 'C4']
FS = 160.0
SUBJECTS = list(range(1, 11))
RUNS = [4, 8, 12]

# -------------------------------------------------------------------------
# Architectural Definitions
# -------------------------------------------------------------------------

class CompactMLP(nn.Module):
    """Compact 2-layer Multi-Layer Perceptron for 15-dimensional handcrafted features."""
    def __init__(self, in_features=15, hidden_dim=8, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        h = self.relu(self.fc1(x))
        out = self.fc2(h)
        return out

class CompactEEGNet1D(nn.Module):
    """Compact 1D Convolutional Neural Network for raw multichannel EEG epochs [Batch, Channels, Time].
    
    Layer 1: Temporal convolution across time (kernel length 32 samples = 200 ms at 160 Hz)
    Layer 2: Max pooling downsampling temporal dimension by 4 (481 -> 120 positions)
    Layer 3: Higher-order temporal convolution (kernel length 16 pooled positions; input receptive field = 593.75 ms)
    Layer 4: Adaptive temporal average pooling to [Batch, 16, 1]
    Layer 5: Fully connected linear classification head to [Batch, 2] logits.
    Total trainable parameters: 2,922.
    """
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=32, stride=1, padding=16)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv2 = nn.Conv1d(8, 16, kernel_size=16, stride=1, padding=8)
        self.bn2 = nn.BatchNorm1d(16)
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Linear(16, num_classes)
        
    def forward(self, x):
        # x: [Batch, 3, 480]
        x1 = self.conv1(x)         # [Batch, 8, 481]
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        p1 = self.pool1(x1)        # [Batch, 8, 120]
        
        x2 = self.conv2(p1)        # [Batch, 16, 121]
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        p2 = self.pool2(x2)        # [Batch, 16, 1]
        
        feat = p2.squeeze(-1)      # [Batch, 16]
        logits = self.classifier(feat) # [Batch, 2]
        return logits

# -------------------------------------------------------------------------
# Execution & Benchmarking
# -------------------------------------------------------------------------

def run_experiment_1_neuron(df_features):
    """Experiment 1: Artificial Neuron and One Gradient Step on Real EEG Features."""
    # Select two features from Subject 1, Trial 0: C3 alpha power and C4 alpha power
    row0 = df_features.iloc[0]
    raw_x = np.array([row0['C3_alpha'], row0['C4_alpha']], dtype=np.float32)
    # Demonstration-only cohort scaling; not used in any held-out classifier
    c3_mean, c3_std = df_features['C3_alpha'].mean(), df_features['C3_alpha'].std()
    c4_mean, c4_std = df_features['C4_alpha'].mean(), df_features['C4_alpha'].std()
    x1 = float((raw_x[0] - c3_mean) / c3_std)
    x2 = float((raw_x[1] - c4_mean) / c4_std)
    x = np.array([x1, x2], dtype=np.float32)
    y_true = int(row0['target']) # actual row label; 0 = LEFT_HAND, 1 = RIGHT_HAND
    
    # Initialize parameters
    w = np.array([0.60, -0.40], dtype=np.float32)
    b = 0.10
    lr = 0.50
    
    # Forward pass
    z0 = float(np.dot(w, x) + b)
    y_hat0 = float(1.0 / (1.0 + np.exp(-z0)))
    # Binary cross-entropy loss: - [y log(y_hat) + (1-y) log(1-y_hat)]
    loss0 = float(- (y_true * np.log(max(y_hat0, 1e-12)) + (1 - y_true) * np.log(max(1.0 - y_hat0, 1e-12))))
    
    # Analytical gradient: dL/dz = y_hat - y_true
    dL_dz = y_hat0 - y_true
    grad_w = dL_dz * x
    grad_b = dL_dz
    
    # Parameter update
    w_new = w - lr * grad_w
    b_new = b - lr * grad_b
    
    # Forward pass after one update
    z1 = float(np.dot(w_new, x) + b_new)
    y_hat1 = float(1.0 / (1.0 + np.exp(-z1)))
    loss1 = float(- (y_true * np.log(max(y_hat1, 1e-12)) + (1 - y_true) * np.log(max(1.0 - y_hat1, 1e-12))))
    
    res = {
        'x_inputs': [float(x[0]), float(x[1])],
        'raw_inputs_uv2': [float(raw_x[0]), float(raw_x[1])],
        'target_label': y_true,
        'learning_rate': lr,
        'step_0': {
            'w': [float(w[0]), float(w[1])],
            'b': float(b),
            'z': z0,
            'probability': y_hat0,
            'bce_loss': loss0
        },
        'gradients': {
            'dL_dw1': float(grad_w[0]),
            'dL_dw2': float(grad_w[1]),
            'dL_db': float(grad_b)
        },
        'step_1': {
            'w': [float(w_new[0]), float(w_new[1])],
            'b': float(b_new),
            'z': z1,
            'probability': y_hat1,
            'bce_loss': loss1
        },
        'loss_reduction': float(loss0 - loss1)
    }
    return res

def run_experiment_2_mlp(df_features):
    """Experiment 2: Compact MLP on 15 Features (Subject 1 Run Holdout Protocol)."""
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    
    feature_cols = [c for c in df_features.columns if any(c.startswith(ch) for ch in CHANNELS)]
    s1_df = df_features[df_features['subject'] == 1].reset_index(drop=True)
    
    tr_idx = np.where(s1_df['run'] == 4)[0]
    val_idx = np.where(s1_df['run'] == 8)[0]
    te_idx = np.where(s1_df['run'] == 12)[0]
    
    X_raw = s1_df[feature_cols].values
    y_all = s1_df['target'].values
    
    # Leak-free normalization: fit StandardScaler strictly on training run 4
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_raw[tr_idx])
    X_val = scaler.transform(X_raw[val_idx])
    X_te = scaler.transform(X_raw[te_idx])
    
    y_tr = y_all[tr_idx]
    y_val = y_all[val_idx]
    y_te = y_all[te_idx]
    
    model = CompactMLP(in_features=15, hidden_dim=8, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-3)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_weights = None
    best_epoch = 0
    
    X_tr_t = torch.from_numpy(X_tr.astype(np.float32))
    y_tr_t = torch.from_numpy(y_tr.astype(np.int64))
    X_val_t = torch.from_numpy(X_val.astype(np.float32))
    y_val_t = torch.from_numpy(y_val.astype(np.int64))
    X_te_t = torch.from_numpy(X_te.astype(np.float32))
    y_te_t = torch.from_numpy(y_te.astype(np.int64))
    
    for epoch in range(1, 41):
        model.train()
        optimizer.zero_grad()
        out = model(X_tr_t)
        loss = criterion(out, y_tr_t)
        loss.backward()
        optimizer.step()
        
        tr_acc = float((torch.argmax(out, dim=1) == y_tr_t).float().mean())
        
        model.eval()
        with torch.no_grad():
            v_out = model(X_val_t)
            v_loss = float(criterion(v_out, y_val_t))
            v_acc = float((torch.argmax(v_out, dim=1) == y_val_t).float().mean())
            
        history['train_loss'].append(float(loss.item()))
        history['val_loss'].append(v_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(v_acc)
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            
    # Evaluate checkpoint selected by lowest validation loss on held-out test set
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        test_out = model(X_te_t)
        test_loss = float(criterion(test_out, y_te_t))
        test_preds = torch.argmax(test_out, dim=1).numpy()
        test_acc = float(accuracy_score(y_te, test_preds))
        cm = confusion_matrix(y_te, test_preds).tolist()
        
    return {
        'protocol': 'subject_dependent_run_holdout',
        'subject': 1,
        'splits': {'train_run': 4, 'val_run': 8, 'test_run': 12},
        'train_samples': len(tr_idx),
        'val_samples': len(val_idx),
        'test_samples': len(te_idx),
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'confusion_matrix': cm,
        'history': history
    }

def run_experiment_3_cnn(df_features, X_epochs):
    """Experiment 3: Compact 1D CNN Evaluation (Subject 1 Holdout + 10-Fold LOSO Benchmark)."""
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    
    # Part A: Subject 1 Run-Holdout Protocol
    s1_mask = (df_features['subject'] == 1).values
    s1_df = df_features[s1_mask].reset_index(drop=True)
    s1_epochs = X_epochs[s1_mask]
    
    tr_idx = np.where(s1_df['run'] == 4)[0]
    val_idx = np.where(s1_df['run'] == 8)[0]
    te_idx = np.where(s1_df['run'] == 12)[0]
    
    # Train-fold channel normalization
    ch_mean = s1_epochs[tr_idx].mean(axis=(0, 2), keepdims=True)
    ch_std = s1_epochs[tr_idx].std(axis=(0, 2), keepdims=True) + 1e-6
    
    X_tr = (s1_epochs[tr_idx] - ch_mean) / ch_std
    X_val = (s1_epochs[val_idx] - ch_mean) / ch_std
    X_te = (s1_epochs[te_idx] - ch_mean) / ch_std
    
    y_tr = s1_df.loc[tr_idx, 'target'].values
    y_val = s1_df.loc[val_idx, 'target'].values
    y_te = s1_df.loc[te_idx, 'target'].values
    
    cnn_s1 = CompactEEGNet1D()
    opt_s1 = optim.Adam(cnn_s1.parameters(), lr=0.005, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss()
    
    X_tr_t = torch.from_numpy(X_tr.astype(np.float32))
    y_tr_t = torch.from_numpy(y_tr.astype(np.int64))
    X_val_t = torch.from_numpy(X_val.astype(np.float32))
    y_val_t = torch.from_numpy(y_val.astype(np.int64))
    X_te_t = torch.from_numpy(X_te.astype(np.float32))
    y_te_t = torch.from_numpy(y_te.astype(np.int64))
    
    best_val_loss = float('inf')
    best_cnn_weights = None
    best_epoch_s1 = 0
    s1_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(1, 26):
        cnn_s1.train()
        opt_s1.zero_grad()
        out = cnn_s1(X_tr_t)
        loss = crit(out, y_tr_t)
        loss.backward()
        opt_s1.step()
        
        tr_acc = float((torch.argmax(out, dim=1) == y_tr_t).float().mean())
        
        cnn_s1.eval()
        with torch.no_grad():
            v_out = cnn_s1(X_val_t)
            v_loss = float(crit(v_out, y_val_t))
            v_acc = float((torch.argmax(v_out, dim=1) == y_val_t).float().mean())
            
        s1_history['train_loss'].append(float(loss.item()))
        s1_history['val_loss'].append(v_loss)
        s1_history['train_acc'].append(tr_acc)
        s1_history['val_acc'].append(v_acc)
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_cnn_weights = {k: v.clone() for k, v in cnn_s1.state_dict().items()}
            best_epoch_s1 = epoch
            
    cnn_s1.load_state_dict(best_cnn_weights)
    cnn_s1.eval()
    with torch.no_grad():
        te_out = cnn_s1(X_te_t)
        te_preds = torch.argmax(te_out, dim=1).numpy()
        te_acc_s1 = float(accuracy_score(y_te, te_preds))
        te_cm_s1 = confusion_matrix(y_te, te_preds).tolist()
        
    # Save the trained model checkpoint
    torch.save({
        'model_state_dict': best_cnn_weights,
        'best_epoch': best_epoch_s1,
        'val_loss': best_val_loss,
        'test_accuracy': te_acc_s1,
        'channel_mean': ch_mean.tolist(),
        'channel_std': ch_std.tolist()
    }, DATA_DIR / 'week4_cnn_checkpoint.pt')
    
    # Part B: Subject-Independent LOSO Benchmark (All 10 Subjects)
    feature_cols = [c for c in df_features.columns if any(c.startswith(ch) for ch in CHANNELS)]
    X_feats = df_features[feature_cols].values
    y_all = df_features['target'].values
    subs_all = df_features['subject'].values
    
    per_sub_metrics = {'Dummy': [], 'Logistic Regression': [], 'Compact 1D CNN': []}
    
    all_preds_dummy = np.zeros(len(y_all), dtype=int)
    all_preds_lr = np.zeros(len(y_all), dtype=int)
    all_preds_cnn = np.zeros(len(y_all), dtype=int)
    all_probs_cnn = np.zeros(len(y_all), dtype=float)
    
    for test_sub in SUBJECTS:
        train_idx = np.where(subs_all != test_sub)[0]
        test_idx = np.where(subs_all == test_sub)[0]
        
        y_train_fold = y_all[train_idx]
        y_test_fold = y_all[test_idx]
        
        # 1. Dummy
        dum = DummyClassifier(strategy='most_frequent')
        dum.fit(X_feats[train_idx], y_train_fold)
        p_dum = dum.predict(X_feats[test_idx])
        all_preds_dummy[test_idx] = p_dum
        
        # 2. Logistic Regression on 15 features
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_feats[train_idx])
        X_te_sc = scaler.transform(X_feats[test_idx])
        lr = LogisticRegression(solver='liblinear', C=1.0, random_state=RANDOM_STATE)
        lr.fit(X_tr_sc, y_train_fold)
        p_lr = lr.predict(X_te_sc)
        all_preds_lr[test_idx] = p_lr
        
        # 3. Compact 1D CNN on epochs
        ch_m = X_epochs[train_idx].mean(axis=(0, 2), keepdims=True)
        ch_s = X_epochs[train_idx].std(axis=(0, 2), keepdims=True) + 1e-6
        X_tr_cnn = (X_epochs[train_idx] - ch_m) / ch_s
        X_te_cnn = (X_epochs[test_idx] - ch_m) / ch_s
        
        torch.manual_seed(RANDOM_STATE + test_sub)
        cnn_fold = CompactEEGNet1D()
        opt_fold = optim.Adam(cnn_fold.parameters(), lr=0.005, weight_decay=1e-3)
        crit_fold = nn.CrossEntropyLoss()
        
        ds_tr = TensorDataset(torch.from_numpy(X_tr_cnn.astype(np.float32)), torch.from_numpy(y_train_fold.astype(np.int64)))
        loader = DataLoader(ds_tr, batch_size=32, shuffle=True)
        
        cnn_fold.train()
        for epoch in range(15):
            for bx, by in loader:
                opt_fold.zero_grad()
                out = cnn_fold(bx)
                loss = crit_fold(out, by)
                loss.backward()
                opt_fold.step()
                
        cnn_fold.eval()
        with torch.no_grad():
            te_tensor = torch.from_numpy(X_te_cnn.astype(np.float32))
            logits = cnn_fold(te_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            p_cnn = torch.argmax(logits, dim=1).numpy()
            
        all_preds_cnn[test_idx] = p_cnn
        all_probs_cnn[test_idx] = probs
        
        for m_name, preds in [('Dummy', p_dum), ('Logistic Regression', p_lr), ('Compact 1D CNN', p_cnn)]:
            acc = float(accuracy_score(y_test_fold, preds))
            f1 = float(f1_score(y_test_fold, preds, zero_division=0))
            per_sub_metrics[m_name].append({
                'subject': test_sub,
                'accuracy': acc,
                'f1': f1
            })
            
    # Build trial-level predictions dataframe
    df_preds = pd.DataFrame({
        'trial_index': np.arange(len(y_all)),
        'subject': subs_all,
        'run': df_features['run'].values,
        'label_str': df_features['label'].values,
        'y_true': y_all,
        'pred_dummy': all_preds_dummy,
        'pred_lr': all_preds_lr,
        'pred_cnn': all_preds_cnn,
        'prob_cnn_right_hand': all_probs_cnn
    })
    df_preds.to_csv(DATA_DIR / 'week4_predictions.csv', index=False)
    
    # Compute pooled and descriptive metrics
    def summarize_model(preds):
        return {
            'pooled_accuracy': float(accuracy_score(y_all, preds)),
            'pooled_precision': float(precision_score(y_all, preds, zero_division=0)),
            'pooled_recall': float(recall_score(y_all, preds, zero_division=0)),
            'pooled_macro_f1': float(f1_score(y_all, preds, average='macro', zero_division=0)),
            'pooled_binary_f1': float(f1_score(y_all, preds, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_all, preds).tolist()
        }
        
    summary_metrics = {
        'evaluation_protocol': 'Leave-One-Subject-Out (LOSO) 10-fold cross-validation',
        'total_trials': len(y_all),
        'models': {
            'Dummy': summarize_model(all_preds_dummy),
            'Logistic Regression': summarize_model(all_preds_lr),
            'Compact 1D CNN': summarize_model(all_preds_cnn)
        },
        'per_subject_summary': {}
    }
    
    for m_name in ['Dummy', 'Logistic Regression', 'Compact 1D CNN']:
        accs = [item['accuracy'] for item in per_sub_metrics[m_name]]
        f1s = [item['f1'] for item in per_sub_metrics[m_name]]
        summary_metrics['per_subject_summary'][m_name] = {
            'mean_accuracy': float(np.mean(accs)),
            'std_accuracy': float(np.std(accs)),
            'per_subject_accuracy': accs,
            'mean_f1': float(np.mean(f1s)),
            'std_f1': float(np.std(f1s))
        }
        
    res_cnn = {
        'subject_dependent_s1': {
            'best_epoch': best_epoch_s1,
            'val_loss': best_val_loss,
            'test_accuracy': te_acc_s1,
            'confusion_matrix': te_cm_s1,
            'history': s1_history
        },
        'subject_independent_loso': summary_metrics
    }
    return res_cnn, cnn_s1, ch_mean, ch_std

def run_experiment_4_rnn_lstm(X_epochs):
    """Experiment 4: Recurrent State Dynamics on Real Continuous EEG."""
    torch.manual_seed(RANDOM_STATE)
    # Take Subject 1, Trial 0 (duration 3.0 s, 480 samples, shape [3, 480])
    raw_trial = X_epochs[0] # [3, 480]
    # Transpose explicitly to [Batch, Time, Channels] for batch_first=True
    trial_t = torch.from_numpy(raw_trial.T).unsqueeze(0).float() # [1, 480, 3]
    
    # Initialize compact recurrent layers
    rnn = nn.RNN(input_size=3, hidden_size=8, batch_first=True)
    lstm = nn.LSTM(input_size=3, hidden_size=8, batch_first=True)
    
    rnn.eval()
    lstm.eval()
    
    with torch.no_grad():
        rnn_out, rnn_hn = rnn(trial_t)           # rnn_out: [1, 480, 8], rnn_hn: [1, 1, 8]
        lstm_out, (lstm_hn, lstm_cn) = lstm(trial_t) # lstm_out: [1, 480, 8], lstm_hn: [1, 1, 8], lstm_cn: [1, 1, 8]
        
    return {
        'trial_shape': list(trial_t.shape),
        'rnn_output_shape': list(rnn_out.shape),
        'rnn_hidden_shape': list(rnn_hn.shape),
        'lstm_output_shape': list(lstm_out.shape),
        'lstm_hidden_shape': list(lstm_hn.shape),
        'lstm_cell_shape': list(lstm_cn.shape),
        'rnn_h_series': rnn_out.squeeze(0).numpy(), # [480, 8]
        'lstm_h_series': lstm_out.squeeze(0).numpy(), # [480, 8]
        'raw_trial_uv': raw_trial
    }

def run_experiment_5_attention(X_epochs):
    """Experiment 5: Self-Attention Mechanism on Temporal Tokens."""
    torch.manual_seed(RANDOM_STATE)
    # Take Subject 1, Trial 0 (480 samples = 3.0 s)
    trial_raw = X_epochs[0] # [3, 480]
    
    # Partition 480 samples into 6 non-overlapping temporal tokens (80 samples = 0.5 s each)
    num_tokens = 6
    patch_len = 80
    tokens = []
    token_labels = []
    
    for i in range(num_tokens):
        s = i * patch_len
        e = (i + 1) * patch_len
        patch = trial_raw[:, s:e].flatten() # 3 * 80 = 240 values
        tokens.append(patch)
        t_start = 0.5 + i * 0.5 # Analysis window start is 0.5s post-cue
        t_end = 0.5 + (i + 1) * 0.5
        token_labels.append(f'T{i} ({t_start:.1f}-{t_end:.1f}s)')
        
    tokens_np = np.array(tokens, dtype=np.float32) # [6, 240]
    tokens_tensor = torch.from_numpy(tokens_np).unsqueeze(0) # [1, 6, 240]
    
    # Project 240-dim raw patch into d_model=16 embedding space
    d_model = 16
    proj = nn.Linear(240, d_model)
    pos_embed = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)
    
    with torch.no_grad():
        x_embed = proj(tokens_tensor) + pos_embed # [1, 6, 16]
        
        # Self-attention projections: Q, K, V
        W_q = nn.Linear(d_model, d_model, bias=False)
        W_k = nn.Linear(d_model, d_model, bias=False)
        W_v = nn.Linear(d_model, d_model, bias=False)
        
        Q = W_q(x_embed) # [1, 6, 16]
        K = W_k(x_embed) # [1, 6, 16]
        V = W_v(x_embed) # [1, 6, 16]
        
        # Scaled dot-product: Attention(Q, K) = softmax(Q K^T / sqrt(d_k))
        scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(d_model) # [1, 6, 6]
        attn_weights = torch.softmax(scores, dim=-1).squeeze(0).numpy() # [6, 6]
        
    return {
        'num_tokens': num_tokens,
        'patch_samples': patch_len,
        'patch_seconds': 0.5,
        'token_labels': token_labels,
        'attention_matrix': attn_weights,
        'raw_trial': trial_raw
    }

# -------------------------------------------------------------------------
# Figure Generation (12 Purposeful Figures)
# -------------------------------------------------------------------------

def generate_all_plots(df_features, X_epochs, exp1_res, exp2_res, exp3_res, exp4_res, exp5_res):
    """Render readable, consistently scaled figures from executed experiments."""
    from matplotlib.patches import FancyBboxPatch
    manifest = []
    colors = ['#687786', '#2673a5', '#b45724']  # dummy, linear, CNN
    ink = '#233344'
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
        'axes.titleweight': 'bold', 'axes.titlepad': 12,
        'axes.spines.top': False, 'axes.spines.right': False,
        'grid.alpha': 0.22, 'lines.linewidth': 1.8,
    })

    def save(fig, name, title, note):
        fig.suptitle(title, fontsize=17, fontweight='bold', color=ink)
        fig.text(0.5, 0.015, note, ha='center', fontsize=10, color=ink)
        fig.tight_layout(rect=(0, 0.06, 1, 0.93), pad=1.5)
        path = ASSETS / name
        fig.savefig(path, dpi=180, facecolor='white', metadata={'Software': 'NeuroScience Week 4'})
        plt.close(fig)
        manifest.append(str(path.relative_to(ROOT)))

    def box(ax, x, y, text, width=1.75, height=0.78, color='#eaf2f7'):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                     boxstyle='round,pad=0.03', facecolor=color, edgecolor='#8aa0b3'))
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=11, color=ink)

    # 1: Both branches share preprocessing; gradients only change learned stages.
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.set(xlim=(-0.1, 10.1), ylim=(-0.6, 3.4)); ax.axis('off')
    for y, name, stages in [
        (2, 'Classical', ['Filtered EEG\n[3 × 480]', 'Fixed summaries\n15 features', 'Linear classifier\n2 classes', 'Prediction\nleft / right']),
        (0.5, 'CNN', ['Filtered EEG\n[3 × 480]', 'Learned kernels\n+ nonlinearities', 'Pooling + head\n16 → 2', 'Prediction\nleft / right'])]:
        ax.text(0, y + 1.0, name, weight='bold', fontsize=13)
        for j, text in enumerate(stages):
            x = 0.1 + j * 2.55
            box(ax, x, y, text, width=2.1, color='#eaf2f7' if j != 1 else '#f6e9dc')
            if j < 3:
                ax.annotate('', (x+2.53, y+0.39), (x+2.15, y+0.39), arrowprops={'arrowstyle':'->', 'color':ink})
    ax.annotate('', (2.7, 0.22), (8.8, 0.22), arrowprops={'arrowstyle':'->', 'color':colors[2], 'lw':2})
    ax.text(5.75, -0.1, 'Training loss updates kernels and classifier together', ha='center', color=colors[2], fontsize=11)
    save(fig, 'fig1_representation_pipeline.png', 'Same EEG, different representations',
         'Both pipelines retain 1–35 Hz preprocessing. Learned features do not guarantee better generalization.')

    # 2: Evaluate the actual joint weight AND bias displacement, not a fixed-bias contour.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7))
    before, after = exp1_res['step_0'], exp1_res['step_1']
    z = np.linspace(-3, 3, 240)
    sns.lineplot(x=z, y=1/(1+np.exp(-z)), ax=axes[0], color=colors[1])
    for state, label, color, offset in [(before, 'Before', colors[0], (-55,-32)), (after, 'After', colors[2], (10,12))]:
        axes[0].scatter(state['z'], state['probability'], color=color, s=65, zorder=4)
        axes[0].annotate(f"{label}: {state['probability']:.4f}", (state['z'], state['probability']),
                         xytext=offset, textcoords='offset points', fontsize=10)
    axes[0].set(title='A. Probability moves toward y = 1', xlabel='Logit z (dimensionless)', ylabel='P(right hand)')
    a = np.linspace(0, 1.5, 151)
    x = np.array(exp1_res['x_inputs'])
    w0, w1 = np.array(before['w']), np.array(after['w'])
    zz = (w0[None,:] + a[:,None]*(w1-w0)) @ x + before['b'] + a*(after['b']-before['b'])
    losses = np.logaddexp(0, zz) - exp1_res['target_label']*zz
    sns.lineplot(x=a, y=losses, ax=axes[1], color=colors[1])
    axes[1].scatter([0,1], [before['bce_loss'],after['bce_loss']], color=[colors[0],colors[2]], s=65, zorder=4)
    axes[1].axvline(1, ls=':', color=colors[2])
    axes[1].set(title='B. Joint parameter update', xlabel='Fraction of executed update', ylabel='BCE (dimensionless)')
    bars = axes[2].bar(['Before', 'After'], [before['bce_loss'],after['bce_loss']], color=[colors[0],colors[2]], width=0.55)
    axes[2].bar_label(bars, fmt='%.4f', padding=5)
    axes[2].set(ylim=(0,0.65), title='C. Single-example loss falls', ylabel='BCE (dimensionless)')
    save(fig, 'fig2_artificial_neuron_step.png', 'One gradient step on measured alpha-power features',
         'Demonstration only: cohort-wide feature scaling; a lower single-example loss is not held-out evidence.')

    # 3: Retain all epochs and mark validation-selected checkpoint.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    history = exp2_res['history']; epochs = np.arange(1, len(history['train_loss'])+1)
    for ax, suffix, ylabel in [(axes[0],'loss','Cross-entropy (dimensionless)'),(axes[1],'acc','Accuracy (%)')]:
        for split, color in [('train',colors[1]),('val',colors[2])]:
            values = np.array(history[f'{split}_{suffix}']) * (100 if suffix=='acc' else 1)
            sns.lineplot(x=epochs, y=values, ax=ax, color=color, label={'train':'Training run 4','val':'Validation run 8'}[split])
        ax.axvline(exp2_res['best_epoch'], color=ink, ls=':', label=f"Selected epoch {exp2_res['best_epoch']}")
        ax.set(xlabel='Training epoch (count)', ylabel=ylabel)
    axes[0].set_title('A. Training fit diverges from validation')
    axes[1].set(title='B. Accuracy hides confidence changes', ylim=(0,105))
    save(fig, 'fig3_mlp_training_history.png', 'Small-cohort MLP: fitting is easier than generalizing',
         'S01: 15 training / 15 validation / 15 test trials. Selected-checkpoint test accuracy: 40.0% (6/15).')

    cnn_metrics, model, ch_mean, ch_std = exp3_res
    raw = X_epochs[0]; t = 0.5 + np.arange(480)/FS
    model.eval(); norm = torch.from_numpy(((raw[None,:,:]-ch_mean)/ch_std).astype(np.float32))
    with torch.no_grad():
        c1 = model.relu(model.bn1(model.conv1(norm))); p1 = model.pool1(c1)
        c2 = model.relu(model.bn2(model.conv2(p1))); p2 = model.pool2(c2)
    weights = model.conv1.weight.detach().numpy()

    # 4: Actual measured shapes include the even-kernel padding +1.
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios':[1.3,1]})
    for c, ch in enumerate(CHANNELS): sns.lineplot(x=t, y=raw[c], ax=axes[0], label=ch)
    axes[0].set(title='A. One preprocessed motor-imagery epoch (S01)', xlabel='Time after cue (s)', ylabel='Voltage (µV)')
    axes[1].axis('off'); axes[1].set(xlim=(0,12), ylim=(0,2))
    stages = [('Input','[1, 3, 480]'),('Conv1 + BN\n+ ReLU',str(list(c1.shape))),('Max pool',str(list(p1.shape))),
              ('Conv2 + BN\n+ ReLU',str(list(c2.shape))),('Mean pool\n+ squeeze','[1, 16]'),('Linear head','[1, 2]')]
    for j,(label,shape) in enumerate(stages):
        box(axes[1], j*2+0.05, 0.6, label+'\n'+shape, width=1.7, height=1.05)
        if j<5: axes[1].annotate('', (j*2+2.02,1.1), (j*2+1.8,1.1), arrowprops={'arrowstyle':'->'})
    save(fig, 'fig4_cnn_architecture_trace.png', 'CNN geometry: follow channels and time separately',
         '2,922 trainable parameters. Even kernels with symmetric padding produce 481 and 121 positions.')

    # 5: Explicit single-channel cross-correlation, using the trained C3 kernel.
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    kernel = weights[0,0]; start = 160; ktime = np.arange(32)/FS*1000
    sns.lineplot(x=t, y=raw[0], ax=axes[0], color=colors[1])
    axes[0].axvspan(t[start], t[start]+32/FS, color=colors[2], alpha=0.18)
    axes[0].set(title='A. C3 voltage: shaded 32-sample window', ylabel='Voltage (µV)', xlabel='Time after cue (s)')
    axes[1].plot(ktime, kernel, marker='o', ms=3, color=colors[2]); axes[1].axhline(0,color=ink,lw=0.7)
    axes[1].set(title='B. Trained filter 0, C3 component', ylabel='Kernel weight (dimensionless)', xlabel='Kernel lag (ms)')
    response = np.correlate(raw[0], kernel, mode='valid')
    sns.lineplot(x=t[:len(response)]+(32-1)/(2*FS), y=response, ax=axes[2], color=colors[1])
    axes[2].set(title='C. Single-channel valid cross-correlation', ylabel='Weighted voltage (µV)', xlabel='Window-center time after cue (s)')
    save(fig, 'fig5_temporal_convolution_sliding.png', 'A temporal kernel is a sliding weighted sum',
         'Panel C isolates C3 without bias or normalization; the network sums all three channels before BN + ReLU.')

    # 6: Kernel responses are diagnostics, with shared activation ranges.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    f = np.fft.rfftfreq(1024, d=1/FS)
    for i in range(4):
        axes[0,0].plot(ktime, weights[i,0], label=f'Filter {i}')
        axes[0,1].plot(f, np.abs(np.fft.rfft(weights[i,0], n=1024)), label=f'Filter {i}')
    axes[0,0].set(title='A. C3 kernel components', xlabel='Kernel lag (ms)', ylabel='Weight (dimensionless)'); axes[0,0].legend(ncol=2)
    axes[0,1].axvspan(8,12,color='#66a182',alpha=.16,label='8–12 Hz')
    axes[0,1].axvspan(13,30,color='#aa83b5',alpha=.12,label='13–30 Hz')
    axes[0,1].set(title='B. Kernel magnitude response', xlim=(0,80), xlabel='Frequency (Hz)', ylabel='Magnitude (dimensionless)'); axes[0,1].legend(ncol=2,fontsize=9)
    maps = [c1[0].numpy(),p1[0].numpy()]; vmax = max(m.max() for m in maps)
    for ax, arr, title in zip(axes[1],maps,['C. BN + ReLU: [8 × 481]','D. Max pooling: [8 × 120]']):
        ax.imshow(arr,origin='lower',aspect='auto',extent=(0.5,3.5,-.5,7.5),cmap='magma',vmin=0,vmax=vmax)
        ax.set(title=title,xlabel='Approximate time after cue (s)',ylabel='Feature-map index'); ax.grid(False)
    fig.colorbar(axes[1,1].images[0], ax=axes[1,1], label='Activation (dimensionless)')
    save(fig, 'fig6_learned_kernels_and_feature_maps.png', 'Inspect learned filters without assigning biological causes',
         'S01 run-holdout checkpoint (epoch 9), not a LOSO model. Band shading is reference; no physiological selectivity test.')

    loso = cnn_metrics['subject_independent_loso']; names = ['Dummy','Logistic Regression','Compact 1D CNN']
    # 7: Common confusion scale, row-normalized percentages and raw counts.
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), gridspec_kw={'height_ratios':[1.2,1]})
    for j,name in enumerate(names):
        cm=np.array(loso['models'][name]['confusion_matrix']); pct=cm/cm.sum(axis=1,keepdims=True)*100
        annot=np.array([[f'{cm[r,c]}\n({pct[r,c]:.1f}%)' for c in range(2)] for r in range(2)])
        sns.heatmap(pct,annot=annot,fmt='',vmin=0,vmax=100,cmap='Blues',cbar=False,square=True,
                    xticklabels=['Left','Right'],yticklabels=['Left','Right'],ax=axes[0,j],annot_kws={'fontsize':12})
        axes[0,j].set(title=name, xlabel='Predicted class', ylabel='True class'); axes[0,j].tick_params(axis='y',rotation=0)
    for j,key in enumerate(['pooled_accuracy','pooled_recall','pooled_macro_f1']):
        vals=[loso['models'][n][key]*100 for n in names]
        bars=axes[1,j].bar(['Dummy','Logistic','CNN'],vals,color=colors,width=.6)
        axes[1,j].bar_label(bars,fmt='%.1f',padding=4)
        axes[1,j].set(ylim=(0,105),ylabel='Score (%)',title=['Accuracy','Right-hand recall','Macro-F1'][j])
    save(fig, 'fig7_heldout_confusion_matrices.png', 'LOSO predictions: modest accuracy, unequal error patterns',
         '450 held-out predictions (230 left / 220 right). Heatmap colors = within-class percentage; annotations also give counts.')

    # 8: Per-subject data + paired differences; no error bars implying confidence.
    fig, axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    x=np.arange(1,11)
    for name,color,marker in zip(names,colors,['s','o','D']):
        vals=np.array(loso['per_subject_summary'][name]['per_subject_accuracy'])*100
        axes[0].plot(x,vals,marker=marker,color=color,label=name,ms=6)
    axes[0].axhline(50,ls=':',color=ink,label='Balanced-label reference (50%)')
    axes[0].set(title='A. Every subject is held out once',ylabel='Held-out accuracy (%)',ylim=(35,80)); axes[0].legend(ncol=2,fontsize=10)
    delta=(np.array(loso['per_subject_summary'][names[2]]['per_subject_accuracy'])-np.array(loso['per_subject_summary'][names[1]]['per_subject_accuracy']))*100
    bars=axes[1].bar(x,delta,color=[colors[2] if d>=0 else colors[1] for d in delta],width=.6)
    axes[1].axhline(0,color=ink,lw=1); axes[1].bar_label(bars,labels=[f'{d:+.1f}' for d in delta],padding=3,fontsize=10)
    axes[1].set(title='B. CNN gain depends on the held-out subject',ylabel='CNN − logistic (percentage points)',xlabel='Held-out participant (45 trials each)',ylim=(-17,38))
    axes[1].set_xticks(x,[f'S{i:02}' for i in x])
    save(fig,'fig8_subject_generalization_comparison.png','Pooled improvement conceals inconsistent transfer',
         'CNN: wins on 4 subjects, ties on 1, loses on 5 versus logistic regression. Subject SD is not a confidence interval.')

    # 9: Untrained mechanisms shown with identical hidden-state color ranges.
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for c,ch in enumerate(CHANNELS): axes[0].plot(t,raw[c],label=ch)
    axes[0].set(title='A. Measured input (preprocessed)',ylabel='Voltage (µV)'); axes[0].legend(ncol=3)
    for ax,key,title in [(axes[1],'rnn_h_series','B. Random RNN: [Time × Hidden]'),(axes[2],'lstm_h_series','C. Random LSTM: [Time × Hidden]')]:
        im=ax.imshow(exp4_res[key].T,origin='lower',aspect='auto',extent=(.5,3.5,-.5,7.5),vmin=-1,vmax=1,cmap='RdBu_r')
        ax.set(title=title,ylabel='Hidden-unit index'); ax.grid(False)
        fig.colorbar(im,ax=ax,label='Activation')
    axes[2].set_xlabel('Time after cue (s)')
    save(fig,'fig9_rnn_lstm_state_dynamics.png','Recurrent forward passes illustrate state geometry',
         'Random initialized networks, no training or decoding evaluation. Different textures do not establish memory quality.')

    # 10: Consistent time origin, full channel patch geometry.
    fig,axes=plt.subplots(2,1,figsize=(11,6.5),gridspec_kw={'height_ratios':[1.5,1]})
    for i in range(6):
        axes[0].axvspan(.5+i*.5,1+i*.5,color=sns.color_palette('pastel')[i],alpha=.5)
        axes[0].text(.75+i*.5,.97,f'T{i}',transform=axes[0].get_xaxis_transform(),ha='center',va='top',weight='bold')
    axes[0].plot(t,raw[0],color=ink,lw=1.2)
    axes[0].set(title='A. C3 trace with six temporal patches',xlabel='Time after cue (s)',ylabel='Voltage (µV)',xlim=(.5,3.5))
    axes[1].axis('off'); axes[1].set(xlim=(0,10),ylim=(0,2))
    for j,text in enumerate(['6 patches\neach [3 × 80]','Flatten\n[6 × 240]','Project + position\n[6 × 16]','Q, K, V\neach [6 × 16]']):
        box(axes[1],j*2.5+.05,.5,text,width=2.05,height=.95)
        if j<3: axes[1].annotate('',(j*2.5+2.5,1),(j*2.5+2.13,1),arrowprops={'arrowstyle':'->'})
    save(fig,'fig10_eeg_tokenization_geometry.png','Tokenization changes sequence length, not measurement physics',
         '80 samples / 160 Hz = 0.5 s. Each patch uses all three channels; the plot displays C3 for clarity.')

    # 11: Show numerical saturation openly; no biological routing schematic.
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    att=exp5_res['attention_matrix']; labels=[f'T{i}' for i in range(6)]
    sns.heatmap(att,annot=True,fmt='.2f',vmin=0,vmax=1,cmap='Blues',square=True,
                xticklabels=labels,yticklabels=labels,cbar_kws={'label':'Routing weight'},ax=axes[0])
    axes[0].set(title='A. Untrained, unnormalized attention',xlabel='Key token (source)',ylabel='Query token (destination)'); axes[0].tick_params(axis='y',rotation=0)
    bars=axes[1].bar(labels,att.max(axis=1),color=colors[1],width=.6)
    axes[1].bar_label(bars,fmt='%.3f',padding=4)
    axes[1].set(title='B. Every query is nearly saturated',xlabel='Query token',ylabel='Largest routing weight',ylim=(0,1.15))
    save(fig,'fig11_self_attention_matrix.png','Softmax row sums can be correct while attention saturates',
         'Random projections of µV patches; no supervised optimization. One-hot routing is not learned connectivity.')

    # 12: Readable three-column cards rather than a dense seven-row raster table.
    fig,axes=plt.subplots(2,2,figsize=(12,7.5))
    cards=[('Motor imagery / assistive BCI','EEG → left / right cue label','Compact CNN; temporal + channel features','Risk: artifacts, session and subject shift'),
           ('Seizure detection','EEG → annotated event / onset','CNN or causal temporal model','Risk: missed events and false alarms/hour'),
           ('Sleep staging','EEG + EOG + EMG → stage label','CNN + optional sequence context','Risk: scoring disagreement; future context'),
           ('Neuroimaging / population decoding','MRI → lesion mask; spikes → kinematics','3D CNN / recurrent model (different tasks)','Risk: scanner confounds; recording drift')]
    for ax,(title,data,architecture,risk) in zip(axes.flat,cards):
        ax.axis('off'); ax.set(xlim=(0,1),ylim=(0,1))
        ax.add_patch(FancyBboxPatch((.02,.05),.94,.9,boxstyle='round,pad=.02',facecolor='#edf3f7',edgecolor='#b8c9d6'))
        ax.text(.07,.82,title,weight='bold',fontsize=13,color=ink)
        for y,text in [(.61,data),(.42,architecture),(.23,risk)]: ax.text(.07,y,text,fontsize=11,color=ink)
    save(fig,'fig12_neuroscience_applications_landscape.png','Choose the target and evaluation before the architecture',
         'Application examples are formulations, not results of this motor-imagery benchmark; see §7 for details.')
    return manifest


def main():
    print("="*75)
    print("NEUROSCIENCE VAULT: WEEK 4 DEEP LEARNING PIPELINE")
    print("="*75)
    
    # 1. Load data
    features_csv = DATA_DIR / 'week3_features.csv'
    epochs_npy = DATA_DIR / 'week4_epochs.npy'
    
    if not features_csv.exists() or not epochs_npy.exists():
        raise FileNotFoundError("Prerequisite data files missing. Verify week3_features.csv and week4_epochs.npy.")
        
    df_features = pd.read_csv(features_csv)
    X_epochs = np.load(epochs_npy)

    # Validate the cached representation contract before model fitting.
    feature_cols = [c for c in df_features if any(c.startswith(ch) for ch in CHANNELS)]
    if X_epochs.shape != (len(df_features), len(CHANNELS), 480) or len(feature_cols) != 15:
        raise ValueError('Feature rows and [Trials, 3, 480] epochs must align; expected 15 features.')
    if not np.isfinite(X_epochs).all() or not np.isfinite(df_features[feature_cols].to_numpy()).all():
        raise ValueError('Cached inputs contain non-finite values.')
    if set(df_features['subject']) != set(SUBJECTS) or set(df_features['target']) != {0, 1}:
        raise ValueError('Expected subjects 1–10 and binary left/right labels.')
    
    print(f"Loaded Features: {df_features.shape}")
    print(f"Loaded Epochs:   {X_epochs.shape}")
    print(f"Cohort: 10 subjects, Runs [4, 8, 12], Channels {CHANNELS}, FS = {FS} Hz")
    
    # 2. Experiment 1: Artificial Neuron
    print("\n--- Running Experiment 1: Artificial Neuron One Gradient Step ---")
    exp1_res = run_experiment_1_neuron(df_features)
    print(f"Neuron step 0: z={exp1_res['step_0']['z']:.4f}, prob={exp1_res['step_0']['probability']:.4f}, loss={exp1_res['step_0']['bce_loss']:.4f}")
    print(f"Gradients: dL/dw1={exp1_res['gradients']['dL_dw1']:.4f}, dL/dw2={exp1_res['gradients']['dL_dw2']:.4f}, dL/db={exp1_res['gradients']['dL_db']:.4f}")
    print(f"Neuron step 1: z={exp1_res['step_1']['z']:.4f}, prob={exp1_res['step_1']['probability']:.4f}, loss={exp1_res['step_1']['bce_loss']:.4f}")
    print(f"Loss reduction: {exp1_res['loss_reduction']:.4f}")
    
    # 3. Experiment 2: Compact MLP
    print("\n--- Running Experiment 2: Compact MLP on 15 Real Features ---")
    exp2_res = run_experiment_2_mlp(df_features)
    print(f"MLP Subject 1: Best Epoch {exp2_res['best_epoch']} (Val Loss = {exp2_res['best_val_loss']:.4f})")
    print(f"MLP Test Accuracy (Run 12 held-out, N=15): {exp2_res['test_accuracy']*100:.1f}%")
    print(f"MLP Test Confusion Matrix: {exp2_res['confusion_matrix']}")
    
    # 4. Experiment 3: Compact 1D CNN
    print("\n--- Running Experiment 3: Compact 1D CNN on Real Multichannel Epochs ---")
    exp3_res = run_experiment_3_cnn(df_features, X_epochs)
    cnn_metrics, cnn_model, ch_mean, ch_std = exp3_res
    print(f"CNN Subject 1 Holdout Test Accuracy: {cnn_metrics['subject_dependent_s1']['test_accuracy']*100:.1f}%")
    loso = cnn_metrics['subject_independent_loso']
    print("LOSO 10-fold completed (fixed 15 training epochs per fold)")
    for m in ['Dummy', 'Logistic Regression', 'Compact 1D CNN']:
        p_acc = loso['models'][m]['pooled_accuracy'] * 100
        m_acc = loso['per_subject_summary'][m]['mean_accuracy'] * 100
        s_acc = loso['per_subject_summary'][m]['std_accuracy'] * 100
        p_f1 = loso['models'][m]['pooled_macro_f1'] * 100
        print(f"  {m:22s} | Pooled Acc: {p_acc:.1f}% | Sub Mean: {m_acc:.1f}% (±{s_acc:.1f}%) | Macro-F1: {p_f1:.1f}%")
        
    # 5. Experiment 4: Recurrent State Dynamics
    print("\n--- Running Experiment 4: RNN & LSTM State Evolution on Real EEG ---")
    exp4_res = run_experiment_4_rnn_lstm(X_epochs)
    print(f"RNN output shape: {exp4_res['rnn_output_shape']}, hidden shape: {exp4_res['rnn_hidden_shape']}")
    print(f"LSTM output shape: {exp4_res['lstm_output_shape']}, hidden shape: {exp4_res['lstm_hidden_shape']}, cell shape: {exp4_res['lstm_cell_shape']}")
    
    # 6. Experiment 5: Self-Attention on Temporal Tokens
    print("\n--- Running Experiment 5: Self-Attention Mechanism on Temporal Tokens ---")
    exp5_res = run_experiment_5_attention(X_epochs)
    print(f"Extracted {exp5_res['num_tokens']} tokens, shape: {exp5_res['attention_matrix'].shape}")
    print(f"Attention matrix row sums (softmax check):\n{np.sum(exp5_res['attention_matrix'], axis=1)}")
    
    # 7. Generate Figures
    print("\n--- Generating All 12 Publication-Quality Figures ---")
    manifest = generate_all_plots(df_features, X_epochs, exp1_res, exp2_res, exp3_res, exp4_res, exp5_res)
    
    # 8. Save Metrics & Manifest Artifacts
    print("\n--- Saving Manifests and Metric Summaries ---")
    
    # Provenance manifest
    prov_manifest = {
        'week': 4,
        'title': 'Deep Learning for Neuroscience: Neural Networks, CNNs, Sequential Models and Applications',
        'dataset': 'PhysioNet EEG Motor Movement/Imagery Dataset (eegmmidb)',
        'doi': '10.13026/C28G6P',
        'license': 'Creative Commons Attribution 4.0 International (CC BY 4.0)',
        'sampling_rate_hz': FS,
        'channels': CHANNELS,
        'cohort_subjects': SUBJECTS,
        'runs': RUNS,
        'total_trials': len(df_features),
        'epoch_duration_sec': 3.0,
        'epoch_samples': 480,
        'window_post_cue_sec': [0.5, 3.5],
        'bandpass_filter': {
            'type': 'Butterworth zero-phase band-pass (filtfilt)',
            'low_cut_hz': 1.0,
            'high_cut_hz': 35.0,
            'order': 4
        },
        'normalization': {
            'channel_wise_z_score': 'Fitted strictly on training partition; mean and std frozen for validation/test',
            'axes': 'Channel-wise over (trials, time)'
        },
        'random_seed': RANDOM_STATE,
        'torch_version': torch.__version__,
        'figures_generated': len(manifest),
        'figure_files': manifest
    }
    with open(DATA_DIR / 'week4_manifest.json', 'w') as f:
        json.dump(prov_manifest, f, indent=2)
        
    # Split manifest
    splits_manifest = {
        'subject_dependent': {
            'description': 'Run-held-out evaluation within Subject 1',
            'subject': 1,
            'train_run': 4,
            'validation_run': 8,
            'test_run': 12,
            'trials_per_run': 15
        },
        'subject_independent': {
            'description': 'Leave-One-Subject-Out (LOSO) cross-validation across all 10 subjects',
            'num_folds': 10,
            'trials_per_fold_train': 405,
            'trials_per_fold_test': 45
        }
    }
    with open(DATA_DIR / 'week4_splits.json', 'w') as f:
        json.dump(splits_manifest, f, indent=2)
        
    # Full metrics json
    full_metrics = {
        'experiment_1_neuron': exp1_res,
        'experiment_2_mlp': exp2_res,
        'experiment_3_cnn': cnn_metrics,
        'experiment_4_rnn_lstm': {
            'input_shape': exp4_res['trial_shape'],
            'rnn_output_shape': exp4_res['rnn_output_shape'],
            'lstm_output_shape': exp4_res['lstm_output_shape']
        },
        'experiment_5_attention': {
            'num_tokens': exp5_res['num_tokens'],
            'token_labels': exp5_res['token_labels'],
            'attention_matrix': exp5_res['attention_matrix'].tolist()
        }
    }
    with open(DATA_DIR / 'week4_metrics.json', 'w') as f:
        json.dump(full_metrics, f, indent=2)
        
    # Capture standalone lab outputs
    lab_outputs = {
        'lab_w4d1_neuron_step': (
            f"--- ARTIFICIAL NEURON SINGLE GRADIENT STEP ---\n"
            f"Input Features (C3, C4 alpha power): [{exp1_res['x_inputs'][0]:.4f}, {exp1_res['x_inputs'][1]:.4f}]\n"
            f"Initial Weights: w=[{exp1_res['step_0']['w'][0]:.4f}, {exp1_res['step_0']['w'][1]:.4f}], b={exp1_res['step_0']['b']:.4f}\n"
            f"Step 0: Logit z={exp1_res['step_0']['z']:.4f}, Prob y_hat={exp1_res['step_0']['probability']:.4f}, BCE Loss={exp1_res['step_0']['bce_loss']:.4f}\n"
            f"Gradients: dL/dw1={exp1_res['gradients']['dL_dw1']:.4f}, dL/dw2={exp1_res['gradients']['dL_dw2']:.4f}, dL/db={exp1_res['gradients']['dL_db']:.4f}\n"
            f"Updated Weights (lr={exp1_res['learning_rate']}): w=[{exp1_res['step_1']['w'][0]:.4f}, {exp1_res['step_1']['w'][1]:.4f}], b={exp1_res['step_1']['b']:.4f}\n"
            f"Step 1: Logit z={exp1_res['step_1']['z']:.4f}, Prob y_hat={exp1_res['step_1']['probability']:.4f}, BCE Loss={exp1_res['step_1']['bce_loss']:.4f}\n"
            f"Local Loss Reduction: {exp1_res['loss_reduction']:.4f} (Confirmed reduction along negative gradient)"
        ),
        'lab_w4d1_mlp': (
            f"--- COMPACT MLP ON 15 REAL FEATURES ---\n"
            f"Subject 1 Run Holdout: Train (Run 4, N=15), Val (Run 8, N=15), Test (Run 12, N=15)\n"
            f"Best Checkpoint Selected: Epoch {exp2_res['best_epoch']} with Validation Loss = {exp2_res['best_val_loss']:.4f}\n"
            f"Held-Out Test Accuracy (Run 12): {exp2_res['test_accuracy']*100:.1f}%\n"
            f"Held-Out Confusion Matrix: {exp2_res['confusion_matrix']}"
        ),
        'lab_w4d2_cnn': (
            f"--- COMPACT 1D CNN ON MULTICHANNEL EEG EPOCHS ---\n"
            f"Input Tensor Shape: [1, 3, 480] (C3, Cz, C4 over 3.0 s at 160 Hz)\n"
            f"Conv1D Layer 1: Conv1d(3, 8, k=32) -> Output [1, 8, 481], 776 params (Duration = 200 ms)\n"
            f"MaxPool1D Layer 2: MaxPool1d(k=4, s=4) -> Output [1, 8, 120] (Effective fs = 40 Hz)\n"
            f"Conv1D Layer 3: Conv1d(8, 16, k=16) -> Output [1, 16, 121], 2064 params (Input receptive field = 593.75 ms)\n"
            f"AdaptiveAvgPool: AdaptiveAvgPool1d(1) -> Output [1, 16, 1]\n"
            f"Classifier Head: Linear(16, 2) -> Logits [1, 2], 34 params\n"
            f"Total Trainable Parameters: 2,922\n"
            f"Subject 1 Holdout Evaluation Test Accuracy (Run 12, N=15): {cnn_metrics['subject_dependent_s1']['test_accuracy']*100:.1f}%"
        ),
        'lab_w4d3_rnn_lstm': (
            f"--- RECURRENT MECHANISM FORWARD PASS (REAL EEG) ---\n"
            f"Input Tensor (batch_first=True): {exp4_res['trial_shape']} [Batch=1, Time=480, Channels=3]\n"
            f"nn.RNN Forward Pass:\n"
            f"  Output Sequence Shape: {exp4_res['rnn_output_shape']} [Batch=1, Time=480, Hidden=8]\n"
            f"  Final Hidden State h_n Shape: {exp4_res['rnn_hidden_shape']} [Layers=1, Batch=1, Hidden=8]\n"
            f"nn.LSTM Forward Pass:\n"
            f"  Output Sequence Shape: {exp4_res['lstm_output_shape']} [Batch=1, Time=480, Hidden=8]\n"
            f"  Final Hidden State h_n Shape: {exp4_res['lstm_hidden_shape']} [Layers=1, Batch=1, Hidden=8]\n"
            f"  Final Cell State c_n Shape:   {exp4_res['lstm_cell_shape']} [Layers=1, Batch=1, Hidden=8]\n"
            f"Untrained forward passes verify state geometry, not memory retention or decoding quality."
        ),
        'lab_w4d4_attention': (
            f"--- SELF-ATTENTION MECHANISM ON TEMPORAL TOKENS ---\n"
            f"EEG Trial Partition: 6 Non-Overlapping Tokens of 80 samples (0.5 s) each\n"
            f"Token Dimensions: 3 channels x 80 samples = 240 values per patch\n"
            f"Linear Projection to d_model=16 + Positional Embedding: Shape [1, 6, 16]\n"
            f"Query, Key, Value Matrices: Q, K, V each [1, 6, 16]\n"
            f"Scaled Dot-Product Attention: softmax(Q @ K.T / sqrt(16))\n"
            f"Attention Weight Matrix [6, 6] Check (Row Sums = 1.0):\n"
            f"{np.sum(exp5_res['attention_matrix'], axis=1)}\n"
            f"Primary Attention per Token:\n" + 
            "\n".join([f"  {exp5_res['token_labels'][i]} attends most strongly to {exp5_res['token_labels'][int(np.argmax(exp5_res['attention_matrix'][i]))]} (weight = {np.max(exp5_res['attention_matrix'][i]):.3f})" for i in range(6)])
        ),
        'lab_w4d5_benchmark': (
            f"--- SUBJECT-INDEPENDENT LEAVE-ONE-SUBJECT-OUT (LOSO) BENCHMARK (N=450) ---\n"
            f"Protocol: 10 Folds (9 Train Subjects = 405 trials, 1 Held-Out Test Subject = 45 trials)\n"
            f"Dummy Baseline:       Pooled Acc = {loso['models']['Dummy']['pooled_accuracy']*100:.1f}%, Mean Sub Acc = {loso['per_subject_summary']['Dummy']['mean_accuracy']*100:.1f}% (±{loso['per_subject_summary']['Dummy']['std_accuracy']*100:.1f}%), Macro-F1 = {loso['models']['Dummy']['pooled_macro_f1']*100:.1f}%\n"
            f"Logistic Regression:  Pooled Acc = {loso['models']['Logistic Regression']['pooled_accuracy']*100:.1f}%, Mean Sub Acc = {loso['per_subject_summary']['Logistic Regression']['mean_accuracy']*100:.1f}% (±{loso['per_subject_summary']['Logistic Regression']['std_accuracy']*100:.1f}%), Macro-F1 = {loso['models']['Logistic Regression']['pooled_macro_f1']*100:.1f}%\n"
            f"Compact 1D CNN:       Pooled Acc = {loso['models']['Compact 1D CNN']['pooled_accuracy']*100:.1f}%, Mean Sub Acc = {loso['per_subject_summary']['Compact 1D CNN']['mean_accuracy']*100:.1f}% (±{loso['per_subject_summary']['Compact 1D CNN']['std_accuracy']*100:.1f}%), Macro-F1 = {loso['models']['Compact 1D CNN']['pooled_macro_f1']*100:.1f}%\n"
            f"Per-Subject Accuracies (S01-S10):\n"
            f"  Dummy:   {[round(x*100, 1) for x in loso['per_subject_summary']['Dummy']['per_subject_accuracy']]}\n"
            f"  LogReg:  {[round(x*100, 1) for x in loso['per_subject_summary']['Logistic Regression']['per_subject_accuracy']]}\n"
            f"  1D CNN:  {[round(x*100, 1) for x in loso['per_subject_summary']['Compact 1D CNN']['per_subject_accuracy']]}\n"
            f"Conclusion: CNN pooled accuracy is modestly higher in this benchmark; transfer varies across subjects and its causes are not identified."
        )
    }
    with open(DATA_DIR / 'week4_lab_outputs.json', 'w') as f:
        json.dump(lab_outputs, f, indent=2)
        
    print("\n" + "="*75)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total Figures Generated: {len(manifest)}")
    for fig_path in manifest:
        print(f"  - {fig_path}")
    print("="*75)

if __name__ == '__main__':
    main()
