import numpy as np
import torch

def numpy_pe(pos, d):
    p = np.asarray(pos)[:,None]
    inv = 10000 ** (-2*np.arange(d//2)/d)
    phase = p*inv[None,:]
    out = np.empty((len(p),d)); out[:,0::2]=np.sin(phase); out[:,1::2]=np.cos(phase)
    return out

torch.manual_seed(42)
pos, d = np.arange(10), 16
manual = numpy_pe(pos,d)
pt = torch.zeros(10,d); p=torch.arange(10,dtype=torch.float32)[:,None]
i=torch.arange(0,d,2,dtype=torch.float32); phase=p*torch.exp(-np.log(10000)*i/d)
pt[:,0::2]=phase.sin(); pt[:,1::2]=phase.cos()
print(f"shape={manual.shape}; max |from scratch - formula implementation|={np.max(np.abs(manual-pt.numpy())):.8f}")
print(f"entry range=[{manual.min():.6f},{manual.max():.6f}]")
