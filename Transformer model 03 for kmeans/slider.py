import numpy as np
def slider(signal, w=100):
    out = []
    m = signal.shape[0]
    for i in range(m-w+1):
        out.append(signal[i:i+w].flatten())
    out = np.array(out)
    return out