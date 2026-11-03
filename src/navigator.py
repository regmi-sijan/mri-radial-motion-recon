"""Extract respiratory motion from the repeated k-space center."""
import numpy as np


def navigate(samples, tr, states):
    dc = samples[:, :, samples.shape[-1] // 2]
    traces = np.concatenate((dc.real, dc.imag), axis=0)
    # Retain Fourier modes in the respiratory band. This zero-phase filter
    # uses a periodic extension, so scan boundary effects are possible.
    frequencies = np.fft.rfftfreq(dc.shape[1], tr)
    band = (frequencies >= .1) & (frequencies <= .5)
    filtered = np.fft.irfft(np.fft.rfft(traces, axis=-1) * band,
                            n=dc.shape[1], axis=-1)
    loadings, _, _ = np.linalg.svd(filtered, full_matrices=False)
    signal = loadings[:, 0] @ filtered
    # In this phantom the inferior coil gains signal during inspiration.
    inferior = len(dc) // 4
    phase = np.angle(dc[inferior].mean())
    reference = (dc[inferior] * np.exp(-1j * phase)).real
    reference = np.fft.irfft(np.fft.rfft(reference) * band, n=len(reference))
    if signal @ reference < 0:
        signal = -signal
    return signal, np.array_split(np.argsort(signal, kind="stable"), states)
