import numpy as np
import polars as pl
from scipy import signal

def adaptive_filter(
    df: pl.DataFrame,
    fs: float,
    channels: list[str],
    time_col: str = "time",
    noise_db_thresh: float = 6.0,
    min_low_cut_hz: float = 5.0,
    min_bandwidth_hz: float = 50.0,
    line_freq: float = 60.0,
    max_freq: float | None = None,
):

    filtered_df = df.clone()
    metadata = {}

    for ch in channels:
        x = df[ch].to_numpy()

        # --- 1. PSD estimation ---
        freqs, psd = signal.welch(
            x,
            fs=fs,
            nperseg=int(fs * 2),
            noverlap=int(fs),
            scaling="density",
        )

        if max_freq is not None:
            mask = freqs <= max_freq
            freqs, psd = freqs[mask], psd[mask]

        psd_db = 10 * np.log10(psd + 1e-12)

        # --- 2. Robust noise floor ---
        noise_floor_db = np.median(psd_db)

        # --- 3. Signal-bearing frequencies ---
        signal_mask = psd_db > (noise_floor_db + noise_db_thresh)

        if not np.any(signal_mask):
            # Fallback: no detectable signal
            metadata[ch] = {"status": "no_signal_detected"}
            continue

        # --- 4. Identify contiguous bands ---
        bands = []
        start = None
        for i, val in enumerate(signal_mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                bands.append((freqs[start], freqs[i - 1]))
                start = None
        if start is not None:
            bands.append((freqs[start], freqs[-1]))

        # Select widest band
        band = max(bands, key=lambda b: b[1] - b[0])

        if (band[1] - band[0]) < min_bandwidth_hz:
            metadata[ch] = {"status": "band_too_narrow"}
            continue

        f_lo = max(band[0], min_low_cut_hz)
        f_hi = band[1]

        if f_lo <= 0 or f_hi <= f_lo:
            metadata[ch] = {
                "status": "invalid_band",
                "raw_band": (float(band[0]), float(band[1])),
            }
            continue
        # --- 5. Line noise detection ---
        notch_freqs = []
        for k in range(1, int(freqs[-1] // line_freq) + 1):
            target = k * line_freq
            idx = np.argmin(np.abs(freqs - target))
            if psd_db[idx] > noise_floor_db + noise_db_thresh:
                notch_freqs.append(target)

        # --- 6. Filter construction ---
        sos = []

        bp = signal.butter(
            N=4,
            Wn=[f_lo, f_hi],
            btype="bandpass",
            fs=fs,
            output="sos",
        )
        sos.append(bp)

        for nf in notch_freqs:
            notch = signal.iirnotch(nf, Q=30, fs=fs)
            sos.append(signal.tf2sos(*notch))

        sos = np.vstack(sos)

        # --- 7. Apply filter ---
        x_filt = signal.sosfiltfilt(sos, x)

        filtered_df = filtered_df.with_columns(
            pl.Series(name=ch, values=x_filt)
        )

        metadata[ch] = {
            "band": (float(f_lo), float(f_hi)),
            "noise_floor_db": float(noise_floor_db),
            "notches": notch_freqs,
        }

    return filtered_df, metadata