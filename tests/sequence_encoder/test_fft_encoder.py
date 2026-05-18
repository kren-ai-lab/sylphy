from __future__ import annotations

import polars as pl
import polars.selectors as cs

from sylphy.sequence_encoder import FFTEncoder


def test_fft_encoder_half_spectrum_and_padding() -> None:
    """Verify FFT encoder produces half-spectrum magnitude with zero-padding."""
    df = pl.DataFrame(
        {
            "p_0": [1.0, 0.0],
            "p_1": [0.0, 1.0],
            "p_2": [1.0, 1.0],
            "sequence": ["AAA", "CCC"],
        },
    )
    enc = FFTEncoder(dataset=df, sequence_column="sequence", debug=True)
    enc.run_process()
    X = enc.coded_dataset
    assert X is not None
    # Next power of two >= 3 is 4 → half-spectrum = 2
    numeric = X.select(cs.numeric())
    assert numeric.width == 2
    assert (numeric.to_numpy() >= 0).all()
    assert X["sequence"].to_list() == ["AAA", "CCC"]
