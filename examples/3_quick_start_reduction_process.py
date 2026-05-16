"""Quick start example for Sylphy dimensionality reduction."""

from __future__ import annotations

import warnings

import pandas as pd

from sylphy.reductions.factory import get_available_methods, reduce_dimensionality
from sylphy.sequence_encoder import PhysicochemicalEncoder

warnings.filterwarnings("ignore")

DATA = [
    {"id_seq": i, "sequence": seq}
    for i, seq in enumerate(
        [
            "AFTGTGGGSSGHYT",
            "LPLPLKKLKMMNVN",
            "SASDRRDDQWSED",
            "MKWVTFISLLFLFSS",
            "GAVLKVLTTGLPALI",
            "DDDEEEGGGSSSNNN",
            "ACDEFGHIKLMNPQR",
            "STVWYKRHDEACFGI",
        ]
    )
]


def main() -> None:
    df = pd.DataFrame(DATA)

    print("Available linear methods:", get_available_methods(kind="linear"))
    print("Available nonlinear methods:", get_available_methods(kind="nonlinear"))

    encoder = PhysicochemicalEncoder(dataset=df, sequence_column="sequence", max_length=15)
    encoder.run_process()
    feature_cols = [c for c in encoder.coded_dataset.columns if c != "sequence"]
    X = encoder.coded_dataset[feature_cols].to_numpy()
    print(f"\nInput matrix: {X.shape}")

    # PCA (linear)
    model, transformed = reduce_dimensionality(
        "pca", X, n_components=2, return_type="numpy", preprocess="standardize", random_state=0, debug=False
    )
    print(f"PCA: {transformed.shape} | explained variance: {model.explained_variance_ratio_.sum():.3f}")

    # TruncatedSVD (linear, no centering required)
    _, transformed_svd = reduce_dimensionality(
        "truncated_svd", X, n_components=2, return_type="pandas", random_state=0, debug=False
    )
    print(f"TruncatedSVD: {transformed_svd.shape}")
    print(transformed_svd)

    # t-SNE (nonlinear, sklearn — no optional deps)
    _, transformed_tsne = reduce_dimensionality(
        "tsne", X, n_components=2, perplexity=3, return_type="numpy", random_state=0, debug=False
    )
    print(f"t-SNE: {transformed_tsne.shape}")

    # Isomap (nonlinear, sklearn)
    _, transformed_iso = reduce_dimensionality(
        "isomap", X, n_components=2, n_neighbors=3, return_type="numpy", debug=False
    )
    print(f"Isomap: {transformed_iso.shape}")

    # PCA with pandas return and normalization preprocessing
    _, transformed_pd = reduce_dimensionality(
        "pca", X, n_components=3, return_type="pandas", preprocess="normalize", random_state=0, debug=False
    )
    print(f"PCA (pandas, 3 components): {transformed_pd.shape}")
    print(transformed_pd)


if __name__ == "__main__":
    main()
