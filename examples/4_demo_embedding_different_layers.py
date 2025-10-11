"""
Example: extracting first, last, and specific layers from ESM2.

Notes on layer indexing:
- Many HF models expose hidden_states as: [embeddings, layer1, layer2, ..., layerN]
- That means:
    layers=0           -> embeddings (pre-transformer)
    layers=1           -> first encoder layer
    layers="last"      -> last entry in hidden_states (usually layerN)
    layers="last4"     -> last 4 entries (e.g., [N-3, N-2, N-1, N])
Adjust if your backend defines different semantics.
"""

import pandas as pd
from sylphy.embedding_extractor import ESMBasedEmbedding

def make_toy_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "sequence": [
            "MKT",
            "ACDEFGHIKLMNPQRST",
            "GGGSSSPPP",
            "MPEPTIDESEQX"
        ],
    }).set_index("id")

def run_case(backend, *, layers, layer_agg="mean", pool="mean", tag=""):
    """
    Runs the backend with the requested layer selection / aggregation / pooling,
    prints a quick summary, and exports a CSV.
    """
    backend.run_process(
        max_length=512,
        batch_size=4,
        layers=layers,        # "last" | "last4" | 0 | 1 | [ints]
        layer_agg=layer_agg,  # "mean" | "sum" | "concat"
        pool=pool,            # "mean" | "cls" | "none" (depending on your base)
    )
    print(f"[OK][{tag}] Shape = {backend.coded_dataset.shape}")
    print(backend.coded_dataset.head(2))
    backend.export_encoder(f"./emb_esm2_{tag}.csv", file_format="csv")

def main():
    df = make_toy_df()

    # Instantiate ESM2 once; reuse for different layer selections
    esm2 = ESMBasedEmbedding(
        dataset=df,
        column_seq="sequence",
        name_model="facebook/esm2_t6_8M_UR50D",
        name_tokenizer="facebook/esm2_t6_8M_UR50D",
        name_device="cuda",
        precision="fp16",     # set "fp32" if you prefer
        oom_backoff=True,
        debug=False,
    )

    # 1) LAST layer (typical choice)
    run_case(esm2, layers="last", layer_agg="mean", pool="mean", tag="last_mean")

    # 2) FIRST hidden_states entry (often the embedding layer, pre-transformer)
    #    If you want the *first encoder layer*, use layers=1 (next block).
    run_case(esm2, layers=0, layer_agg="mean", pool="mean", tag="first_entry_mean")

    # 3) FIRST encoder layer explicitly (if hidden_states includes embeddings at index 0)
    run_case(esm2, layers=1, layer_agg="mean", pool="mean", tag="encoder_layer1_mean")

    # 4) LAST 4 layers aggregated
    #    For esm2_t6_8M_UR50D there are 6 transformer layers; "last4" is valid.
    #    Try layer_agg="concat" if you want to concatenate instead of averaging.
    run_case(esm2, layers="last4", layer_agg="mean", pool="mean", tag="last4_mean")

    # Concatenation
    run_case(esm2, layers="last4", layer_agg="concat", pool="mean", tag="last4_concat")
if __name__ == "__main__":
    main()
