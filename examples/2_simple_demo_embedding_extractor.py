import pandas as pd

from sylphy.embedding_extractor import (
    Ankh2BasedEmbedding,
    BertBasedEmbedding,
    ESMBasedEmbedding,
    ESMCBasedEmbedding,
    MistralBasedEmbedding,
    Prot5Based,
)


def make_toy_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "sequence": ["MKT", "ACDEFGHIKLMNPQRST", "GGGSSSPPP", "MPEPTIDESEQX"],
        }
    ).set_index("id")


# --- ESM-C ---
def run_backend_esmc(backend, name_out):
    """
    ESM-C uses its own `embedding_process` method and returns a DataFrame.
    We assign it to `coded_dataset` for consistency with other backends
    and then reuse `export_encoder(...)`.
    """
    df_emb = backend.embedding_process(
        batch_size=4,  # tune for your VRAM
        seq_len=512,  # optional: pad/truncate before encoding
        layers="last",  # "last" | "last4" | "all" | int | [ints]
        layer_agg="mean",  # "mean" | "sum" | "concat"
        pool="mean",  # "mean" | "cls" | "eos"
    )
    backend.coded_dataset = df_emb  # keep same API shape as other backends
    print(f"[OK][ESMC] Shape = {backend.coded_dataset.shape}")
    print(backend.coded_dataset.head(2))
    backend.export_encoder(f"./emb_{name_out}.csv", file_format="csv")


def run_backend(backend, name_out):
    backend.run_process(
        max_length=512,
        batch_size=4,
        layers="last",
        layer_agg="mean",
        pool="mean",
    )
    print(f"[OK] Shape = {backend.coded_dataset.shape}")
    print(backend.coded_dataset.head(2))
    backend.export_encoder(f"./emb_{name_out}.csv", file_format="csv")


def main():
    df = make_toy_df()

    # --- ESM2 ---
    esm2 = ESMBasedEmbedding(
        dataset=df,
        column_seq="sequence",
        name_model="facebook/esm2_t6_8M_UR50D",
        name_tokenizer="facebook/esm2_t6_8M_UR50D",
        name_device="cuda",
        precision="fp16",
        oom_backoff=True,
        debug=False,
    )
    run_backend(esm2, "esm2_last_mean_mean")

    # --- ProtT5 ---
    prot5 = Prot5Based(
        dataset=df,
        column_seq="sequence",
        name_model="Rostlab/prot_t5_xl_uniref50",
        name_tokenizer="Rostlab/prot_t5_xl_uniref50",
        name_device="cuda",
        precision="fp16",
        oom_backoff=True,
        debug=False,
    )
    run_backend(prot5, "protT5_last_mean_mean")

    # --- Bert ---
    bert = BertBasedEmbedding(
        dataset=df,
        column_seq="sequence",
        name_model="Rostlab/prot_bert",
        name_tokenizer="Rostlab/prot_bert",
        name_device="cuda",
        precision="fp16",
        oom_backoff=True,
        debug=False,
    )
    run_backend(bert, "bert_last_mean_mean")

    # --- Mistral ---
    mistral = MistralBasedEmbedding(
        dataset=df,
        column_seq="sequence",
        name_model="RaphaelMourad/Mistral-Prot-v1-134M",
        name_tokenizer="RaphaelMourad/Mistral-Prot-v1-134M",
        name_device="cuda",
        precision="fp32",
        oom_backoff=True,
        debug=False,
    )
    run_backend(mistral, "mistral_last_mean_mean")

    # --- Ankh ---
    ankh = Ankh2BasedEmbedding(
        dataset=df,
        column_seq="sequence",
        name_model="ElnaggarLab/ankh2-ext1",
        name_tokenizer="ElnaggarLab/ankh2-ext1",
        name_device="cuda",
        precision="fp32",
        oom_backoff=True,
        debug=False,
    )
    run_backend(ankh, "ankh_last_mean")

    esmc = ESMCBasedEmbedding(
        dataset=df,
        column_seq="sequence",
        name_model="esmc_300m",  # or any registered ESM-C key you resolved
        name_device="cuda",  # falls back to CPU if CUDA not available
        precision="fp32",  # "fp32" | "fp16" | "bf16"
        oom_backoff=True,
        debug=False,
    )
    run_backend_esmc(esmc, "esmc_last_mean_mean")


if __name__ == "__main__":
    main()
