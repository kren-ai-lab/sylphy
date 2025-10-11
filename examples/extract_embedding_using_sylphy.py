import pandas as pd
import sys
import torch

from sylphy.embedding_extractor import (
    ESMBasedEmbedding,   
    Prot5Based,          
    BertBasedEmbedding,
    MistralBasedEmbedding,
)

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
    backend.export_encoder(f"{name_out}", file_format="csv")
    torch.cuda.empty_cache()

def main():
    df_train = pd.read_excel(sys.argv[1])
    df_test = pd.read_excel(sys.argv[2])

    path_export = sys.argv[3]

    # --- ESM2 ---
    esm2 = ESMBasedEmbedding(
        dataset=df_train,
        column_seq="seq",
        name_model="facebook/esm2_t6_8M_UR50D",   
        name_tokenizer="facebook/esm2_t6_8M_UR50D",
        name_device="cuda",       # "cuda" o "cpu"
        precision="fp16",         # "fp32" | "fp16" | "bf16"
        oom_backoff=True,         
        debug=False,
    )
    run_backend(esm2, f"{path_export}esm2_embedding/train_dataset.csv")

    esm2 = ESMBasedEmbedding(
        dataset=df_test,
        column_seq="seq",
        name_model="facebook/esm2_t6_8M_UR50D",   
        name_tokenizer="facebook/esm2_t6_8M_UR50D",
        name_device="cuda",       # "cuda" o "cpu"
        precision="fp16",         # "fp32" | "fp16" | "bf16"
        oom_backoff=True,         
        debug=False,
    )
    run_backend(esm2, f"{path_export}esm2_embedding/test_dataset.csv")

    # --- ProtT5 ---
    prot5 = Prot5Based(
        dataset=df_train,
        column_seq="seq",
        name_model="Rostlab/prot_t5_xl_uniref50",
        name_tokenizer="Rostlab/prot_t5_xl_uniref50",
        name_device="cuda",
        precision="fp16",
        oom_backoff=True,
        debug=False,
    )
    run_backend(prot5, f"{path_export}prot5_embedding/train_dataset.csv")

    prot5 = Prot5Based(
        dataset=df_test,
        column_seq="seq",
        name_model="Rostlab/prot_t5_xl_uniref50",
        name_tokenizer="Rostlab/prot_t5_xl_uniref50",
        name_device="cuda",
        precision="fp16",
        oom_backoff=True,
        debug=False,
    )
    run_backend(prot5, f"{path_export}prot5_embedding/test_dataset.csv")

    # --- Bert ---
    bert = BertBasedEmbedding(
        dataset=df_train,
        column_seq="seq",
        name_model="Rostlab/prot_bert",
        name_tokenizer="Rostlab/prot_bert",
        name_device="cuda",
        precision="fp16",
        oom_backoff=True,
        debug=False,
    )
    run_backend(bert, f"{path_export}bert_embedding/train_dataset.csv")

    bert = BertBasedEmbedding(
        dataset=df_test,
        column_seq="seq",
        name_model="Rostlab/prot_bert",
        name_tokenizer="Rostlab/prot_bert",
        name_device="cuda",
        precision="fp16",
        oom_backoff=True,
        debug=False,
    )
    run_backend(bert, f"{path_export}bert_embedding/test_dataset.csv")

    # --- Mistral ---
    mistral = MistralBasedEmbedding(
        dataset=df_train,
        column_seq="seq",
        name_model="RaphaelMourad/Mistral-Prot-v1-134M",
        name_tokenizer="RaphaelMourad/Mistral-Prot-v1-134M",
        name_device="cuda",
        precision="fp32",
        oom_backoff=True,
        debug=False,
    )
    run_backend(mistral, f"{path_export}mistral_embedding/train_dataset.csv")

    mistral = MistralBasedEmbedding(
        dataset=df_test,
        column_seq="seq",
        name_model="RaphaelMourad/Mistral-Prot-v1-134M",
        name_tokenizer="RaphaelMourad/Mistral-Prot-v1-134M",
        name_device="cuda",
        precision="fp32",
        oom_backoff=True,
        debug=False,
    )
    run_backend(mistral, f"{path_export}mistral_embedding/test_dataset.csv")

if __name__ == "__main__":
    main()
