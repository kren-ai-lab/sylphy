"""Quick start example for Sylphy sequence encoders."""

from __future__ import annotations

import warnings

import polars as pl

from sylphy.sequence_encoder import (
    FFTEncoder,
    FrequencyEncoder,
    KMerEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PhysicochemicalEncoder,
)
from sylphy.sequence_encoder.factory import create_encoder

warnings.filterwarnings("ignore")

DATA = [
    {"id_seq": 1, "sequence": "AFTGTGGGSSGHYT"},
    {"id_seq": 2, "sequence": "LPLPLKKLKMMNVN"},
    {"id_seq": 3, "sequence": "SASDRRDDQWSED"},
]


def main() -> None:
    df = pl.DataFrame(DATA)

    one_hot = OneHotEncoder(dataset=df, sequence_column="sequence", max_length=20)
    one_hot.run_process()
    print("one_hot:", one_hot.coded_dataset.shape)

    ordinal = OrdinalEncoder(dataset=df, sequence_column="sequence", max_length=20)
    ordinal.run_process()
    print("ordinal:", ordinal.coded_dataset.shape)

    frequency = FrequencyEncoder(dataset=df, sequence_column="sequence")
    frequency.run_process()
    print("frequency:", frequency.coded_dataset.shape)

    kmer = KMerEncoder(dataset=df, sequence_column="sequence", size_kmer=4)
    kmer.run_process()
    print("kmer:", kmer.coded_dataset.shape)

    phys = PhysicochemicalEncoder(dataset=df, sequence_column="sequence", max_length=20)
    phys.run_process()
    print("physicochemical:", phys.coded_dataset.shape)

    fft = FFTEncoder(dataset=phys.coded_dataset, sequence_column="sequence")
    fft.run_process()
    print("fft:", fft.coded_dataset.shape)

    one_hot_factory = create_encoder("onehot", dataset=df, sequence_column="sequence", max_length=20)
    one_hot_factory.run_process()
    print("factory (onehot):", one_hot_factory.coded_dataset.shape)


if __name__ == "__main__":
    main()
