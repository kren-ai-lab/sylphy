import sys
import warnings

import polars as pl

from sylphy.sequence_encoder import (
    KMerEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PhysicochemicalEncoder,
)

warnings.filterwarnings("ignore")


def make_encoder(encoder, name_export):
    encoder.run_process()
    encoder.coded_dataset.write_csv(name_export)


print("Processing data")
df_training = pl.read_excel(sys.argv[1])
df_testing = pl.read_excel(sys.argv[2])

path_export = sys.argv[3]

### One Hot
print("Coding with One Hot")
one_hot_encoder_train = OneHotEncoder(dataset=df_training, sequence_column="seq", max_length=548)

one_hot_encoder_test = OneHotEncoder(dataset=df_testing, sequence_column="seq", max_length=548)

make_encoder(one_hot_encoder_train, f"{path_export}onehot/train_dataset.csv")
make_encoder(one_hot_encoder_test, f"{path_export}onehot/test_dataset.csv")

### Ordinal
print("Coding with Ordinal")
ordinal_encoder_train = OrdinalEncoder(dataset=df_training, sequence_column="seq", max_length=548)

ordinal_encoder_test = OrdinalEncoder(dataset=df_testing, sequence_column="seq", max_length=548)

make_encoder(ordinal_encoder_train, f"{path_export}ordinal/train_dataset.csv")
make_encoder(ordinal_encoder_test, f"{path_export}ordinal/test_dataset.csv")

### KMers
print("Coding with kmers")
kmer_train = KMerEncoder(dataset=df_training, sequence_column="seq")

kmer_test = KMerEncoder(dataset=df_testing, sequence_column="seq")

make_encoder(kmer_train, f"{path_export}kmers/train_dataset.csv")
make_encoder(kmer_test, f"{path_export}kmers/test_dataset.csv")

### Physicochemical
print("Coding with Physicochemical property")
phy_train = PhysicochemicalEncoder(dataset=df_training, sequence_column="seq", max_length=548)

phy_test = PhysicochemicalEncoder(dataset=df_testing, sequence_column="seq", max_length=548)

make_encoder(phy_train, f"{path_export}physicochemical_property/train_dataset.csv")
make_encoder(phy_test, f"{path_export}physicochemical_property/test_dataset.csv")
