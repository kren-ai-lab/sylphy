import typer
from bioclust.cli.cache import app as cache_app
from bioclust.cli.encoder_sequences import app as encoder_app
from bioclust.cli.get_embeddings import app as embedding_extractor_app
from bioclust.cli.cluster_characterization import app as cluster_characterization_app
from bioclust.cli.collect_data import app as collector_data_app
from bioclust.cli.predict_structures import app as structure_prediction
from bioclust.cli.run_clustering import app as run_clustering

app = typer.Typer(name="bioclust", add_completion=False, help="A machine learning-based tool to facilitate the clustering of protein sequences")

app.add_typer(
    embedding_extractor_app, 
    name="get-embedding", 
    help="Tool to get embedding from pre-trained protein language models to represent sequences numerically")

app.add_typer(
    encoder_app, 
    name="encode-sequences", 
    help="Tool to represent numerically protein sequences using different approaches")

app.add_typer(
    cache_app, 
    name="cache", 
    help="Inspect and manage cache directory")

app.add_typer(
    cluster_characterization_app, 
    name="cluster-characterization", 
    help="Apply strategies to characterize and compare generated clusters")

app.add_typer(
    collector_data_app, 
    name="data-collector", 
    help="Collecting strategies for datasets, sequences, and protein structures")

app.add_typer(
    structure_prediction, 
    name="structure-predictor", 
    help="Predicting structure from sequences using ESMFold pre-trained model")

app.add_typer(
    run_clustering, 
    name="run-clustering", 
    help="Perform clustering strategies available in bioclust library")

if __name__ == "__main__":
    app()