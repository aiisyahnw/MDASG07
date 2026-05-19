"""
ASG 05 – Main Pipeline
Orchestrates the entire machine learning workflow.
"""

from src.data_ingestion import ingest_data
from src.pre_processing import preprocess
from src.train import train
from src.evaluation import evaluate


def run_main_pipeline():
    print("Memulai Pipeline Spaceship Titanic")
    print("_" * 40)

    # Step 1: Data Ingestion
    print("Step 1: Data Ingestion...")
    ingest_data()

    # Step 2: Preprocessing
    print("Step 2: Data Preprocessing...")
    train_data, test_data, preprocessor = preprocess()

    # Step 3: Training
    print("Step 3: Model Training...")
    pipeline = train(train_data, preprocessor)

    # Step 4: Evaluation
    print("Step 4: Model Evaluation...")
    accuracy, precision, recall, f1 = evaluate(test_data, pipeline)

    print("_" * 40)
    print("Pipeline selesai!")

    # Print summary
    print("\nFINAL RESULTS: ")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}  <-- main metric")

    # Decision based on F1-score
    if f1 > 0.70:
        print("Status: Model layak digunakan.")
    else:
        print("Status: Model belum mumpuni.")


if __name__ == "__main__":
    run_main_pipeline()