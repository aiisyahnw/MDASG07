"""
ASG 05 – Step 1: Data Ingestion
Reads raw train.csv and saves it to the ingested/ folder.
"""

from pathlib import Path
import pandas as pd 

#raw data > ingested > processed > model

BASE_DIR = Path(__file__).parent #folder src
INGESTED_DIR = BASE_DIR / "ingested" #bikin folder ingested di src
INPUT_FILE = BASE_DIR.parent / "data/train.csv" #ambil dara train.csv dr folder parent nya src (PROJECT MDASG04)
OUTPUT_FILE = INGESTED_DIR / "spaceship_train.csv" #taro hasil ingest di folder ingested

def ingest_data():
    INGESTED_DIR.mkdir(parents=True, exist_ok=True) #mkdir=buat folder ini. kl blm ada buat folder ingested dir, kl udh ada lgsg aja 

    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return
    
    df = pd.read_csv(INPUT_FILE)
    assert not df.empty, "empty" #kalau dataest kodong > error, ada > lanjut
    df.to_csv(OUTPUT_FILE, index=False)

    print("Berhasil!")
    print(f"Sumber: {INPUT_FILE}")
    print(f"Tujuan: {OUTPUT_FILE}")

if __name__ == "__main__": #kalau file ini di jalanin lgsg,
                            #run fx ingest_data
    ingest_data()