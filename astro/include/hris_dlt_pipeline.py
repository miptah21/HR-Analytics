import dlt
import csv
from pathlib import Path

@dlt.resource(name="employee_attrition_records", write_disposition="replace")
def hris_source():
    """
    Simulates extracting data from an HRIS API (like Workday or BambooHR)
    by reading the local CSV dataset. In a real production environment,
    this generator would paginate through API responses.
    """
    # In Airflow/Astro, include/ is mapped, so we resolve path relative to this script
    csv_path = Path(__file__).parent / "datasets" / "HR-Employee-Attrition.csv"
    
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Yielding rows directly allows dlt to infer schema and load efficiently
            yield row

def run_pipeline():
    """
    Executes the dlt pipeline.
    Lands the data into the `raw_hris` schema in PostgreSQL.
    """
    import os
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError(
            "DATABASE_URL environment variable is required. "
            "Set it to your PostgreSQL connection string."
        )

    pipeline = dlt.pipeline(
        pipeline_name="hris_ingestion",
        destination=dlt.destinations.postgres(credentials=db_url),
        dataset_name="raw_hris" # schema name in Postgres
    )
    
    # Run the pipeline with the simulated HRIS source
    load_info = pipeline.run(hris_source())
    print(f"Pipeline ran successfully. Load info: {load_info}")
    return load_info

if __name__ == "__main__":
    run_pipeline()
