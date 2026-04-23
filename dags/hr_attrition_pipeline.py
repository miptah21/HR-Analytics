from airflow.decorators import dag, task
from pendulum import datetime
import subprocess

@dag(
    dag_id='hr_attrition_batch_scoring',
    schedule='@daily',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['hr_analytics', 'mlops'],
    description='Nightly batch scoring of employee attrition risk'
)
def hr_attrition_pipeline():
    
    @task
    def extract_hris_data():
        # In production, this would query Workday/Snowflake via an Airflow Provider
        print("Extracting live HRIS data...")
        return True

    @task
    def run_attrition_model(success: bool):
        # Trigger the main training/scoring script
        # This will evaluate fairness and save new risk scores to outputs/
        result = subprocess.run(
            ["uv", "run", "python", "src/train_attrition_model.py"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise Exception("Model training and scoring failed.")
            
        return "outputs/risk_scores.csv"

    @task
    def update_dashboard_database(file_path: str):
        # Upsert the new risk scores into the SQL backend driving Power BI
        print(f"Upserting {file_path} to Power BI staging tables...")

    
    data_ready = extract_hris_data()
    scores = run_attrition_model(data_ready)
    update_dashboard_database(scores)

pipeline = hr_attrition_pipeline()
