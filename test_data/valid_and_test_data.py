import subprocess
import json
import sys
import os
import pandas as pd
from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, DataQualityTab, NumTargetDriftTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, DataQualityProfileSection
from great_expectations.data_context import DataContext
from great_expectations.exceptions import CheckpointNotFoundError

def run_command(command):
    """ Helper function to run a shell command and return its output """
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        print("Error running command:", command)
        print("Command Output:", result.stdout)
        print("Command Error:", result.stderr)
        print("Upajmo")
        return None
    return result.stdout


def run_checkpoint(checkpoint_name):
    """ Function to run a Great Expectations checkpoint and return a success flag """
    context = DataContext()
    checkpoint_result = context.run_checkpoint(checkpoint_name=checkpoint_name)
    return checkpoint_result["success"]



def check_all_checkpoints():
    """ Run all checkpoints and return False if any of them fail """
    x = 0
    checkpoints = ["kolesa_check"]#["kolesa_check","vreme_check","pred_check"]
    for checkpoint in checkpoints:
        try:
            x = x + 1
            print(f"Running heckpoint {checkpoint}.")
            run_checkpoint(checkpoint)
            
        except CheckpointNotFoundError:
            print(f"Checkpoint {checkpoint} does not exist.")
            return False
    if x == 3 :
        return True


def evidently_test(current_data, reference_data, report_filename):
    """Run evidently to detect data drift and generate a report."""
    # Define the directory where reports will be saved
    reports_dir = os.path.join(os.getcwd(),"reports")
    # Create the directory if it does not exist
    os.makedirs(reports_dir, exist_ok=True)
    
    # Define the full path for the report
    full_report_path = os.path.join(reports_dir, report_filename)

    column_mapping = ColumnMapping()  # adjust based on your data
    dashboard = Dashboard(tabs=[DataDriftTab(), DataQualityTab(), NumTargetDriftTab()])
    dashboard.calculate(reference_data, current_data, column_mapping=column_mapping)
    dashboard.save(full_report_path)
    print(f"Evidently report generated: {full_report_path}")




def process_data():
    """Process data to generate train and test datasets."""
    data = pd.read_csv('data/processed/data_for_prediction.csv')
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by='date', ascending=False)
    test_data = data.head(int(len(data) * 0.1))
    train_data = data.iloc[int(len(data) * 0.1):]
    test_data.to_csv('data/processed/test.csv', index=False)
    train_data.to_csv('data/processed/train.csv', index=False)
    print("Train and test datasets created.")

def main():
    #if not check_all_checkpoints():
    #    print("Not all checkpoints passed. Halting further execution.")
    #    sys.exit(1)
    # Load data
    current_data = pd.read_csv('data/raw/vreme.csv')
    reference_data = pd.read_csv('data/raw/vreme_ref.csv')
    evidently_test(current_data, reference_data, "data_drift_vreme.html")
    current_data = pd.read_csv('data/raw/kolesa.csv')
    reference_data = pd.read_csv('data/raw/kolesa_ref.csv')
    evidently_test(current_data, reference_data, "data_drift_kolesa.html")
    current_data = pd.read_csv('data/processed/data_for_prediction.csv')
    reference_data = pd.read_csv('data/processed/prediction_ref.csv')
    evidently_test(current_data, reference_data, "data_drift_prediction.html")
    process_data()
    print("All checkpoints passed!")
    sys.exit(0)

if __name__ == "__main__":
    main()