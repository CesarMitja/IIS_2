import os

def test_processed_data_exists():
    assert os.path.exists('data/processed'), "Processed data directory does not exist"
    assert len(os.listdir('data/processed')) > 0, "No data files in processed data directory"

def test_model_file_exists():
    assert os.path.exists('models'), "Models directory does not exist"
    assert len(os.listdir('models')) > 0, "No model files in models directory"