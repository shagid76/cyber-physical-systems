from unittest.mock import patch

import pandas as pd
from main import app
from temperature_predictor import TemperaturePredictor


def test_train():
    predictor = TemperaturePredictor(file_path="dummy.csv")

    mock_data = pd.DataFrame({
        "timestamp": ["20250329T1200", "20250329T1300", "20250329T1400", "20250329T1500"],
        "temperature": [15.5, 16.2, 16.8, 17.0]
    })

    with patch("pandas.read_csv", return_value=mock_data):
        mae = predictor.train()

    assert mae >= 0


def test_forecast_20_years(mock_predictor):
    with patch("main.TemperaturePredictor", return_value=mock_predictor):
        with app.test_client() as client:
            response = client.get('/temperature/forecast/20years')

            assert response.status_code == 200

            json_data = response.get_json()
            assert isinstance(json_data, list)
            assert len(json_data) > 0
            assert "date" in json_data[0]
            assert "predicted_temperature" in json_data[0]
