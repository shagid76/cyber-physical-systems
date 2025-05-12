from flask import Flask, jsonify
from temperature_predictor import TemperaturePredictor

app = Flask(__name__)

predictor = TemperaturePredictor(file_path="resources/data.csv", seed=1)

mae = predictor.train()

@app.route('/temperature/forecast/20years', methods=['GET'])
def forecast_20_years():
    try:
        predictions = predictor.predict_for_next_20_years()
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)