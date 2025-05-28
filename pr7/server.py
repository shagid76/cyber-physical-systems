from flask import Flask, request, jsonify, render_template
import random
from ant import City, AntColony

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['GET'])
def run_ant_colony():
    try:
        n = int(request.args.get('n', 10))
        ants = int(request.args.get('ants', 10))
        iterations = int(request.args.get('iterations', 100))
        decay = float(request.args.get('decay', 0.1))

        cities = [City(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
        colony = AntColony(cities, n_ants=ants, n_iterations=iterations, decay=decay)
        best_path, best_distance = colony.run()

        return jsonify({
            'best_path': best_path,
            'best_distance': round(best_distance, 2),
            'city_coordinates': [{'x': c.x, 'y': c.y} for c in cities]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)