from flask import Flask
from flask import jsonify
from flask import request
import csv


app = Flask(__name__)

@app.route('/')
def root():
    return 'navigate to /api'

@app.route('/api')
def api():
    data = []

    with open('London.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        
        for entry in reader:
            data.append(dict(entry))

    if request.args:
        index = int(request.args.get('index'))
        limit = int(request.args.get('limit'))
    
        return jsonify({'data': data[index:limit + index]})
    else:
        return jsonify({'data': data})


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
