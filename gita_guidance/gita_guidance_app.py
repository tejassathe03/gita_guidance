from flask import Flask, render_template, request, jsonify
from gita_guidance_engine import get_gita_guidance_for_problem
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# Ensure GROQ_API_KEY is set
if not os.environ.get('GROQ_API_KEY'):
    raise ValueError('GROQ_API_KEY environment variable is not set!')


@app.route('/')
def home():
    return render_template('gita_guidance_home.html')


@app.route('/gita-guidance', methods=['POST'])
def gita_guidance():
    try:
        data = request.json
        problem = data.get('problem', '').strip()

        if not problem:
            return jsonify({'error': 'Please describe your challenge'}), 400

        result = get_gita_guidance_for_problem(problem)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
