from flask import Flask, render_template, request, jsonify
from translation_grammar_engine import acharya_translation
from transliteration_engine import devanagari_to_iast
import os
import sys
import time
from transformers import AutoTokenizer
import tiktoken

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# Ensure GROQ_API_KEY is set
if not os.environ.get('GROQ_API_KEY'):
    raise ValueError('GROQ_API_KEY environment variable is not set!')

# Pre-load Sarvam tokenizer once to avoid repeated downloads
sarvam_tokenizer = None

def get_sarvam_tokenizer():
    global sarvam_tokenizer
    if sarvam_tokenizer is None:
        try:
            sarvam_tokenizer = AutoTokenizer.from_pretrained('sarvamai/sarvam-2b')
        except Exception as e:
            print(f'Warning: Could not load Sarvam tokenizer: {e}')
    return sarvam_tokenizer


def count_tokens_with_timing(text: str, encoding_name: str = 'cl100k_base') -> tuple:
    start_time = time.time()
    encoding = tiktoken.get_encoding(encoding_name)
    token_count = len(encoding.encode(text))
    elapsed_time = (time.time() - start_time) * 1000
    return token_count, elapsed_time


@app.route('/')
def home():
    return render_template('translation_engine_home.html')


@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json
        verse = data.get('verse', '').strip()

        if not verse:
            return jsonify({'error': 'Please enter a verse'}), 400

        result = acharya_translation(verse)
        return jsonify({'translation': result, 'original_verse': verse})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze_tokens():
    try:
        data = request.json
        verse = data.get('verse', '').strip()

        if not verse:
            return jsonify({'error': 'Please enter a verse'}), 400

        start_trans = time.time()
        iast_verse = devanagari_to_iast(verse)
        trans_time_ms = (time.time() - start_trans) * 1000

        tokens_dev_west, time_dev_west = count_tokens_with_timing(verse)
        tokens_iast_west, time_iast_west = count_tokens_with_timing(iast_verse)

        tokenizer = get_sarvam_tokenizer()
        tokens_dev_sarvam = None
        time_dev_sarvam = None

        if tokenizer is not None:
            try:
                start_sarvam = time.time()
                tokens_dev_sarvam = len(tokenizer.encode(verse))
                time_dev_sarvam = (time.time() - start_sarvam) * 1000
            except Exception as e:
                print(f'Sarvam tokenization error: {e}')

        fracture_ratio = tokens_dev_west / tokens_iast_west if tokens_iast_west > 0 else 0
        iast_total_time = trans_time_ms + time_iast_west

        sarvam_efficiency = None
        speedup_vs_west = None
        if tokens_dev_sarvam is not None:
            sarvam_efficiency = tokens_dev_west / tokens_dev_sarvam
            speedup_vs_west = time_dev_west / time_dev_sarvam if time_dev_sarvam > 0 else 0

        return jsonify({
            'metrics': {
                'transliteration': {
                    'text': iast_verse,
                    'time_ms': round(trans_time_ms, 2)
                },
                'western_llms': {
                    'raw_devanagari': {
                        'tokens': tokens_dev_west,
                        'time_ms': round(time_dev_west, 2)
                    },
                    'iast_transliterated': {
                        'tokens': tokens_iast_west,
                        'time_ms': round(time_iast_west, 2),
                        'total_pipeline_ms': round(iast_total_time, 2)
                    }
                },
                'native_indic_sarvam': {
                    'tokens': tokens_dev_sarvam,
                    'time_ms': round(time_dev_sarvam, 2) if time_dev_sarvam is not None else None
                },
                'efficiency': {
                    'fracture_ratio': round(fracture_ratio, 2),
                    'token_reduction_iast': round((1 - tokens_iast_west / tokens_dev_west) * 100, 1),
                    'sarvam_efficiency': round(sarvam_efficiency, 2) if sarvam_efficiency is not None else None,
                    'speedup_vs_west': round(speedup_vs_west, 2) if speedup_vs_west is not None else None
                }
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
