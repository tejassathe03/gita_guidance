import os
import warnings
import logging
from openai import OpenAI

# Suppress non-critical warnings and logging
warnings.filterwarnings("ignore", category=UserWarning, message=".*gensim.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*sentencepiece.*")
warnings.filterwarnings("ignore", message=".*Mapper.*AbstractNominal.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress noisy logging from dependencies
logging.getLogger("root").setLevel(logging.ERROR)
logging.getLogger("gensim").setLevel(logging.ERROR)
logging.getLogger("sentencepiece").setLevel(logging.ERROR)

# 1. Import our custom IAST transliterator from the previous file
from transliteration_engine import devanagari_to_iast

# 2. Import Sanskrit Parser for Sandhi Splitting
try:
    from sanskrit_parser.parser.sandhi_analyzer import LexicalSandhiAnalyzer
    from sanskrit_parser.base.sanskrit_base import SanskritObject, DEVANAGARI
    HAS_PARSER = True
except ImportError:
    HAS_PARSER = False
    # Silently fall back to LLM for Sandhi splitting

# Initialize the API Client (lazy-loaded to avoid errors if key is not set)
# To use Groq instead of OpenAI, change the base_url to "https://api.groq.com/openai/v1"
# and use your Groq API key.
_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please set it before running this script."
            )
        _client = OpenAI(
            api_key=api_key, 
            base_url="https://api.groq.com/openai/v1" # Routes directly to Groq
        )
    return _client

def get_sandhi_splits(verse: str) -> list[str]:
    """
    Uses python-sanskrit-parser to break compound Devanagari words into root words.
    """
    if not HAS_PARSER:
        # Fallback: If the heavy library isn't installed, we pass the raw verse 
        # and let the LLM handle the splitting in the next step.
        return [verse]

    analyzer = LexicalSandhiAnalyzer()
    verse_obj = SanskritObject(verse, encoding=DEVANAGARI)
    
    # Analyze the verse and extract the most probable split
    try:
        graph = analyzer.getSandhiSplits(verse_obj)
        if graph:
            # Get the first (most probable) valid split path
            splits = graph.find_all_paths(1)[0]
            # Convert splits back to readable string format
            return [str(word) for word in splits]
        return [verse]
    except Exception as e:
        print(f"Parser error: {e}")
        return [verse]

def acharya_translation(devanagari_verse: str, model: str = "llama-3.3-70b-versatile") -> str:
    """
    The core engine: Transliterates, splits, and queries the LLM acting as a Paninian Acharya.
    """
    # Step 1: Transliterate to IAST to prevent token fracture
    iast_verse = devanagari_to_iast(devanagari_verse)
    
    # Step 2: Split the Sandhi (Compound words)
    # Note: We split the Devanagari first, then transliterate the resulting roots for the LLM
    root_words_dev = get_sandhi_splits(devanagari_verse)
    root_words_iast = [devanagari_to_iast(word) for word in root_words_dev]
    
    # Step 3: Engineer the Strict System Prompt
    system_prompt = (
        "You are a Paninian grammar engine and expert Sanskrit Acharya. "
        "Your task is to provide highly accurate, mechanical translations of Sanskrit verses. "
        "You will be provided with the transliterated IAST verse, along with its separated root words (Sandhi split). "
        "1. Provide the English meaning for each root word. "
        "2. Combine them into a single, coherent, philosophically accurate English sentence. "
        "Do not add unnecessary commentary. Be precise and structural."
    )

    user_prompt = (
        f"Original Verse (IAST): {iast_verse}\n"
        f"Split Root Words: {', '.join(root_words_iast)}\n\n"
        "Execute the grammatical breakdown and translation."
    )

    # Step 4: Call the LLM (OpenAI or Groq)
    print("Calling the Acharya Engine...\n" + "-"*40)
    client = _get_client()
    response = client.chat.completions.create(
        model=model, # E.g., "llama3-70b-8192" if using Groq
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1 # Low temperature for deterministic, factual translation
    )

    return response.choices[0].message.content

# --- TEST-ONLY ---
'''
if __name__ == "__main__":
    # Test Verse: Brahma Sutra 1.1.2
    # "janmādyasya yataḥ" -> "From whom the origin etc. of this universe proceed."
    test_verse = "जन्माद्यस्य यतः"
    
    print(f"Input Verse: {test_verse}")
    
    try:
        result = acharya_translation(test_verse)
        print("\n[ACHARYA OUTPUT]")
        print(result)
    except Exception as e:
        print(f"\nAPI Error: Ensure your OPENAI_API_KEY (or GROQ_API_KEY) environment variable is set. \nDetails: {e}")
'''