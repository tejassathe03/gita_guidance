import tiktoken
from indic_transliteration import sanscript

def devanagari_to_iast(text: str) -> str:
    """Converts Devanagari to IAST."""
    return sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def evaluate_token_fracture():
    # The Texts (Using Bhagavad Gita 2.47)
    english_translation = "You have a right to perform your prescribed duty, but you are not entitled to the fruits of action. Never consider yourself the cause of the results of your activities, and never be attached to not doing your duty."
    devanagari_verse = "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन। मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि॥"
    
    # Transliterate
    iast_verse = devanagari_to_iast(devanagari_verse)
    
    # Count Tokens
    tokens_eng = count_tokens(english_translation)
    tokens_dev = count_tokens(devanagari_verse)
    tokens_iast = count_tokens(iast_verse)
    
    # Calculate the fracture ratio
    fracture_ratio = (tokens_dev / tokens_iast) * 100
    
    # Print Clean Terminal Table
    print("\n" + "=" * 80)
    print(f"{'FORMAT':<20} | {'TOKENS':<8} | {'TEXT SNIPPET'}")
    print("-" * 80)
    print(f"{'English Translation':<20} | {tokens_eng:<8} | {english_translation[:45]}...")
    print(f"{'Devanagari Script':<20} | {tokens_dev:<8} | {devanagari_verse[:45]}...")
    print(f"{'IAST Transliteration':<20} | {tokens_iast:<8} | {iast_verse[:45]}...")
    print("=" * 80)
    
    print("\n[ARCHITECTURAL INSIGHT]")
    print(f"Standard LLMs fracture Devanagari by {fracture_ratio:.0f}%. To build a true Vedic AI, we must transliterate to IAST before vectorizing.\n")

''' --- IGNORE ---
if __name__ == "__main__":
    evaluate_token_fracture()
'''