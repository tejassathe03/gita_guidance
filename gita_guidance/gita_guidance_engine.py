import re
import os
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np
from openai import OpenAI
from gita.utils import get_verse, get_all_verses, get_sanskrit_verse
from gita.utils import verses, sanskrit_verses

# Initialize LLM Client
_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        _client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
    return _client

def get_gita_verse_for_problem(problem:str):
    """Legacy function to get a Gita verse for a given problem using semantic search."""

    # Load a fast, lightweight embedding model (Runs 100% locally)
    # 'all-MiniLM-L6-v2' is tiny, accurate, and optimized for CPU speed.
    print("Loading local semantic search model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Pre-calculate embeddings for all verses in English.
    # In production, you can calculate this once and cache the matrix to a file.
    verse_entries = [
        (chapter_num, verse_num, text)
        for chapter_num, chapter in verses.items()
        for verse_num, text in chapter.items()
    ]
    book_embeddings = model.encode([entry[2] for entry in verse_entries], convert_to_tensor=True)
    print("Book indexed successfully.\n")

    """
    Converts the user query to an embedding, compares it against the book,
    and returns the exact matching chapter and verse.
    """
    # Embed the user's life question
    query_embedding = model.encode(problem, convert_to_tensor=True)

    # Compute cosine similarity scores between the query and all book sentences
    cosine_scores = util.cos_sim(query_embedding, book_embeddings)[0]
    
    # Get the index of the highest score
    best_match_idx = int(np.argmax(cosine_scores.cpu().numpy()))
    highest_score = cosine_scores[best_match_idx].item()
    
    # Guardrail: If the score is too low, the book probably doesn't answer the question
    confidence_threshold = 0.2  # Adjust this threshold based on testing
    if highest_score < confidence_threshold:
        print(f"No relevant verse found. Highest similarity score: {highest_score:.4f}")
        return None

    chapter_num, sentence_num, matched_text = verse_entries[best_match_idx]
    print(f"Best match: Chapter {chapter_num}, Verse {sentence_num} with similarity score {highest_score:.4f}")

    return {
        "chapter_number": chapter_num,
        "sentence_number": sentence_num,
        "text": matched_text
    }

def get_gita_guidance_for_problem(problem: str, model: str = "llama-3.3-70b-versatile") -> dict:
    """
    Takes a real-life problem and suggests relevant Bhagavad Gita verses with wisdom.
    Returns: {
        'problem': original problem,
        'suggested_verse': {'chapter': int, 'verse': int, 'text': str, 'meaning': str},
        'gita_wisdom': str,
        'practical_suggestions': list[str]
    }
    """
    match = get_gita_verse_for_problem(problem)
    
    # Step 1: Query LLM to identify which Gita chapter/verse is most relevant
    system_prompt = (
        "You are a wise, concise, and practical spiritual guide and Bhagavad Gita expert. "
        "When given a real-life problem, you will:\n"
        "1. Identify the most relevant Bhagavad Gita verse (chapter and verse number, always from all 18 chapters).\n"
        "2. Explain how that verse applies to the problem in a profound, direct, and practical way.\n"
        "3. Provide 3-5 crisp, specific, actionable suggestions based on the Gita's wisdom (avoid generic advice, make each suggestion unique and implementable).\n"
        "4. Add a thought provoking question for reflection based on the verse - if applicable\n"
        "Respond ONLY in valid JSON format with no markdown or extra text."
    )

    user_prompt = (
        f"Real-life problem: {problem}\n\n"
        "Respond with JSON in this exact format (and ONLY JSON, no markdown):\n"
        '{{\n'
        '  "chapter": <int>,\n'
        '  "verse": <int>,\n'
        '  "gita_wisdom": "<explanation of how verse applies>",\n'
        '  "practical_suggestions": ["suggestion1", "suggestion2", "suggestion3"]\n'
        '  "reflection_question": "<short question>"\n'
        '}}'
    )

    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5
    )

    # Parse LLM response
    response_text = response.choices[0].message.content.strip()

    # Clean up markdown if present
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()

    llm_data = json.loads(response_text)

    # Step 2: Fetch the actual Gita verse using the semantic-search match.
    if match is None:
        raise ValueError("No matching Gita verse found for the problem.")

    chapter = match["chapter_number"]
    verse_num = match["sentence_number"]
    
    # Use gita-py for English and Sanskrit if available
    try:
        verse_text_en = get_verse(chapter, verse_num)
        sanskrit_verse_text = get_sanskrit_verse(chapter, verse_num)
        if verse_text_en and "not available" not in verse_text_en.lower():
            verse_text = verse_text_en
            verse_meaning = ""
        else:
            # Try get_all_verses for full database
            all_verses = get_all_verses(chapter)
            found = False
            for v in all_verses:
                if v.get("chapter") == chapter and v.get("verse") == verse_num:
                    verse_text = v.get("text", f"Bhagavad Gita {chapter}.{verse_num}")
                    verse_meaning = v.get("meaning", "Meaning not available.")
                    found = True
                    break
            if not found:
                verse_text = f"Bhagavad Gita {chapter}.{verse_num}"
                verse_meaning = "Verse not found in database. Please refer to a complete Gita translation."
    except Exception as e:
        print(f"Error fetching verse: {e}")
        verse_text = f"Bhagavad Gita {chapter}.{verse_num}"
        verse_meaning = "Unable to fetch verse details."

    reflection_question = llm_data.get("reflection_question", "")
    
    return {
        "problem": problem,
        "suggested_verse": {
            "chapter": chapter,
            "verse": verse_num,
            "text": sanskrit_verse_text,
            "meaning": verse_text,
            "reflection_question": reflection_question
        },
        "gita_wisdom": llm_data["gita_wisdom"],
        "practical_suggestions": llm_data.get("practical_suggestions", []),
        "reflection_question": reflection_question
    }

if __name__ == "__main__":
    test_problem = "I'm facing constant stress at work due to tight deadlines and perfectionist expectations. How can I find peace?"
    
    print("Testing Gita Guidance Engine...")
    print(f"Problem: {test_problem}\n")
    
    try:
        result = get_gita_verse_for_problem(test_problem)
        print("GITA WISDOM GUIDANCE:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
