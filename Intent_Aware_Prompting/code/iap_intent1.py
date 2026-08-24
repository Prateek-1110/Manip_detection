import os
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# API Key Setup
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please check your .env file.")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# Model Selection
gemini_model = "google/gemini-2.0-flash-lite-preview-02-05:free"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), 'Dataset')

# Import Dataset
csv_path = os.path.join(DATASET_DIR, 'test.csv')
if not os.path.exists(csv_path):
    fallback_path = os.path.join(DATASET_DIR, 'mentalmanip_con_cleaned.csv')
    if os.path.exists(fallback_path):
        print(f"[Info] test.csv not found in {DATASET_DIR}. Falling back to mentalmanip_con_cleaned.csv")
        csv_path = fallback_path
    else:
        raise FileNotFoundError(f"Neither test.csv nor mentalmanip_con_cleaned.csv was found in {DATASET_DIR}")

test = pd.read_csv(csv_path)
# SAFETY LIMIT: Reduce rows to prevent exhausting OpenRouter credits (Change or remove to run full dataset)
LIMIT_ROWS = 100 
if LIMIT_ROWS and len(test) > LIMIT_ROWS:
    print(f"[Safety Warning] Truncating dataset from {len(test)} to {LIMIT_ROWS} rows to save API credits.")
    test = test.head(LIMIT_ROWS).copy()

# Constructor: Person1 Intent
def intent_p1(data):
    system_prompt = """
    I will provide you with a dialogue. 
    Please summarize the intent of the statement made by Person1 in one sentence.
    """
    
    def analyze_dialogue(dialogue):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=gemini_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": dialogue}
                    ],
                    temperature=0.1,
                    top_p=0.5,
                    max_tokens=100
                )
                time.sleep(2) # Respect OpenRouter rate limits
                return response.choices[0].message.content.strip()
            
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    print(f"Rate limit hit. Waiting 30s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(30)
                else:
                    print(f"Error: {e}")
                    return "Error extracting intent."
        return "Failed after retries."

    print(f"Extracting Person 1 intents using {gemini_model}...")
    data['Intent_p1'] = data['Dialogue'].apply(analyze_dialogue)
    data.to_csv(os.path.join(DATASET_DIR, 'intent1_gemini-2.0-flash-lite.csv'), index=False)
    return data

if __name__ == "__main__":
    print("------Person1 Intent------")
    intent1 = intent_p1(test)
    print(intent1.head())
