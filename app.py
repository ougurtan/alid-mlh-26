"""
Recipe Memory - Flask Backend (v3)
QWER Hacks 2026 | Innovate Track

Setup:
    pip install flask google-generativeai

    export GEMINI_API_KEY="your-key-here"

Run:
    python app.py
"""

import os
import json
import subprocess
import urllib.request
import urllib.parse
from flask import Flask, render_template, request, jsonify

# ============================================================
# CONFIG
# ============================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_KEY_HERE")
CPP_BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp", "classifier")

app = Flask(__name__, template_folder="static", static_folder="static")

# ============================================================
# GEMINI
# ============================================================
gemini_model = None

def init_gemini():
    global gemini_model
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        print("✅ Gemini API connected")
    except Exception as e:
        print(f"⚠️  Gemini not configured: {e}")

# ============================================================
# C++ CLASSIFIER
# ============================================================
def classify_dish(description):
    try:
        result = subprocess.run(
            [CPP_BINARY, description],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            print(f"Classifier stderr: {result.stderr}")
            return None
    except FileNotFoundError:
        print(f"⚠️  C++ binary not found at {CPP_BINARY}")
        return None
    except Exception as e:
        print(f"Classifier error: {e}")
        return None

# ============================================================
# GEMINI: Primary identifier + history + recipe + allergens
# ============================================================
def get_dish_info(user_description, classification, allergies):
    if not gemini_model:
        return {
            "identified_dish": "Unknown",
            "error": "Gemini API not configured. Set GEMINI_API_KEY env variable.",
            "history": "Configure Gemini to see dish history.",
            "recipe": {"ingredients": [], "steps": []},
            "allergen_warnings": [],
            "sustainable_swaps": [],
            "youtube_search_query": ""
        }

    ml_context = ""
    all_predictions_str = ""
    if classification:
        preds = classification.get("dish_predictions", [])
        if preds:
            top = preds[0]
            ml_context = f"""
Our ML classifier suggests: {top.get('dish', 'unknown')} 
from {top.get('region', 'unknown')} ({top.get('continent', 'unknown')}) 
with {top.get('confidence', 0):.0%} confidence.
Other ML guesses: {', '.join([p['dish'] for p in preds[1:]])}
NOTE: The classifier only knows 20 dishes. If none fit, identify the correct dish yourself."""
            # Build list of all predicted dish names for multi-analysis
            all_predictions_str = ", ".join([p['dish'] for p in preds[:3]])

    allergy_str = ", ".join(allergies) if allergies else "none specified"

    prompt = f"""A user is trying to remember a dish from their family. Here's their memory:
"{user_description}"

{ml_context}

The user has these allergies/dietary restrictions: {allergy_str}

YOUR JOB: Return FULL information for the top match AND for each of these other predicted dishes: {all_predictions_str}

Each dish needs: history, recipe, allergen analysis, sustainable swaps, etc.

ALLERGEN RISK: Consider ALL common recipe variants. If a dish has versions with and without the allergen, the risk should be proportional (e.g. 50% if half of common recipes contain it).

Respond in EXACTLY this JSON format. No markdown, no code blocks, ONLY raw JSON:
{{
    "dishes": [
        {{
            "dish_name": "top match dish name",
            "region_of_origin": "region",
            "continent": "continent",
            "allergen_risk_percent": 0,
            "risk_explanation": "why this risk level",
            "history": "Warm 2-3 paragraph history. Write like a grandmother.",
            "recipe": {{
                "servings": "4",
                "prep_time": "time",
                "cook_time": "time",
                "ingredients": ["ingredient 1 with amount", "ingredient 2"],
                "steps": ["step 1", "step 2", "step 3"]
            }},
            "allergen_warnings": [
                {{
                    "allergen": "name",
                    "found_in": "ingredient",
                    "severity": "high or moderate",
                    "substitute": "safe alternative",
                    "variant_note": "which variants safe vs unsafe"
                }}
            ],
            "safe_variants": ["safe versions"],
            "unsafe_variants": ["unsafe versions"],
            "sustainable_swaps": [
                {{
                    "original": "ingredient",
                    "swap": "alternative",
                    "why": "benefit",
                    "impact": "savings"
                }}
            ],
            "youtube_search_query": "search query",
            "tips": "cooking tips",
            "fun_fact": "fun fact"
        }},
        {{
            "dish_name": "second predicted dish",
            "region_of_origin": "region",
            "continent": "continent",
            "allergen_risk_percent": 0,
            "risk_explanation": "why",
            "history": "2-3 paragraph history",
            "recipe": {{"servings":"4","prep_time":"time","cook_time":"time","ingredients":[],"steps":[]}},
            "allergen_warnings": [],
            "safe_variants": [],
            "unsafe_variants": [],
            "sustainable_swaps": [],
            "youtube_search_query": "query",
            "tips": "tips",
            "fun_fact": "fact"
        }},
        {{
            "dish_name": "third predicted dish",
            "region_of_origin": "region",
            "continent": "continent",
            "allergen_risk_percent": 0,
            "risk_explanation": "why",
            "history": "2-3 paragraph history",
            "recipe": {{"servings":"4","prep_time":"time","cook_time":"time","ingredients":[],"steps":[]}},
            "allergen_warnings": [],
            "safe_variants": [],
            "unsafe_variants": [],
            "sustainable_swaps": [],
            "youtube_search_query": "query",
            "tips": "tips",
            "fun_fact": "fact"
        }}
    ]
}}

IMPORTANT:
- The "dishes" array must have one entry per predicted dish (up to 3)
- Each entry must have FULL history, recipe, allergens, swaps — not abbreviated
- allergen_risk_percent must be a NUMBER (0-100)
- Consider ALL recipe variants when calculating risk
- Include 2-3 sustainable_swaps per dish
- Histories should be warm and detailed"""

    try:
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()

        # Clean markdown formatting
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        # Sometimes Gemini adds trailing commas or extra text
        # Find the JSON object boundaries
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        return json.loads(text)

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        raw = response.text if response else "None"
        print(f"Raw: {raw[:500]}")
        return {
            "identified_dish": "Unknown",
            "history": raw,
            "recipe": {"ingredients": [], "steps": []},
            "allergen_warnings": [],
            "allergen_risk_percent": 0,
            "safe_variants": [],
            "unsafe_variants": [],
            "sustainable_swaps": [],
            "youtube_search_query": "",
            "tips": ""
        }
    except Exception as e:
        print(f"Gemini error: {e}")
        return {
            "identified_dish": "Error",
            "error": str(e),
            "history": f"Error: {str(e)}",
            "recipe": {"ingredients": [], "steps": []},
            "allergen_warnings": [],
            "allergen_risk_percent": 0,
            "safe_variants": [],
            "unsafe_variants": [],
            "sustainable_swaps": [],
            "youtube_search_query": "",
            "tips": ""
        }

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/recover", methods=["POST"])
def recover_recipe():
    data = request.json
    description = data.get("description", "").strip()
    allergies = data.get("allergies", [])

    if not description:
        return jsonify({"error": "Please describe the dish you remember!"}), 400

    classification = classify_dish(description)
    dish_info = get_dish_info(description, classification, allergies)

    return jsonify({
        "classification": classification,
        "dish_info": dish_info
    })

@app.route("/api/dish-image")
def dish_image():
    """Get a dish image from Wikipedia. Returns JSON with image URL."""
    dish = request.args.get("dish", "food")
    try:
        # Wikipedia API: search for page and get main image
        encoded = urllib.parse.quote(dish)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "ALLERGUESS/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            image_url = data.get("thumbnail", {}).get("source", "")
            # Get higher res version by modifying the thumbnail URL
            if image_url:
                image_url = image_url.replace("/220px-", "/400px-")
            return jsonify({"image_url": image_url, "dish": dish})
    except Exception as e:
        # Try with "(food)" suffix if plain name fails
        try:
            encoded = urllib.parse.quote(dish + " (food)")
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
            req = urllib.request.Request(url, headers={"User-Agent": "ALLERGUESS/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                image_url = data.get("thumbnail", {}).get("source", "")
                if image_url:
                    image_url = image_url.replace("/220px-", "/400px-")
                return jsonify({"image_url": image_url, "dish": dish})
        except:
            return jsonify({"image_url": "", "dish": dish})

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "gemini": gemini_model is not None,
        "classifier": os.path.exists(CPP_BINARY)
    })

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🍽️  ALLERGUESS — Innovate Track")
    print("   QWER Hacks 2026")
    print("="*50 + "\n")

    init_gemini()

    if not os.path.exists(CPP_BINARY):
        print(f"\n⚠️  C++ classifier not compiled!")
        print(f"   Run: cd cpp && g++ -o classifier classifier.cpp\n")
    else:
        print("✅ C++ classifier found")

    print(f"\n🚀 Starting server at http://localhost:5000\n")
    app.run(debug=True, port=5000, host="0.0.0.0")
