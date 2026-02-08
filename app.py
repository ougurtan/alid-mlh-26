"""
Recipe Memory - Flask Backend (v2)
Innovate Track: Dish recovery + Allergy safety + Sustainable food future

Changes from v1:
- Gemini is now the PRIMARY dish identifier (knows every dish)
- C++ classifier is SECONDARY (adds ML confidence layer for known dishes)
- Dish images via Unsplash
- Works even if user describes a dish not in training data

Setup:
    pip install flask google-generativeai pymongo

    export GEMINI_API_KEY="your-key-here"
    export MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net/"

Run:
    python app.py
"""

import os
import json
import subprocess
from datetime import datetime
from flask import Flask, render_template, request, jsonify

# ============================================================
# CONFIG
# ============================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_KEY_HERE")
MONGO_URI = os.environ.get("MONGO_URI", "")
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
# MONGODB
# ============================================================
mongo_db = None

def init_mongo():
    global mongo_db
    if not MONGO_URI:
        print("⚠️  MongoDB not configured (app works without it)")
        return
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI)
        mongo_db = client["recipe_memory"]
        client.admin.command('ping')
        print("✅ MongoDB connected")
    except Exception as e:
        print(f"⚠️  MongoDB failed: {e}")

def save_recipe(data):
    if not mongo_db:
        return None
    try:
        data["created_at"] = datetime.utcnow().isoformat()
        result = mongo_db["recovered_recipes"].insert_one(data)
        return str(result.inserted_id)
    except Exception as e:
        print(f"MongoDB save error: {e}")
        return None

def get_recent_recipes(limit=12):
    if not mongo_db:
        return []
    try:
        recipes = list(mongo_db["recovered_recipes"].find(
            {},
            {"_id": 0, "user_description": 1,
             "dish_info.identified_dish": 1,
             "dish_info.region_of_origin": 1,
             "dish_info.continent": 1,
             "dish_info.image_url": 1}
        ).sort("_id", -1).limit(limit))
        return recipes
    except:
        return []

# ============================================================
# C++ CLASSIFIER (secondary — adds ML layer for judges)
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
# DISH IMAGE
# ============================================================
def get_dish_image(dish_name):
    """Get a dish image URL using Unsplash source (no API key needed)."""
    clean_name = dish_name.replace(" ", "+").replace("/", "+")
    return f"https://source.unsplash.com/400x300/?{clean_name}+food+dish"

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
            "youtube_search_query": "",
            "image_url": ""
        }

    # Build context from C++ classifier (secondary info for Gemini)
    ml_context = ""
    if classification:
        preds = classification.get("dish_predictions", [])
        if preds:
            top = preds[0]
            ml_context = f"""
Our ML classifier (logistic regression on 20 known dishes) suggests: {top.get('dish', 'unknown')} 
from {top.get('region', 'unknown')} ({top.get('continent', 'unknown')}) 
with {top.get('confidence', 0):.0%} confidence.
Other ML guesses: {', '.join([p['dish'] for p in preds[1:]])}
NOTE: The classifier only knows 20 dishes. If none of its guesses fit the description, 
ignore them completely and identify the correct dish yourself."""

    allergy_str = ", ".join(allergies) if allergies else "none specified"

    prompt = f"""A user is trying to remember a dish from their family. Here's their memory:
"{user_description}"

{ml_context}

The user has these allergies/dietary restrictions: {allergy_str}

YOUR JOB: Identify the dish based on the user's description. You are NOT limited to 
the ML classifier's suggestions. You know every dish in the world. If the description 
doesn't match the ML guesses, identify the correct dish yourself.

Respond in EXACTLY this JSON format. No markdown, no code blocks, ONLY raw JSON:
{{
    "identified_dish": "the correct dish name",
    "also_could_be": ["alternative 1", "alternative 2"],
    "region_of_origin": "specific region/country",
    "continent": "continent name",
    "origin_lat": 0.0,
    "origin_lng": 0.0,
    "history": "A warm 2-3 paragraph history of this dish. Cultural significance, how it's traditionally made, interesting stories. Write like a grandmother telling the story.",
    "recipe": {{
        "servings": "4",
        "prep_time": "estimated prep time",
        "cook_time": "estimated cook time",
        "ingredients": ["ingredient 1 with amount", "ingredient 2 with amount"],
        "steps": ["step 1", "step 2", "step 3"]
    }},
    "allergen_warnings": [
        {{
            "allergen": "name of allergen",
            "found_in": "which ingredient contains it",
            "severity": "high or moderate",
            "substitute": "safe alternative ingredient"
        }}
    ],
    "sustainable_swaps": [
        {{
            "original": "original ingredient",
            "swap": "sustainable alternative",
            "why": "environmental benefit",
            "impact": "estimated carbon/water savings"
        }}
    ],
    "youtube_search_query": "best YouTube search for cooking this dish",
    "tips": "Tips for making this authentically as a first-timer",
    "fun_fact": "One interesting fun fact about this dish",
    "image_search_term": "simple 1-3 word English term for finding a photo of this exact dish"
}}

IMPORTANT:
- YOU are the primary identifier. The ML classifier is just a hint.
- If the user describes a dish the ML doesn't know, identify it yourself.
- If the user has allergies, flag EVERY dangerous ingredient
- Always include at least 2-3 sustainable_swaps
- Be warm and encouraging"""

    try:
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()

        # Clean markdown formatting
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        dish_info = json.loads(text)

        # Add image URL
        image_term = dish_info.get("image_search_term", dish_info.get("identified_dish", "food"))
        dish_info["image_url"] = get_dish_image(image_term)

        return dish_info

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        raw = response.text if response else "None"
        print(f"Raw response (first 500 chars): {raw[:500]}")
        return {
            "identified_dish": "Unknown",
            "history": raw,
            "recipe": {"ingredients": [], "steps": []},
            "allergen_warnings": [],
            "sustainable_swaps": [],
            "youtube_search_query": "",
            "tips": "",
            "image_url": get_dish_image("food")
        }
    except Exception as e:
        print(f"Gemini error: {e}")
        return {
            "identified_dish": "Error",
            "error": str(e),
            "history": f"Error: {str(e)}",
            "recipe": {"ingredients": [], "steps": []},
            "allergen_warnings": [],
            "sustainable_swaps": [],
            "youtube_search_query": "",
            "tips": "",
            "image_url": ""
        }

# ============================================================
# API ROUTES
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

    # Step 1: C++ classifier (secondary ML layer)
    classification = classify_dish(description)

    # Step 2: Gemini as PRIMARY identifier
    dish_info = get_dish_info(description, classification, allergies)

    # Step 3: Save to MongoDB
    mongo_id = save_recipe({
        "user_description": description,
        "allergies": allergies,
        "classification": classification,
        "dish_info": dish_info
    })

    return jsonify({
        "classification": classification,
        "dish_info": dish_info,
        "saved": mongo_id is not None
    })

@app.route("/api/recent")
def recent():
    recipes = get_recent_recipes()
    return jsonify({"recipes": recipes})

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "gemini": gemini_model is not None,
        "mongodb": mongo_db is not None,
        "classifier": os.path.exists(CPP_BINARY)
    })

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🍽️  Recipe Memory — Innovate Track")
    print("   QWER Hacks 2026")
    print("="*50 + "\n")

    init_gemini()
    init_mongo()

    if not os.path.exists(CPP_BINARY):
        print(f"\n⚠️  C++ classifier not compiled!")
        print(f"   Run: cd cpp && g++ -o classifier classifier.cpp")
        print(f"   App works without it but won't have ML classification.\n")
    else:
        print("✅ C++ classifier found")

    print(f"\n🚀 Starting server at http://localhost:5000\n")
    app.run(debug=True, port=5000, host="0.0.0.0")
