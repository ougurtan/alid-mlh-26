/*
 * Recipe Memory - Dish & Geography Classifier
 * Logistic Regression with Gradient Descent in C++
 * 
 * Two classifiers:
 * 1. Dish classifier - predicts which dish from description
 * 2. Geography classifier - predicts continent of origin
 * 
 * Usage: ./classifier "flaky pastry cheese spinach turkey"
 * Output: JSON with predicted dish, continent, and confidence scores
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <cctype>

using namespace std;

// ============================================================
// DATA STRUCTURES
// ============================================================

struct DishClass {
    string name;
    string region;       // specific region
    string continent;    // broad geography
    vector<string> keywords;
    vector<double> weights;
    double bias = 0.0;
};

struct ContinentClass {
    string name;
    vector<double> weights;
    double bias = 0.0;
};

// ============================================================
// CORE ML FUNCTIONS
// ============================================================

// Sigmoid function: squashes any number into range (0, 1)
double sigmoid(double z) {
    // Clamp to avoid overflow
    if (z > 500) return 1.0;
    if (z < -500) return 0.0;
    return 1.0 / (1.0 + exp(-z));
}

// Convert text to lowercase and split into words
vector<string> tokenize(const string& text) {
    vector<string> tokens;
    string word;
    for (char c : text) {
        if (isalpha(c)) {
            word += tolower(c);
        } else if (!word.empty()) {
            tokens.push_back(word);
            word.clear();
        }
    }
    if (!word.empty()) tokens.push_back(word);
    return tokens;
}

// Build vocabulary from all dish keywords
vector<string> buildVocabulary(const vector<DishClass>& dishes) {
    map<string, bool> seen;
    vector<string> vocab;
    for (const auto& dish : dishes) {
        for (const auto& kw : dish.keywords) {
            if (!seen[kw]) {
                seen[kw] = true;
                vocab.push_back(kw);
            }
        }
    }
    return vocab;
}

// Bag of words: text → feature vector
vector<double> extractFeatures(const string& text, const vector<string>& vocab) {
    vector<string> tokens = tokenize(text);
    map<string, bool> tokenSet;
    for (const auto& t : tokens) tokenSet[t] = true;

    vector<double> features(vocab.size(), 0.0);
    for (size_t i = 0; i < vocab.size(); i++) {
        if (tokenSet.count(vocab[i])) {
            features[i] = 1.0;
        }
    }
    return features;
}

// Predict probability
double predict(const vector<double>& features, const vector<double>& weights, double bias) {
    double z = bias;
    for (size_t i = 0; i < features.size(); i++) {
        z += features[i] * weights[i];
    }
    return sigmoid(z);
}

// Train one binary classifier using gradient descent
void trainBinaryClassifier(
    vector<double>& weights,
    double& bias,
    const vector<vector<double>>& allFeatures,
    const vector<int>& labels,
    int vocabSize,
    double learningRate = 0.1,
    int epochs = 300
) {
    int n = allFeatures.size();
    weights.assign(vocabSize, 0.0);
    bias = 0.0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double dBias = 0.0;
        vector<double> dWeights(vocabSize, 0.0);

        for (int i = 0; i < n; i++) {
            double pred = predict(allFeatures[i], weights, bias);
            double error = pred - labels[i];
            dBias += error;
            for (int j = 0; j < vocabSize; j++) {
                dWeights[j] += error * allFeatures[i][j];
            }
        }

        bias -= learningRate * (dBias / n);
        for (int j = 0; j < vocabSize; j++) {
            weights[j] -= learningRate * (dWeights[j] / n);
        }
    }
}

// ============================================================
// TRAINING DATA - 20 dishes across 6 continents
// ============================================================

vector<DishClass> initDishes() {
    vector<DishClass> dishes;

    // === ASIA ===
    dishes.push_back({"Börek", "Turkish / Balkan", "Asia",
        {"flaky", "pastry", "cheese", "spinach", "feta", "phyllo", "filo",
         "dough", "layers", "baked", "savory", "turkish", "bosnian", "balkan",
         "crispy", "buttery", "rolled", "stuffed", "meat", "potato"}, {}});

    dishes.push_back({"Biryani", "South Asian", "Asia",
        {"rice", "spiced", "layered", "saffron", "chicken", "lamb", "yogurt",
         "indian", "pakistani", "fragrant", "basmati", "cardamom", "cinnamon",
         "onion", "fried", "mint", "cilantro", "raita", "turmeric"}, {}});

    dishes.push_back({"Dumplings (Jiaozi)", "Chinese", "Asia",
        {"dough", "wrapped", "pork", "cabbage", "steamed", "fried", "boiled",
         "chinese", "lunar", "new", "year", "pleated", "minced", "ginger",
         "soy", "sauce", "dumpling", "filling", "wrapper", "scallion"}, {}});

    dishes.push_back({"Sushi", "Japanese", "Asia",
        {"rice", "fish", "raw", "seaweed", "nori", "rolled", "vinegar",
         "wasabi", "soy", "salmon", "tuna", "japanese", "fresh", "avocado",
         "ginger", "maki", "nigiri", "sticky"}, {}});

    dishes.push_back({"Pho", "Vietnamese", "Asia",
        {"soup", "noodle", "broth", "beef", "herbs", "basil", "bean",
         "sprouts", "vietnamese", "clear", "hot", "lime", "hoisin",
         "sriracha", "noodles", "star", "anise", "aromatic"}, {}});

    dishes.push_back({"Kimchi Jjigae", "Korean", "Asia",
        {"stew", "kimchi", "spicy", "tofu", "pork", "korean", "fermented",
         "cabbage", "hot", "red", "pepper", "garlic", "bubbling",
         "pot", "comfort", "sesame", "gochujang"}, {}});

    dishes.push_back({"Pad Thai", "Thai", "Asia",
        {"noodle", "stir", "fried", "peanut", "thai", "shrimp", "tofu",
         "bean", "sprouts", "lime", "tamarind", "egg", "scallion",
         "fish", "sauce", "wok", "sweet", "tangy"}, {}});

    dishes.push_back({"Baklava", "Middle Eastern / Turkish", "Asia",
        {"sweet", "pastry", "layers", "phyllo", "nuts", "honey", "syrup",
         "pistachio", "walnut", "butter", "crispy", "flaky", "dessert",
         "turkish", "greek", "lebanese", "sticky", "cinnamon"}, {}});

    // === EUROPE ===
    dishes.push_back({"Pierogi", "Polish / Eastern European", "Europe",
        {"dough", "filled", "potato", "cheese", "boiled", "fried", "onion",
         "sour", "cream", "polish", "dumpling", "savory", "sweet", "berry",
         "sauerkraut", "mushroom", "butter", "eastern", "european"}, {}});

    dishes.push_back({"Paella", "Spanish", "Europe",
        {"rice", "saffron", "seafood", "chicken", "spanish", "pan",
         "shrimp", "mussel", "tomato", "olive", "oil", "garlic",
         "paprika", "peas", "yellow", "crispy", "bottom", "valencia"}, {}});

    dishes.push_back({"Pasta Carbonara", "Italian", "Europe",
        {"pasta", "egg", "pecorino", "guanciale", "black", "pepper",
         "italian", "creamy", "roman", "spaghetti", "parmesan",
         "bacon", "pancetta", "yolk", "al", "dente"}, {}});

    // === NORTH AMERICA ===
    dishes.push_back({"Tamales", "Mexican", "North America",
        {"corn", "masa", "husk", "steamed", "wrapped", "chicken", "pork",
         "chili", "dough", "filling", "mexican", "christmas", "holiday",
         "banana", "leaf", "spicy", "salsa", "lard"}, {}});

    dishes.push_back({"Pupusas", "Salvadoran", "North America",
        {"corn", "stuffed", "cheese", "bean", "pork", "griddle", "thick",
         "flat", "salvadoran", "curtido", "cabbage", "slaw", "masa",
         "filled", "crispy", "handmade", "central", "american"}, {}});

    dishes.push_back({"Gumbo", "Cajun / Southern US", "North America",
        {"stew", "okra", "roux", "shrimp", "sausage", "andouille", "cajun",
         "creole", "louisiana", "rice", "holy", "trinity", "celery",
         "bell", "pepper", "onion", "dark", "thick"}, {}});

    // === SOUTH AMERICA ===
    dishes.push_back({"Empanadas", "Argentine", "South America",
        {"pastry", "filled", "meat", "beef", "baked", "fried", "dough",
         "onion", "egg", "olive", "argentinian", "chilean", "spanish",
         "crispy", "handheld", "turnover", "savory"}, {}});

    dishes.push_back({"Ceviche", "Peruvian", "South America",
        {"raw", "fish", "lime", "citrus", "onion", "cilantro", "chili",
         "peruvian", "fresh", "cold", "marinated", "shrimp", "corn",
         "sweet", "potato", "aji", "amarillo", "cured"}, {}});

    // === AFRICA ===
    dishes.push_back({"Injera with Wat", "Ethiopian", "Africa",
        {"flatbread", "spongy", "sour", "fermented", "stew", "lentil",
         "berbere", "spicy", "ethiopian", "torn", "hands", "communal",
         "teff", "bread", "chicken", "egg"}, {}});

    dishes.push_back({"Jollof Rice", "West African", "Africa",
        {"rice", "tomato", "spiced", "pepper", "onion", "nigerian",
         "ghanaian", "party", "smoky", "red", "chicken", "oil",
         "bay", "leaf", "thyme", "scotch", "bonnet"}, {}});

    dishes.push_back({"Couscous", "North African", "Africa",
        {"grain", "steamed", "semolina", "vegetables", "lamb", "chicken",
         "moroccan", "tunisian", "algerian", "harissa", "chickpea",
         "carrot", "zucchini", "raisin", "spiced", "stew"}, {}});

    // === OCEANIA ===
    dishes.push_back({"Meat Pie", "Australian", "Oceania",
        {"pie", "meat", "beef", "gravy", "pastry", "baked", "australian",
         "ketchup", "sauce", "flaky", "crust", "savory", "handheld",
         "golden", "onion", "hot", "comfort"}, {}});

    return dishes;
}

// ============================================================
// TRAINING
// ============================================================

void generateTrainingData(
    const vector<DishClass>& dishes,
    const vector<string>& vocab,
    vector<vector<double>>& allFeatures,
    vector<vector<int>>& dishLabels,
    vector<vector<int>>& continentLabels,
    vector<string>& continentNames
) {
    int numDishes = dishes.size();

    // Build continent list
    map<string, int> continentMap;
    for (const auto& dish : dishes) {
        if (continentMap.find(dish.continent) == continentMap.end()) {
            int idx = continentMap.size();
            continentMap[dish.continent] = idx;
            continentNames.push_back(dish.continent);
        }
    }
    int numContinents = continentNames.size();

    dishLabels.resize(numDishes);
    continentLabels.resize(numContinents);
    allFeatures.clear();

    for (int d = 0; d < numDishes; d++) {
        // Full keyword example
        string fullDesc = "";
        for (const auto& kw : dishes[d].keywords) fullDesc += kw + " ";
        allFeatures.push_back(extractFeatures(fullDesc, vocab));

        for (int j = 0; j < numDishes; j++)
            dishLabels[j].push_back(j == d ? 1 : 0);
        for (int c = 0; c < numContinents; c++)
            continentLabels[c].push_back(continentMap[dishes[d].continent] == c ? 1 : 0);

        // Partial examples for robustness
        if (dishes[d].keywords.size() >= 4) {
            size_t mid = dishes[d].keywords.size() / 2;

            string firstHalf = "", secondHalf = "";
            for (size_t i = 0; i < mid; i++) firstHalf += dishes[d].keywords[i] + " ";
            for (size_t i = mid; i < dishes[d].keywords.size(); i++) secondHalf += dishes[d].keywords[i] + " ";

            // First half
            allFeatures.push_back(extractFeatures(firstHalf, vocab));
            for (int j = 0; j < numDishes; j++)
                dishLabels[j].push_back(j == d ? 1 : 0);
            for (int c = 0; c < numContinents; c++)
                continentLabels[c].push_back(continentMap[dishes[d].continent] == c ? 1 : 0);

            // Second half
            allFeatures.push_back(extractFeatures(secondHalf, vocab));
            for (int j = 0; j < numDishes; j++)
                dishLabels[j].push_back(j == d ? 1 : 0);
            for (int c = 0; c < numContinents; c++)
                continentLabels[c].push_back(continentMap[dishes[d].continent] == c ? 1 : 0);

            // Random third (first + last few keywords)
            string mixed = "";
            for (size_t i = 0; i < 3 && i < dishes[d].keywords.size(); i++)
                mixed += dishes[d].keywords[i] + " ";
            for (size_t i = dishes[d].keywords.size() - 2; i < dishes[d].keywords.size(); i++)
                mixed += dishes[d].keywords[i] + " ";

            allFeatures.push_back(extractFeatures(mixed, vocab));
            for (int j = 0; j < numDishes; j++)
                dishLabels[j].push_back(j == d ? 1 : 0);
            for (int c = 0; c < numContinents; c++)
                continentLabels[c].push_back(continentMap[dishes[d].continent] == c ? 1 : 0);
        }
    }
}

// ============================================================
// HELPER: escape string for JSON output
// ============================================================
string jsonEscape(const string& s) {
    string result;
    for (char c : s) {
        if (c == '"') result += "\\\"";
        else if (c == '\\') result += "\\\\";
        else if (c == '/') result += "\\/";
        else result += c;
    }
    return result;
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: ./classifier \"description of the dish you remember\"" << endl;
        return 1;
    }

    string userInput = argv[1];

    // Initialize
    vector<DishClass> dishes = initDishes();
    vector<string> vocab = buildVocabulary(dishes);
    int vocabSize = vocab.size();

    // Generate training data
    vector<vector<double>> allFeatures;
    vector<vector<int>> dishLabels, continentLabels;
    vector<string> continentNames;
    generateTrainingData(dishes, vocab, allFeatures, dishLabels, continentLabels, continentNames);

    // Train dish classifiers
    for (size_t d = 0; d < dishes.size(); d++) {
        trainBinaryClassifier(dishes[d].weights, dishes[d].bias,
                              allFeatures, dishLabels[d], vocabSize);
    }

    // Train continent classifiers
    vector<ContinentClass> continents(continentNames.size());
    for (size_t c = 0; c < continents.size(); c++) {
        continents[c].name = continentNames[c];
        trainBinaryClassifier(continents[c].weights, continents[c].bias,
                              allFeatures, continentLabels[c], vocabSize);
    }

    // Classify user input
    vector<double> inputFeatures = extractFeatures(userInput, vocab);

    // Dish predictions
    vector<pair<double, int>> dishScores;
    for (size_t d = 0; d < dishes.size(); d++) {
        double prob = predict(inputFeatures, dishes[d].weights, dishes[d].bias);
        dishScores.push_back({prob, d});
    }
    sort(dishScores.begin(), dishScores.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    // Continent predictions
    vector<pair<double, int>> contScores;
    for (size_t c = 0; c < continents.size(); c++) {
        double prob = predict(inputFeatures, continents[c].weights, continents[c].bias);
        contScores.push_back({prob, c});
    }
    sort(contScores.begin(), contScores.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    // Output JSON
    cout << "{" << endl;
    cout << "  \"input\": \"" << jsonEscape(userInput) << "\"," << endl;

    // Top 3 dish predictions
    cout << "  \"dish_predictions\": [" << endl;
    int topDishes = min(3, (int)dishScores.size());
    for (int i = 0; i < topDishes; i++) {
        int idx = dishScores[i].second;
        cout << "    {" << endl;
        cout << "      \"dish\": \"" << jsonEscape(dishes[idx].name) << "\"," << endl;
        cout << "      \"region\": \"" << jsonEscape(dishes[idx].region) << "\"," << endl;
        cout << "      \"continent\": \"" << jsonEscape(dishes[idx].continent) << "\"," << endl;
        cout << "      \"confidence\": " << dishScores[i].first << endl;
        cout << "    }";
        if (i < topDishes - 1) cout << ",";
        cout << endl;
    }
    cout << "  ]," << endl;

    // Continent predictions
    cout << "  \"continent_predictions\": [" << endl;
    int topConts = min(3, (int)contScores.size());
    for (int i = 0; i < topConts; i++) {
        int idx = contScores[i].second;
        cout << "    {" << endl;
        cout << "      \"continent\": \"" << jsonEscape(continents[idx].name) << "\"," << endl;
        cout << "      \"confidence\": " << contScores[i].first << endl;
        cout << "    }";
        if (i < topConts - 1) cout << ",";
        cout << endl;
    }
    cout << "  ]" << endl;

    cout << "}" << endl;

    return 0;
}
