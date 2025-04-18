from flask import Flask, request, jsonify
import os,json,requests
import joblib,logging
from sentence_transformers import SentenceTransformer
import numpy as np
from pymongo import MongoClient
import certifi
from pinecone import Pinecone
from dotenv import load_dotenv
from bson import ObjectId
import google.generativeai as genai


# --- Configuration & Initialization ---
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Custom JSON encoder for MongoDB ObjectId
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return super().default(o)

app = Flask(__name__)
app.json_encoder = JSONEncoder

# Load models
print("Loading models...")
knn = joblib.load("./retail_rag/models/knn_model.joblib")
model = SentenceTransformer("./retail_rag/models/sentence_transformer")
pc = Pinecone(api_key=os.getenv("Pinecone_API_KEY"))
pinecone_index = pc.Index("product-search")
logging.info("Pinecone index connected.")

# MongoDB
mongo_client = MongoClient(os.getenv("ATLAS_URI"), tlsCAFile=certifi.where())
db = mongo_client["ecommerce_db"]
# Test MongoDB connection
db.command('ping')
logging.info("MongoDB connected.")

# Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
logging.info("Gemini AI model initialized.")
print("Models loaded successfully!")

def preprocess_text(text):
    """Basic text preprocessing"""
    if isinstance(text, str):
        return text.lower().strip()
    return ""

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "usage": {
            "endpoint": "/predict",
            "method": "POST",
            "body": {"query": "your product description here"}
        }
    })

# Add health check endpoint for Render
@app.route('/healthz', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

def predict(query):
    try:          
        # Process and predict
        processed_query = preprocess_text(query)
        query_embedding = model.encode([processed_query])
        
        # Get predictions and distances
        distances, indices = knn.kneighbors(query_embedding)
        distances = distances[0]
        
        # Get predicted category
        category = knn.predict(query_embedding)[0]
        
        # Calculate confidence
        max_distance = 2.0
        confidence = 1 - (np.mean(distances) / max_distance)
        
        return category,query_embedding
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500



def generate_response(query, products, category):
    """Generate response using Gemini API."""
    if not gemini_model:
        logging.error("Gemini model not initialized.")
        return f"I found {len(products)} {category} products. (AI response unavailable)"

    try:
        product_details = "\n".join(
            f"- {p.get('title', 'N/A')} (Brand: {p.get('brand', 'N/A')}, Price: ₹{p.get('price', 'N/A')},Description: {p.get('description', 'N/A')})"
            for p in products[:5]
        )

        prompt = f"""
        You are a friendly and knowledgeable e-commerce assistant. Your goal is to help users find the best products based on their search.

        Context:
        - User's Search Query: "{query}"
        - Predicted Product Category: "{category}"
        - Top {len(products[:5])} Relevant Products Found:
        {product_details}

        Task:
        Generate a concise (2-3 sentences), engaging, and helpful response that:
        1. Acknowledges the user's search query ("{query}").
        2. Confidently presents the found "{category}" products as highly relevant options.
        3. Briefly summarizes key insights (e.g., prominent brands, price range).
        4. Subtly encourages the user by suggesting these are well-suited matches.
        5. Take some content from product description to explain the likelyhood to given query.
        """

        response = gemini_model.generate_content(prompt)
        if hasattr(response, 'text'):
             logging.info(f"Generated Gemini response for '{query}'.")
             return response.text
        else:
             logging.warning(f"Gemini response format unexpected for '{query}'. Response: {response}")
             return f"I found {len(products)} {category} products matching '{query}'. Would you like details?"

    except Exception as e:
        logging.error(f"Gemini Error generating response: {e}", exc_info=True)
        return f"I found {len(products)} {category} products. (AI response generation failed)"

def search_products_logic(query):
    """Core logic for searching products."""
    if model is None or pinecone_index is None or db is None:
        raise ConnectionError("One or more services (Model, Pinecone, MongoDB) not initialized.")

    # Step 1: Get category from the API
    category,query_embedding = predict(query)
    if category.lower() == "other":
        return {
            "status": "not_found",
            "message": "No products available for this query (Category: Other)."
        }

    # # Step 2: Convert query to embedding
    query_embedding = query_embedding.tolist()

    # Step 3: Retrieve products from Pinecone
    pinecone_results = pinecone_index.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True
    )
    matches = pinecone_results.get("matches", [])
    if not matches:
        return {
            "status": "not_found",
            "message": "No similar products found in vector search."
        }

    # Step 4: Post-filter products based on the predicted category
    matched_ids = [match["id"] for match in matches]
    all_products = list(db.products.find({"unique_id": {"$in": matched_ids}}))
    category_products = [
        product for product in all_products
        if product.get("category", "").lower() == category.lower()
    ]

    # Step 5: Handle no results after filtering
    if not category_products:
        return {
            "status": "not_found",
            "message": f"No products found specifically in the '{category}' category after filtering."
        }

    # Step 6: Generate a natural language response
    generated_response = generate_response(query, category_products, category)

    serializable_products = []
    for product in category_products:
        if '_id' in product and isinstance(product['_id'], ObjectId):
            product['_id'] = str(product['_id']) # Convert ObjectId to string
        serializable_products.append(product)
    # Step 7: Return successful results
    return {
        "status": "success",
        "category": category,
        "generated_response": generated_response,
        "results": serializable_products
    }

# --- API Routes ---
@app.route('/search', methods=['POST'])
def search():
    """API endpoint to search for products."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    query = data.get('query', '').strip()

    if not query:
        return jsonify({"error": "Query parameter is missing or empty"}), 400

    logging.info(f"Received search request for query: '{query}'")

    try:
        result = search_products_logic(query)
        # Use Flask's jsonify which respects the app.json_encoder
        return jsonify(result)

    except ConnectionError as e:
         logging.error(f"Service Connection Error during search: {e}")
         return jsonify({"status": "error", "error": f"Service connection failed: {e}"}), 503 # Service Unavailable
    except Exception as e:
        logging.error(f"Unexpected error during search for '{query}': {e}", exc_info=True)
        return jsonify({"status": "error", "error": "An internal server error occurred"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Match Render's port (10000)
    app.run(host='0.0.0.0', port=port)