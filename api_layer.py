from flask import Flask, request, jsonify
import numpy as np
from leech_db import LeechDB
from semantic_router import SemanticRouter
import os

app = Flask(__name__)

# Initialize the Empire Index and Router
# Using the million-vector DB by default once ready
DB_PATH = "leech_empire_million.db"
if not os.path.exists(DB_PATH):
    DB_PATH = "leech_empire_100k.db" # Fallback to existing 100k DB

db = LeechDB(DB_PATH)
router = SemanticRouter(DB_PATH)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "active", "db": DB_PATH, "lattice": "Leech (24D)"})

@app.route('/index', methods=['POST'])
def index_vector():
    """
    Indexes a single vector or batch.
    Input format: {"label": "item_1", "vector": [0.1, 0.2, ...]}
    """
    data = request.json
    label = data.get('label')
    vector = np.array(data.get('vector'))
    
    if vector.shape[0] != 24:
        return jsonify({"error": "Vector must be 24-dimensional"}), 400
        
    db.index_batch([label], vector.reshape(1, -1))
    return jsonify({"status": "indexed", "label": label})

@app.route('/search', methods=['POST'])
def search():
    """
    Performs exact and neighborhood search.
    Input format: {"vector": [...], "fuzzy": true}
    """
    data = request.json
    vector = np.array(data.get('vector'))
    fuzzy = data.get('fuzzy', False)
    
    if vector.shape[0] != 24:
        return jsonify({"error": "Vector must be 24-dimensional"}), 400
    
    if fuzzy:
        results = db.query_neighborhood(vector)
    else:
        results = db.query_exact(vector)
        
    return jsonify({"query_vector": vector.tolist()[:3], "results": results})

@app.route('/route', methods=['POST'])
def route_query():
    """
    Routes a query to a specialized expert based on lattice position.
    """
    data = request.json
    vector = np.array(data.get('vector'))
    
    expert, mode = router.route(vector)
    return jsonify({"expert": expert, "routing_mode": mode})

if __name__ == "__main__":
    print(f"Empire API Layer starting on port 5000... Connected to {DB_PATH}")
    app.run(host='0.0.0.0', port=5000)
