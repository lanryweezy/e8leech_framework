import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_lattice_density(db_path="leech_empire_100k.db"):
    print("--- Generating Leech Lattice Semantic Heatmap ---")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Fetch occupied centroids and their populations
    cursor.execute("SELECT centroid_id, length(labels) - length(replace(labels, ',', '')) + 1 FROM buckets")
    rows = cursor.fetchall()
    
    if not rows:
        print("No data found in DB.")
        return

    keys = [row[0] for row in rows]
    counts = [row[1] for row in rows]
    
    # 2. Convert keys to 24D arrays
    key_arrays = np.array([[int(x) for x in k.split(",")] for k in keys])
    
    # 3. Dimensionality Reduction (TSNE) for 24D -> 2D visualization
    print(f"Reducing {len(key_arrays)} high-dimensional centroids to 2D...")
    # Use a subset if too large for TSNE
    sample_size = min(2000, len(key_arrays))
    indices = np.random.choice(len(key_arrays), sample_size, replace=False)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    vis_data = tsne.fit_transform(key_arrays[indices])
    
    # 4. Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(vis_data[:, 0], vis_data[:, 1], 
                         c=np.array(counts)[indices], 
                         cmap='viridis', 
                         alpha=0.6, 
                         s=10 + 20 * (np.array(counts)[indices] / max(counts)))
    
    plt.colorbar(scatter, label='Bucket Population (Density)')
    plt.title("Leech Lattice Semantic Heatmap (24D -> 2D Projection)")
    plt.xlabel("TSNE Dimension 1")
    plt.ylabel("TSNE Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    output_path = "leech_heatmap.png"
    plt.savefig(output_path)
    print(f"Heatmap saved to {output_path}")
    conn.close()

if __name__ == "__main__":
    visualize_lattice_density()
