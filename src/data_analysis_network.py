import duckdb
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import os
import matplotlib
# Use 'Agg' for server environments to avoid display errors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np

# Importing your Manager
from src.data_gather import DuckDBManager

def main():
    load_dotenv()
    base_folder = Path(os.getenv('DATA_FOLDER'))
    db_path = base_folder / os.getenv('DATABASE')

    # 1. Get species row counts
    print("Fetching species occurrence counts...")
    with DuckDBManager(db_path) as db:
        occ_df = db.con.execute("""
            SELECT scientificName, COUNT(*) as occurrences
            FROM crypticbio
            GROUP BY scientificName
        """).df()

    # Map: Species Name -> Row Count
    occ_dict = dict(zip(occ_df['scientificName'], occ_df['occurrences']))

    # 2. Build the full connectivity graph
    print("Building full species network...")
    query = """
    SELECT DISTINCT
        scientificName AS source_species,
        UNNEST(crypticGroup) AS target_species
    FROM crypticbio
    WHERE crypticGroup IS NOT NULL
    """
    with DuckDBManager(db_path) as db:
        df = db.con.execute(query).df()

    # Clean data (no self-loops, no duplicates)
    df = df[df['source_species'] != df['target_species']]
    df = df.drop_duplicates()
    
    G_full = nx.from_pandas_edgelist(df, 'source_species', 'target_species')

    # 3. Filter Clusters by TOTAL Row Count and Assign Colors
    print("Analyzing sub-clusters for total row count of ~30,000...")
    
    # Target range for the ENTIRE cluster sum
    TARGET_TOTAL = 60000
    TOLERANCE = 40000 # Finds clusters between 25k and 35k total rows
    
    matching_nodes = []
    node_color_map = {} # Map node to cluster ID
    cluster_count = 0
    
    # Iterate through every independent "island" in the graph
    for component in nx.connected_components(G_full):
        # Calculate the sum of rows for all species in this specific cluster
        cluster_total_rows = sum(occ_dict.get(species, 0) for species in component)
        
        if (TARGET_TOTAL - TOLERANCE) <= cluster_total_rows <= (TARGET_TOTAL + TOLERANCE):
            print(f"Found cluster {cluster_count}: Nodes={len(component)}, Total Rows={cluster_total_rows}")
            matching_nodes.extend(list(component))
            for node in component:
                node_color_map[node] = cluster_count
            cluster_count += 1

    if not matching_nodes:
        print("No sub-clusters found meeting the 30,000 total row criteria.")
        return

    # Create a subgraph of only the clusters that met the criteria
    G_final = G_full.subgraph(matching_nodes).copy()

    # 4. Visualization with Color and Layout Enhancements
    print(f"Drawing filtered network with {G_final.number_of_nodes()} total nodes in {cluster_count} clusters...")
    plt.figure(figsize=(12, 12))
    
    # Use spring_layout with careful parameter tuning for clustered graphs
    # Increased optimal distance (k) to help push distinct clusters apart
    pos = nx.spring_layout(G_final, k=1.0/np.sqrt(G_final.number_of_nodes()), iterations=50, seed=42)

    # Node sizes based on individual species counts (scaled for visibility)
    individual_counts = [occ_dict.get(n, 1) for n in G_final.nodes()]
    node_draw_sizes = [50 + (v / max(individual_counts) * 1000) for v in individual_counts]

    # Generate colors based on cluster IDs
    # Using 'tab20' or 'tab10' for discrete, visually distinct colors
    colormap = cm.get_cmap('tab20', cluster_count) # tab20 supports up to 20 distinct clusters well
    node_colors = [colormap(node_color_map[n]) for n in G_final.nodes()]

    nx.draw_networkx_nodes(
        G_final, 
        pos, 
        node_size=node_draw_sizes, 
        node_color=node_colors, 
        alpha=0.9,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Standard edges (internal and potentially between clusters if any)
    nx.draw_networkx_edges(G_final, pos, alpha=0.3, edge_color='gray')
    
    # Add labels so you know which species are in these 30k clusters
    # nx.draw_networkx_labels(G_final, pos, font_size=8, font_weight='bold')

    plt.title(f"Distinct Sub-clusters with ~{TARGET_TOTAL} Combined Rows", fontsize=14)
    plt.axis("off")
    
    output_dir = Path("data_analysis_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / "distinct_clusters_30k_total.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Process complete. Image saved to: {save_path}")

if __name__ == "_main_":
    main()