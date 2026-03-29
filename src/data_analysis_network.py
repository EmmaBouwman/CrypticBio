import duckdb
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
from src.data_gather import DuckDBManager


TARGET_TOTAL = 60000  # Target range for the ENTIRE cluster sum
TOLERANCE = 40000 # Finds clusters between 25k and 35k total rows

load_dotenv()
base_folder = Path(os.getenv('DATA_FOLDER'))
db_path = base_folder / os.getenv('DATABASE')


def count_occurrences():
    print("Fetching species occurrence counts...")
    with DuckDBManager(db_path) as db:
        occ_df = db.con.execute("""
            SELECT scientificName, COUNT(*) as occurrences
            FROM crypticbio
            GROUP BY scientificName
        """).df()

    occ_dict = dict(zip(occ_df['scientificName'], occ_df['occurrences']))
    return occ_dict


def build_network():
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
    return G_full

def filter_network_clusters(G_full, occ_dict):
    matching_nodes = []
    node_color_map = {}
    cluster_count = 0
    
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
    G_filtered = G_full.subgraph(matching_nodes).copy()
    return G_filtered, cluster_count, node_color_map


def get_cluster_data(cluster_count, color_map, occ_dict):
    cluster_summaries = []
    node_records = []

    for cluster_id in range(cluster_count):
        # Get nodes in this cluster
        cluster_nodes = [n for n, cid in color_map.items() if cid == cluster_id]
        
        # Species count
        num_species = len(cluster_nodes)
        
        # Total occurrences
        total_occurrences = sum(occ_dict.get(n, 0) for n in cluster_nodes)
        
        # Average occurrences
        avg_occurrences = total_occurrences / num_species if num_species > 0 else 0

        cluster_summaries.append({
            "cluster_id": cluster_id,
            "num_species": num_species,
            "total_occurrences": total_occurrences,
            "avg_occurrences": avg_occurrences
        })

        # Per-node records
        for node in cluster_nodes:
            node_records.append({
                "species": node,
                "cluster_id": cluster_id,
                "occurrences": occ_dict.get(node, 0)
            })

    # Convert to DataFrames
    cluster_df = pd.DataFrame(cluster_summaries)
    nodes_df = pd.DataFrame(node_records)

    # Sort for readability
    cluster_df = cluster_df.sort_values(by="total_occurrences", ascending=False)
    nodes_df = nodes_df.sort_values(by=["cluster_id", "occurrences"], ascending=[True, False])

    return cluster_df, nodes_df


def draw_boxplot(nodes_df, output_dir):
    grouped = nodes_df.groupby("cluster_id")["occurrences"].apply(list)

    # Prepare data for boxplot
    data_to_plot = grouped.values
    labels = grouped.index

    # Plot
    plt.figure(figsize=(8, 6))
    plt.boxplot(data_to_plot, labels=labels)

    plt.xlabel("Cluster ID")
    plt.ylabel("Occurrences")
    plt.title("Occurrences per Cluster")
    path = output_dir / "boxplot.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')


def save_cluster_data(nodes_df, output_dir, cluster_df, G_filtered):
    # 1. Cluster summary
    cluster_path = output_dir / "cluster_summary.csv"
    cluster_df.to_csv(cluster_path, index=False)

    # 2. Node table
    nodes_path = output_dir / "nodes_with_clusters.csv"
    nodes_df.to_csv(nodes_path, index=False)

    # 3. Edge list (filtered graph only)
    edges_df = nx.to_pandas_edgelist(G_filtered)
    edges_path = output_dir / "edges.csv"
    edges_df.to_csv(edges_path, index=False)

    print(f"Saved cluster summary to: {cluster_path}")
    print(f"Saved node table to: {nodes_path}")
    print(f"Saved edge list to: {edges_path}")

    print(cluster_df.head())
    return cluster_df


def draw_network(output_dir, G_filtered, node_draw_sizes, node_colors):
    plt.figure(figsize=(12, 12))

    pos = nx.spring_layout(G_filtered, k=1.0/np.sqrt(G_filtered.number_of_nodes()), iterations=50, seed=42)

    nx.draw_networkx_nodes(
        G_filtered, 
        pos, 
        node_size=node_draw_sizes, 
        node_color=node_colors, 
        alpha=0.9,
        edgecolors='black',
        linewidths=0.5
    )
    
    nx.draw_networkx_edges(G_filtered, pos, alpha=0.3, edge_color='gray')
    
    # Add labels so you know which species are in these 30k clusters
    #nx.draw_networkx_labels(G_filtered, pos, font_size=8, font_weight='bold')

    plt.title(f"Distinct Sub-clusters with ~{TARGET_TOTAL} combined rows", fontsize=14)
    plt.axis("off")
    
    save_path = output_dir / "distinct_clusters_30k_total.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Image saved to: {save_path}")


def main():

    print("Building full species network...")
    occurrences_dict = count_occurrences()  # Get species row counts
    G_full = build_network()  # Build the full connectivity graph

    # Filter clusters by TOTAL row count and assign colors
    print("Analyzing sub-clusters...")
    G_filtered, nr_clusters, nodes_color_map = filter_network_clusters(G_full, occurrences_dict)  # Build the full connectivity graph

    # Build cluster summary + node table
    print("Building cluster statistics...")
    cluster_df, nodes_df = get_cluster_data(nr_clusters, nodes_color_map, occurrences_dict)


    output_dir = Path("data_analysis_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    draw_boxplot(nodes_df, output_dir)
    save_cluster_data(nodes_df, output_dir, cluster_df, G_filtered)

    individual_counts = [occurrences_dict.get(n, 1) for n in G_filtered.nodes()]
    node_draw_sizes = [50 + (v / max(individual_counts) * 1000) for v in individual_counts]
    colormap = cm.get_cmap('tab20', nr_clusters) # tab20 supports up to 20 distinct clusters well
    node_colors = [colormap(nodes_color_map[n]) for n in G_filtered.nodes()]

    print(f"Drawing filtered network with {G_filtered.number_of_nodes()} total nodes in {nr_clusters} clusters...")
    draw_network(output_dir, G_filtered, node_draw_sizes, node_colors)


if __name__ == "__main__":
    main()