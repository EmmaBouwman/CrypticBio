import os
from pathlib import Path

import matplotlib
import pandas as pd
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.data_gather import DuckDBManager

TARGET_TOTAL = 60000  # Target range for the ENTIRE cluster sum
TOLERANCE = 40000  # Finds clusters between 25k and 35k total rows

load_dotenv()
base_folder = Path(os.getenv("DATA_FOLDER"))
db_path = base_folder / os.getenv("DATABASE")


def count_occurrences():
    """
    Retrieve occurrence counts for each species in the database.
    Queries the `crypticbio` table and counts the number of records for each
    unique `scientificName`.
    Returns:
        dict[str, int]:
            A dictionary mapping species names to their total occurrence counts.
    """
    print("Fetching species occurrence counts...")
    with DuckDBManager(db_path) as db:
        occ_df = db.con.execute("""
            SELECT scientificName, COUNT(*) as occurrences
            FROM crypticbio
            GROUP BY scientificName
        """).df()

    occ_dict = dict(zip(occ_df["scientificName"], occ_df["occurrences"]))
    return occ_dict


def fetch_ids_with_species_and_location():
    """
    Select observation IDs, species names, and geographic coordinates
    from all records from the crypticbio table that contain valid
    latitude and longitude values.
    Returns:
        pandas.DataFrame:
            A DataFrame containing the following columns:
            - id: Unique observation identifier
            - scientificName: species name
            - decimalLatitude: latitude coordinate
            - decimalLongitude: longitude coordinate
    """
    with DuckDBManager(db_path) as db:
        df = db.con.execute("""
            SELECT id, scientificName, decimalLatitude, decimalLongitude
            FROM crypticbio
            WHERE decimalLatitude IS NOT NULL
              AND decimalLongitude IS NOT NULL
        """).df()
    return df


def build_network():
    """
    Build a species connectivity network from cryptic group relationships.
    Creates an undirected NetworkX graph where nodes represent species and
    edges represent relationships between species listed in the same
    crypticGroup.
    Returns:
        networkx.Graph: a graph containing species connectivity relationships.
    """
    query = """
    SELECT DISTINCT
        scientificName AS source_species,
        UNNEST(crypticGroup) AS target_species
    FROM crypticbio
    WHERE crypticGroup IS NOT NULL
    """
    with DuckDBManager(db_path) as db:
        df = db.con.execute(query).df()

    # Clean data (no self-loops and duplicates)
    df = df[df["source_species"] != df["target_species"]]
    df = df.drop_duplicates()

    G_full = nx.from_pandas_edgelist(df, "source_species", "target_species")
    return G_full


def filter_network_clusters(G_full, occ_dict):
    """
    Filter graph clusters based on total occurrence counts.
    Iterates through connected components in the graph and selects clusters
    whose combined occurrence counts fall within the target range.
    Args:
        G_full (networkx.Graph): the full species connectivity graph.
        occ_dict (dict[str, int]): dictionary mapping species names to occurrence counts.
    Returns: tuple[networkx.Graph, int, dict[str, int]] | None:
            Returns a tuple containing:
            - G_filtered: Subgraph containing only matching clusters.
            - cluster_count: Number of matching clusters.
            - node_color_map: Mapping of node names to cluster IDs.
            Returns None if no matching clusters are found.
    """
    matching_nodes = []
    node_color_map = {}
    cluster_count = 0

    for component in nx.connected_components(G_full):
        # Calculate the sum of rows for all species in this specific cluster
        cluster_total_rows = sum(occ_dict.get(species, 0) for species in component)

        if (TARGET_TOTAL - TOLERANCE) <= cluster_total_rows:
            if cluster_total_rows <= (TARGET_TOTAL + TOLERANCE):
                print(
                    f"Found cluster {cluster_count}: Nodes={len(component)}, "
                    + " Total rows={cluster_total_rows}"
                )
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
    """
    Create summary statistics and node data for clusters.
    Computes species counts, total occurrences, and average occurrences
    for each cluster. Also creates a table with info per species.
    Args:
        cluster_count (int): number of identified clusters.
        color_map (dict[str, int]): mapping of species names to cluster IDs.
        occ_dict (dict[str, int]): dictionary mapping species names to occurrence counts.
    Returns: tuple[pandas.DataFrame, pandas.DataFrame]:
            - cluster_df:
                Summary statistics for each cluster.
            - nodes_df:
                Per-species cluster membership and occurrence data.
    """
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

        cluster_summaries.append(
            {
                "cluster_id": cluster_id,
                "num_species": num_species,
                "total_occurrences": total_occurrences,
                "avg_occurrences": avg_occurrences,
            }
        )

        # Per-node records
        for node in cluster_nodes:
            node_records.append(
                {
                    "species": node,
                    "cluster_id": cluster_id,
                    "occurrences": occ_dict.get(node, 0),
                }
            )

    # Convert to DataFrames
    cluster_df = pd.DataFrame(cluster_summaries)
    nodes_df = pd.DataFrame(node_records)

    # Sort for readability
    cluster_df = cluster_df.sort_values(by="total_occurrences", ascending=False)
    nodes_df = nodes_df.sort_values(
        by=["cluster_id", "occurrences"], ascending=[True, False]
    )

    return cluster_df, nodes_df


def map_ids_to_clusters(df, nodes_color_map):
    """
    Assign cluster IDs to observation records. Maps each species in the 
    input dataframe to its corresponding cluster ID using the 
    node-to-cluster mapping.
    Args:
        df (pandas.DataFrame):
            DataFrame containing a `scientificName` column.
        nodes_color_map (dict[str, int]):
            Mapping of species names to cluster IDs.
    Returns:
        pandas.DataFrame: filtered DataFrame containing only 
        rows assigned to a cluster and a cluster_id column.
    """
    df["cluster_id"] = df["scientificName"].map(nodes_color_map)
    df = df.dropna(subset=["cluster_id"])
    df["cluster_id"] = df["cluster_id"].astype(int)
    return df


def pivot_ids_by_cluster(id_df):
    """
    Reshape observation IDs into a wide cluster-based format. Groups 
    observation IDs by cluster and stores them in separate columns.
    Args:
        id_df (pandas.DataFrame): dataFrame containing cluster_id 
                                and id columns.
    Returns:
        pandas.DataFrame: wide-format DataFrame where each column 
        represents a cluster and contains the corresponding observation IDs.
    """
    grouped = id_df.groupby("cluster_id")["id"].apply(list)
    max_len = max(len(lst) for lst in grouped)

    data = {}
    for cluster_id, ids in grouped.items():
        padded = ids + [pd.NA] * (max_len - len(ids))
        data[f"cluster_{cluster_id}"] = padded

    wide_df = pd.DataFrame(data)

    wide_df = wide_df.astype("Int64")

    return wide_df


def save_cluster_ids(id_wide_df, output_dir):
    """
    Save clustered observation IDs to a csv file.
    Args:
        id_wide_df (pandas.DataFrame):
            Wide-format DataFrame of clustered observation IDs.
        output_dir (pathlib.Path):
            Directory where the CSV file will be saved.
    Returns: None
    """
    path = output_dir / "cluster_ids_wide.csv"
    id_wide_df.to_csv(path, index=False)
    print(f"Saved cluster IDs to: {path}")


def draw_boxplot(nodes_df, output_dir):
    """
    Creates a boxplot showing the distribution of occurrence counts for
    species within each cluster.
    Args:
        nodes_df (pandas.DataFrame):
            DataFrame containing cluster IDs and occurrence counts.
        output_dir (pathlib.Path):
            Directory where the plot image will be saved.
    Returns: None
    """
    grouped = nodes_df.groupby("cluster_id")["occurrences"].apply(list)

    data_to_plot = grouped.values
    labels = grouped.index

    plt.figure(figsize=(8, 6))
    plt.boxplot(data_to_plot, labels=labels)

    plt.xlabel("Cluster ID")
    plt.ylabel("Occurrences")
    plt.title("Occurrences per Cluster")
    path = output_dir / "boxplot.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")


def save_cluster_data(nodes_df, output_dir, cluster_df, G_filtered):
    """
    Exports cluster summaries, node-level cluster data, and graph edge
    lists to the specified output directory.
    Args:
        nodes_df (pandas.DataFrame):
            DataFrame containing node-level cluster data.
        output_dir (pathlib.Path):
            Directory where output files will be saved.
        cluster_df (pandas.DataFrame):
            DataFrame containing cluster summary statistics.
        G_filtered (networkx.Graph):
            Filtered graph containing selected clusters.
    Returns:
        pandas.DataFrame:
            The cluster summary DataFrame.
    """
    cluster_path = output_dir / "cluster_summary.csv"
    cluster_df.to_csv(cluster_path, index=False)

    nodes_path = output_dir / "nodes_with_clusters.csv"
    nodes_df.to_csv(nodes_path, index=False)

    edges_df = nx.to_pandas_edgelist(G_filtered)
    edges_path = output_dir / "edges.csv"
    edges_df.to_csv(edges_path, index=False)

    print(f"Saved cluster summary to: {cluster_path}")
    print(f"Saved node table to: {nodes_path}")
    print(f"Saved edge list to: {edges_path}")

    print(cluster_df.head())
    return cluster_df


def draw_network(output_dir, G_filtered, node_draw_sizes, node_colors):
    """
    Draws the filtered graph with node sizes and colors
    representing species occurrence counts and to which cluster it belongs.
    Args:
        output_dir (pathlib.Path):
            directory where the network image will be saved.
        G_filtered (networkx.Graph):
            filtered graph containing selected clusters.
        node_draw_sizes (list[float]):
            sizes used for rendering graph nodes.
        node_colors (list):
            colors assigned to graph nodes.
    Returns: None
    """
    plt.figure(figsize=(12, 12))

    pos = nx.spring_layout(
        G_filtered,
        k=1.0 / np.sqrt(G_filtered.number_of_nodes()),
        iterations=50,
        seed=42,
    )

    nx.draw_networkx_nodes(
        G_filtered,
        pos,
        node_size=node_draw_sizes,
        node_color=node_colors,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.5,
    )

    nx.draw_networkx_edges(G_filtered, pos, alpha=0.3, edge_color="gray")

    # Add labels so you know which species are in these 30k clusters
    # nx.draw_networkx_labels(G_filtered, pos, font_size=8, font_weight='bold')

    plt.title(f"Distinct Sub-clusters with ~{TARGET_TOTAL} combined rows", fontsize=14)
    plt.axis("off")

    save_path = output_dir / "distinct_clusters_30k_total.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Image saved to: {save_path}")


def draw_world_map(df, output_dir, nr_clusters):
    """
    Plot the geographic distribution of clusters by making a scatter plot 
    of latitude and longitude coordinates, colored by cluster.
    Args:
        df (pandas.DataFrame):
            DataFrame containing geographic coordinates and cluster IDs.
        output_dir (pathlib.Path):
            Directory where the map image will be saved
        nr_clusters (int):
            total number of clusters
    Returns: None
    """
    plt.figure(figsize=(14, 7))

    # Use a categorical colormap
    colormap = cm.get_cmap("tab20", nr_clusters)

    for cluster_id in range(nr_clusters):
        cluster_data = df[df["cluster_id"] == cluster_id]

        plt.scatter(
            cluster_data["decimalLongitude"],
            cluster_data["decimalLatitude"],
            s=10,
            alpha=0.6,
            color=colormap(cluster_id),
            label=f"Cluster {cluster_id}",
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Global Distribution of Clusters")

    plt.legend(markerscale=2, fontsize=8)
    plt.grid(True)

    path = output_dir / "worldmap_clusters.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"World map saved to: {path}")


def main():
    """
    Execute the cluster analysis workflow.
    1. Counts species occurrences
    2. Builds the species connectivity network
    3. Filters clusters by total occurrence count
    4. Generates cluster summary statistics
    5. Saves cluster IDs and analysis outputs
    6. Creates visualizations:
       - Boxplots
       - Network graphs
       - Geographic distribution maps
    Returns: None
    """
    with DuckDBManager(db_path) as db:
        total_rows = db.con.execute("""
            SELECT COUNT(*) FROM crypticbio
        """).fetchone()[0]

        print(f"Total observations: {total_rows}")

    occurrences_dict = count_occurrences()  # Get species row counts
    G_full = build_network()  # Build the full connectivity graph

    G_filtered, nr_clusters, nodes_color_map = filter_network_clusters(
        G_full, occurrences_dict
    )

    cluster_df, nodes_df = get_cluster_data(
        nr_clusters, nodes_color_map, occurrences_dict
    )
    filtered_total = cluster_df["total_occurrences"].sum()

    print(f"Total observations in filtered clusters: {filtered_total}")

    output_dir = Path("results/data_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    id_df = fetch_ids_with_species_and_location()
    id_cluster_df = map_ids_to_clusters(id_df, nodes_color_map)
    id_wide_df = pivot_ids_by_cluster(id_cluster_df)
    save_cluster_ids(id_wide_df, output_dir)

    draw_boxplot(nodes_df, output_dir)
    save_cluster_data(nodes_df, output_dir, cluster_df, G_filtered)

    individual_counts = [occurrences_dict.get(n, 1) for n in G_filtered.nodes()]
    node_draw_sizes = [
        50 + (v / max(individual_counts) * 1000) for v in individual_counts
    ]

    colormap = cm.get_cmap("tab20", nr_clusters)
    node_colors = [colormap(nodes_color_map[n]) for n in G_filtered.nodes()]

    draw_network(output_dir, G_filtered, node_draw_sizes, node_colors)

    location_cluster_df = map_ids_to_clusters(id_df, nodes_color_map)

    draw_world_map(location_cluster_df, output_dir, nr_clusters)


if __name__ == "__main__":
    main()
