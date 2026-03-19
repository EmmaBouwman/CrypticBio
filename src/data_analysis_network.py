import duckdb
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import os
from src.data_gather import DuckDBManager
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


load_dotenv()
base_folder = Path(os.getenv('DATA_FOLDER'))
db_path = base_folder / os.getenv('DATABASE')

# Count occurrences per scientificName ##############################################

print("Counting occurrences per species...")

with DuckDBManager(db_path) as db:
    occ_df = db.con.execute("""
        SELECT scientificName, COUNT(*) as occurrences
        FROM crypticbio
        GROUP BY scientificName
    """).df()

occ_dict = dict(zip(occ_df['scientificName'], occ_df['occurrences']))

# Create connectivity map ###########################################################

print("Loading connectivity map...")
query = """
SELECT DISTINCT
    scientificName AS source_species,
    UNNEST(crypticGroup) AS target_species
FROM crypticbio
WHERE crypticGroup IS NOT NULL
ORDER BY source_species
"""
with DuckDBManager(db_path) as db:
    df = db.con.execute(query).df()

# Clean up: remove self-references and duplicates
df = df[df['source_species'] != df['target_species']]
df = df.drop_duplicates()

G = nx.from_pandas_edgelist(df, 'source_species', 'target_species')

# Assign node sizes ###############################################################3#

print("Assigning node sizes...")
node_sizes = []
for node in G.nodes():
    occurrences = occ_dict.get(node, 1)  # default small value if missing
    node_sizes.append(occurrences)

node_sizes = np.array(node_sizes)
node_sizes = 50 + (node_sizes - node_sizes.min()) / (node_sizes.max() - node_sizes.min()) * 1950

# Draw network ######################################################################

print("Drawing network...")

plt.figure(figsize=(12, 12))

pos = nx.spring_layout(G, k=0.15, iterations=20)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    alpha=0.7
)

nx.draw_networkx_edges(
    G,
    pos,
    alpha=0.3
)

plt.title("Species connectivity network (Cryptic Groups)")
plt.axis("off")
plt.show()
plt.savefig(f"src/data_analysis_results/network.png", dpi=300)












# # This query scans the entire table to build a comprehensive connectivity map


# # Fetching the connectivity map
# # Note: For 171M rows, use this result carefully or chunk it if memory is an issue


# import pandas as pd
# import networkx as nx
# from networkx.algorithms.community import asyn_lpa_communities, greedy

# # 1. Load your connectivity map
# # If your CSV is large, this is efficient
# print("Loading data...")
# df = pd.read_csv("./master_species_connectivity_map.csv")
# G = nx.from_pandas_edgelist(df, 'source_species', 'target_species')

# # 2. Find Communities (Clusters)
# # This uses the Clauset-Newman-Moore greedy modularity maximization
# print("Finding clusters... this might take a moment.")
# communities = list(asyn_lpa_communities(G))

# # 3. View the results
# print(f"Found {len(communities)} distinct cryptic clusters.")

# # 4. Map each species to a cluster ID
# cluster_map = {}
# for i, comm in enumerate(communities):
#     for species in comm:
#         cluster_map[species] = i

# # Create a DataFrame to save the cluster labels
# cluster_df = pd.DataFrame.from_dict(cluster_map, orient='index', columns=['cluster_id'])
# cluster_df.to_csv("species_clusters.csv")
# print("Cluster map saved to 'species_clusters.csv'")

# # 5. Look at a specific cluster (e.g., the first one)
# print("\n--- Example Species in Cluster 0 ---")
# print(list(communities[0])[:10]) 