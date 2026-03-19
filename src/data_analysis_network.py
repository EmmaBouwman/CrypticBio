import duckdb
import pandas as pd
# from data_gather import db_path
from dotenv import load_dotenv
from pathlib import Path
import os


load_dotenv()
base_folder = Path(os.getenv('DATA_FOLDER'))
db_path = base_folder / os.getenv('DATABASE')
con = duckdb.connect(db_path, read_only=True)


print("Counting occurrences per species...")

occ_df = con.execute("""
    SELECT scientificName, COUNT(*) as occurrences
    FROM crypticbio
    GROUP BY scientificName
""").df()

# Convert to dictionary for fast lookup
occ_dict = dict(zip(occ_df['scientificName'], occ_df['occurrences']))
print(occ_dict)




















# # This query scans the entire table to build a comprehensive connectivity map
# query = """
# SELECT DISTINCT
#     scientificName AS source_species,
#     UNNEST(crypticGroup) AS target_species
# FROM crypticbio
# WHERE crypticGroup IS NOT NULL
# ORDER BY source_species
# """

# # Fetching the connectivity map
# # Note: For 171M rows, use this result carefully or chunk it if memory is an issue
# df = con.execute(query).df()

# # Clean up: remove self-references and duplicates
# df = df[df['source_species'] != df['target_species']]
# df = df.drop_duplicates()

# # Save the master connectivity map
# df.to_csv("master_species_connectivity_map.csv", index=False)
# print(f"Master map saved with {len(df)} unique connections!")

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