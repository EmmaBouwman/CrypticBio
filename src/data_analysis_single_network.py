import duckdb
import pandas as pd
# from data_gather import db_path
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
# con = duckdb.connect(db_path, read_only=True)

# Count occurrences per scientificName ##############################################

print("Counting occurrences per species...")


with DuckDBManager(db_path) as db:
    occ_df = db.con.execute("""
        SELECT scientificName, COUNT(*) as occurrences
        FROM crypticbio
        GROUP BY scientificName
    """).df()

# Convert to dictionary for fast lookup
occ_dict = dict(zip(occ_df['scientificName'], occ_df['occurrences']))

top_species = set(occ_df.sort_values(by='occurrences', ascending=False)
                 .head(500)['scientificName'])

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

sub_df = df[
    df['source_species'].isin(top_species) &
    df['target_species'].isin(top_species)
]

G = nx.from_pandas_edgelist(sub_df, 'source_species', 'target_species')

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