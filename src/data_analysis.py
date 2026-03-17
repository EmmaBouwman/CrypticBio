import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import duckdb
import os
import matplotlib.pyplot as plt
import time


TABLE_NAME = "crypticbio"

load_dotenv()
base_folder = Path(os.getenv('DATA_FOLDER'))
db_path = base_folder / os.getenv('DATABASE')
con = duckdb.connect(db_path, read_only=True)


def get_cryptic_group(table, connection, start_specie, all_species, max_depth=20):
    """
    Finds the full cryptic group of a species using a recursive SQL query.
    Includes runtime tracking for debugging and performance monitoring.
    """

    query_start = time.time()

    print("--------------------------------------------------")
    print(f"Starting cryptic group search")
    print(f"Start species: {start_specie}")
    print(f"Remaining candidate species: {len(all_species)}")

    result = connection.execute(
        f"""
        WITH RECURSIVE cryptic_network AS (
            -- Anchor: starting species
            SELECT 
                scientificName,
                scientificName AS member,
                0 AS distance
            FROM {table}
            WHERE scientificName = ?

            UNION

            -- Recursive step
            SELECT 
                cn.scientificName,
                unnested.m AS member,
                cn.distance + 1
            FROM cryptic_network cn
            JOIN (
                SELECT scientificName, unnest(crypticGroup) AS m
                FROM {table}
            ) AS unnested
            ON cn.member = unnested.scientificName
            WHERE cn.distance < ?
        )

        SELECT DISTINCT member
        FROM cryptic_network
        """,
        [start_specie, max_depth]
    ).fetchall()

    query_end = time.time()
    query_time = query_end - query_start

    group = [x[0] for x in result if x[0] in all_species]

    print(f"Group size found: {len(group)}")
    print(f"Query runtime: {query_time:.2f} seconds")
    print("--------------------------------------------------")

    return group


def get_all_unique_species(table, connection):
    """
    Returns a list with all unique scientific names in the dataset.
    Args:
        -
    Returns:
        list(str)
    """
    all_species = connection.execute(
        f"""
        SELECT DISTINCT scientificName
        FROM {table}
        """
    ).fetchall()
    all_species_list = [x[0] for x in all_species]

    return all_species_list


def get_all_cryptic_groups(table, connection):
    species = get_all_unique_species(table, connection)

    total_species = len(species)
    used_species = set()
    groups = []

    start_time = time.time()

    while species:
        specie = species[0]
        group = get_cryptic_group(table, connection, specie, species)
        groups.append(group)

        for s in group:
            used_species.add(s)
            if s in species:
                species.remove(s)
        
                processed = len(used_species)
        remaining = len(species)

        elapsed = time.time() - start_time
        avg_time_per_species = elapsed / processed if processed > 0 else 0
        est_remaining = avg_time_per_species * remaining

        print(f"Processed: {processed}/{total_species} ({processed/total_species:.2%})")
        print(f"Current group size: {len(group)}")
        print(f"Remaining species: {remaining}")
        print(f"Elapsed time: {elapsed/60:.2f} minutes")
        print(f"Estimated remaining time: {est_remaining/60:.2f} minutes")
        print("----")
    return groups


def count_occurrences(table, connection, cryptic_groups):
    table_data = []
    for cr_grp in cryptic_groups:
        placeholders = ",".join(["?"] * len(cr_grp))
        result = connection.execute(
            f"""
            SELECT scientificName, COUNT(*) as occurrences
            FROM {table}
            WHERE scientificName IN ({placeholders})
            GROUP BY scientificName
            """,
            cr_grp
        ).fetchall()

        total_occurrences = sum(c for _, c in result)
        nr_distinct_species = len(cr_grp)
        species_names = ', '.join([name for name, _ in result])
        table_data.append([species_names, total_occurrences, nr_distinct_species])

    df = pd.DataFrame(table_data, columns=['Group', 'Total occurrences in dataset', 'Nr distinct species'])

    # Sort by total occurrences
    df = df.sort_values(by='Total occurrences in dataset', ascending=False).reset_index(drop=True)
    return df


def plot_occurrences(data, filename):
    plt.figure(figsize=(10,6))
    plt.bar(data['Group'], data['Total occurrences in dataset'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Total Occurrences in Dataset')
    plt.title('Occurrences per Cryptic Group')
    plt.tight_layout()
    plt.savefig(f"src/data_analysis_results/{filename}", dpi=300)


if __name__ == "__main__":
    table_length = con.execute(f"""
    SELECT COUNT(*) FROM {TABLE_NAME}
    """).fetchall()
    print(table_length)

    all_cryptic_groups = get_all_cryptic_groups(TABLE_NAME, con)
    occurrences_df = count_occurrences(TABLE_NAME, con, all_cryptic_groups)
    occurrences_df.to_csv("data_analysis_results/cryptic_groups_occurrences.csv", index=False)
    plot_occurrences(occurrences_df, "cryptic_groups_plot.png")

