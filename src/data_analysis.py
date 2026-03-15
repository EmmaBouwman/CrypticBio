import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import duckdb
import os
import matplotlib.pyplot as plt


TABLE_NAME = "crypticbio"

load_dotenv()
base_folder = Path(os.getenv('DATA_FOLDER'))
db_path = base_folder / os.getenv('DATABASE')
con = duckdb.connect(db_path, read_only=True)


def get_cryptic_group(table, connection, start_specie, all_species):
    visited = set()
    stack = [start_specie]

    while stack:
        specie = stack.pop()

        if specie in visited or specie not in all_species:
            continue

        visited.add(specie)

        result = connection.execute(
            f"""
            SELECT crypticGroup
            FROM {table}
            WHERE scientificName = ?
            LIMIT 1
            """,
            [specie]
        ).fetchone()

        if result is None:
            continue

        cryptic_group = result[0]

        if isinstance(cryptic_group, str):
            cryptic_group = ast.literal_eval(cryptic_group)

        if cryptic_group:
            for s in cryptic_group:
                if s not in visited and s in all_species:
                    stack.append(s)

    return list(visited)


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
    groups = []
    while species:
        specie = species[0]
        group = get_cryptic_group(table, connection, specie, species)
        groups.append(group)
        for s in group:
            if s in species:
                species.remove(s)
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
    all_cryptic_groups = get_all_cryptic_groups(TABLE_NAME, con)
    occurrences_df = count_occurrences(TABLE_NAME, con, all_cryptic_groups)
    occurrences_df.to_csv("data_analysis_results/cryptic_groups_occurrences.csv", index=False)
    plot_occurrences(occurrences_df, "cryptic_groups_plot.png")

