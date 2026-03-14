#from datasets import load_dataset
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import duckdb
import os

# data = {
#     "scientificName": ["A", "B", "C", "D", "E", "F"],
#     "crypticGroup": [
#         ["B", "C"],  # A connects to B, C
#         ["D"],       # B connects to D
#         [],          # C connects to nothing
#         ["A"],       # D connects back to A
#         ["F"],       # E connects to F
#         []           # F connects to nothing
#     ]
# }

# test_df = pd.DataFrame(data)


load_dotenv()

base_folder = Path(os.getenv('DATA_FOLDER'))
db_path = base_folder / os.getenv('DATABASE')
con = duckdb.connect(db_path, read_only=True)

TABLE_NAME = "crypticbio"


def get_cryptic_group(start_specie):
    visited = set()
    stack = [start_specie]

    while stack:
        specie = stack.pop()

        if specie in visited:
            continue

        visited.add(specie)

        result = con.execute(
            f"""
            SELECT crypticGroup
            FROM {TABLE_NAME}
            WHERE scientificName = ?
            LIMIT 1
            """,
            [specie]
        ).fetchone()

        if result is None:
            continue

        cryptic_group = result[0]

        # convert string → list if needed
        if isinstance(cryptic_group, str):
            cryptic_group = ast.literal_eval(cryptic_group)

        if cryptic_group:
            for s in cryptic_group:
                if s not in visited:
                    stack.append(s)

    return list(visited)


specie = "Salticus scenicus"
group = get_cryptic_group(specie)


# create SQL placeholders (?, ?, ?, ...)
placeholders = ",".join(["?"] * len(group))

result = con.execute(
    f"""
    SELECT scientificName, COUNT(*) as occurrences
    FROM {TABLE_NAME}
    WHERE scientificName IN ({placeholders})
    GROUP BY scientificName
    ORDER BY occurrences DESC
    """,
    group
).fetchall()

print("Occurrences per species:")
for name, count in result:
    print(name, count)

print("Total occurrences in dataset:", sum(c for _, c in result))