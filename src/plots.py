#from datasets import load_dataset
import pandas as pd

df = pd.read_parquet("part_0.parquet")

data = {
    "scientificName": ["A", "B", "C", "D", "E", "F"],
    "crypticGroup": [
        ["B", "C"],  # A connects to B, C
        ["D"],       # B connects to D
        [],          # C connects to nothing
        ["A"],       # D connects back to A
        ["F"],       # E connects to F
        []           # F connects to nothing
    ]
}

test_df = pd.DataFrame(data)


def get_cryptic_group(specie):
    visited = set()
    stack = [specie]

    while stack:
        specie = stack.pop()

        if specie in visited:
            continue

        visited.add(specie)

        rows = test_df[test_df["scientificName"] == specie]
        if rows.empty:
            continue

        cryptic_group = rows["crypticGroup"].iloc[0]

        if cryptic_group is not None:
            for s in cryptic_group:
                if s not in visited:
                    stack.append(s)
    return list(visited)


specie = test_df["scientificName"].iloc[0]
group = get_cryptic_group(specie)

print(group)