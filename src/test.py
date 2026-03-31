import duckdb
from data_analysis import count_occurrences, get_all_cryptic_groups, plot_occurrences

con = duckdb.connect(":memory:")
TABLE_NAME = "test"

# create new test table
con.execute(f"""
CREATE TABLE {TABLE_NAME} (
    scientificName VARCHAR,
    crypticGroup VARCHAR[]
)
""")

# insert test data
con.execute(f"""
INSERT INTO {TABLE_NAME} VALUES
('A', ['B','C']),
('A', ['B','C']),
('B', ['D']),
('C', []),
('D', ['A']),
('E', ['F']),
('F', []),
('G', [])
""")


if __name__ == "__main__":
    all_cryptic_groups = get_all_cryptic_groups(TABLE_NAME, con)
    occurrences_df = count_occurrences(TABLE_NAME, con, all_cryptic_groups)
    occurrences_df.to_csv("src/data_analysis_results/test_groups_occurrences.csv", index=False)
    plot_occurrences(occurrences_df, "test_groups_plot.png")