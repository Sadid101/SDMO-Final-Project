import csv
import pandas as pd
import unicodedata
import string
from itertools import combinations
from Levenshtein import ratio as sim
import os

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = "src"
THRESHOLD = 0.7

# Automatically detect all subprojects inside src/
projects = [
    os.path.join(BASE_DIR, d)
    for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d))
]

print(f"Found {len(projects)} projects: {projects}")


# =====================================================
# Helper Function: Preprocess developer data
# =====================================================
def process(dev):
    name: str = dev[0]

    # Remove punctuation
    trans = name.maketrans("", "", string.punctuation)
    name = name.translate(trans)
    # Remove accents
    name = unicodedata.normalize('NFKD', name)
    name = ''.join([c for c in name if not unicodedata.combining(c)])
    # Lowercase
    name = name.casefold()
    # Strip whitespace
    name = " ".join(name.split())

    # Split name
    parts = name.split(" ")
    if len(parts) == 2:
        first, last = parts
    elif len(parts) == 1:
        first, last = name, ""
    else:
        first, last = parts[0], " ".join(parts[1:])

    i_first = first[0] if len(first) > 1 else ""
    i_last = last[0] if len(last) > 1 else ""

    email: str = dev[1]
    prefix = email.split("@")[0]

    return name, first, last, i_first, i_last, email, prefix


# =====================================================
# Main loop: For each project, compute similarities
# =====================================================
for project_dir in projects:
    csv_path = os.path.join(project_dir, "devs.csv")
    if not os.path.exists(csv_path):
        print(f"⚠️  Skipping {project_dir}: devs.csv not found.")
        continue

    print(f"\n🔍 Processing {project_dir}...")

    # Read developer list
    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        DEVS = [row for row in reader]

    print(f"Loaded {len(DEVS)} developers.")

    # Skip empty or too small lists
    if len(DEVS) < 2:
        print("Not enough developers to compare.")
        continue

    # Compute all pairwise similarities
    SIMILARITY = []
    for dev_a, dev_b in combinations(DEVS, 2):
        name_a, first_a, last_a, i_first_a, i_last_a, email_a, prefix_a = process(dev_a)
        name_b, first_b, last_b, i_first_b, i_last_b, email_b, prefix_b = process(dev_b)

        c1 = sim(name_a, name_b)
        c2 = sim(prefix_b, prefix_a)
        c31 = sim(first_a, first_b)
        c32 = sim(last_a, last_b)
        c4 = c5 = c6 = c7 = False

        if i_first_a and last_a:
            c4 = i_first_a in prefix_b and last_a in prefix_b
        if i_last_a:
            c5 = i_last_a in prefix_b and first_a in prefix_b
        if i_first_b and last_b:
            c6 = i_first_b in prefix_a and last_b in prefix_a
        if i_last_b:
            c7 = i_last_b in prefix_a and first_b in prefix_a

        SIMILARITY.append([
            dev_a[0], email_a,
            dev_b[0], email_b,
            c1, c2, c31, c32, c4, c5, c6, c7
        ])

    # Save full similarity data
    cols = ["name_1", "email_1", "name_2", "email_2",
            "c1", "c2", "c3.1", "c3.2", "c4", "c5", "c6", "c7"]
    df = pd.DataFrame(SIMILARITY, columns=cols)

    full_path = os.path.join(project_dir, "devs_similarity.csv")
    df.to_csv(full_path, index=False, header=True)
    print(f"Saved full similarity data to {full_path}")

    # Apply threshold filtering
    df["c1_check"] = df["c1"] >= THRESHOLD
    df["c2_check"] = df["c2"] >= THRESHOLD
    df["c3_check"] = (df["c3.1"] >= THRESHOLD) & (df["c3.2"] >= THRESHOLD)

    df_filtered = df[df[["c1_check", "c2_check", "c3_check", "c4", "c5", "c6", "c7"]].any(axis=1)]
    df_filtered = df_filtered[cols]

    t_path = os.path.join(project_dir, f"devs_similarity_t={THRESHOLD}.csv")
    df_filtered.to_csv(t_path, index=False, header=True)
    print(f"✅ Filtered results saved to {t_path}")

print("\n🎉 Processing complete for all repositories.")
