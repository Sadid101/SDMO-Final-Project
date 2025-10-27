import csv
import pandas as pd
import unicodedata
import string
import os
from itertools import combinations
from Levenshtein import ratio as sim

# =====================================================
# CONFIGURATION
# =====================================================
BASE_DIR = "src"
THRESHOLD = 0.9

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
    """Cleans and tokenizes a developer's name and email."""
    name: str = dev[0]

    # Remove punctuation, accents, lowercase, and strip whitespace
    trans = name.maketrans("", "", string.punctuation)
    name = name.translate(trans)
    name = unicodedata.normalize('NFKD', name)
    name = ''.join([c for c in name if not unicodedata.combining(c)])
    name = name.casefold()
    name = " ".join(name.split())

    # Split name
    parts = name.split(" ")
    if len(parts) == 2:
        first, last = parts
    elif len(parts) == 1:
        first, last = name, ""
    else:
        first, last = parts[0], " ".join(parts[1:])

    # Get initials
    # Note: Initial must come from a name part of length > 1
    i_first = first[0] if len(first) > 1 else ""
    i_last = last[0] if len(last) > 1 else ""

    email: str = dev[1]
    prefix = email.split("@")[0]

    return name, first, last, i_first, i_last, email, prefix


# =====================================================
# Main loop: For each project, compute and filter similarities
# =====================================================
for project_dir in projects:
    csv_path = os.path.join(project_dir, "devs.csv")
    if not os.path.exists(csv_path):
        print(f"⚠️ Skipping {project_dir}: devs.csv not found.")
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
    MIN_LAST_LENGTH = 2  # Only consider name parts with at least 2 characters

    for dev_a, dev_b in combinations(DEVS, 2):
        name_a, first_a, last_a, i_first_a, i_last_a, email_a, prefix_a = process(dev_a)
        name_b, first_b, last_b, i_first_b, i_last_b, email_b, prefix_b = process(dev_b)

        # C1-C3: Levenshtein Ratio Comparisons
        c1 = sim(name_a, name_b)
        c2 = sim(prefix_a, prefix_b)
        c31 = sim(first_a, first_b) if first_a and first_b else 0
        c32 = sim(last_a, last_b) if last_a and last_b else 0

        # C4-C7: Strict Pattern Checks (Fixed to require exact prefix match)
        
        # Helper function for strict pattern check
        def is_exact_match(pattern, prefix):
            # Check only if both pattern and prefix are non-empty
            if not pattern or not prefix:
                return False
            # Require the pattern to equal the prefix
            return pattern.lower() == prefix.lower()

        # C4-C7: Strict Pattern Checks (Fixed with bool() for safe int() conversion)
        
        # c4: first initial + last of A in prefix of B
        pattern_c4 = i_first_a + last_a
        c4 = int(bool(len(last_a) >= MIN_LAST_LENGTH and 
                      i_first_a and 
                      is_exact_match(pattern_c4, prefix_b)))

        # c5: last initial + first of A in prefix of B
        pattern_c5 = i_last_a + first_a
        # This is where your error likely occurred (Line 117 approximation)
        c5 = int(bool(len(first_a) >= MIN_LAST_LENGTH and 
                      i_last_a and 
                      is_exact_match(pattern_c5, prefix_b)))

        # c6: first initial + last of B in prefix of A
        pattern_c6 = i_first_b + last_b
        c6 = int(bool(len(last_b) >= MIN_LAST_LENGTH and 
                      i_first_b and 
                      is_exact_match(pattern_c6, prefix_a)))

        # c7: last initial + first of B in prefix of A
        pattern_c7 = i_last_b + first_b
        c7 = int(bool(len(first_b) >= MIN_LAST_LENGTH and 
                      i_last_b and 
                      is_exact_match(pattern_c7, prefix_a)))

        SIMILARITY.append([
            dev_a[0], email_a,
            dev_b[0], email_b,
            c1, c2, c31, c32, c4, c5, c6, c7
        ])

    # Save full similarity data
    cols = ["name_1", "email_1", "name_2", "email_2",
            "c1", "c2", "c3.1", "c3.2", "c4", "c5", "c6", "c7"]
    df = pd.DataFrame(SIMILARITY, columns=cols)

    # full_path = os.path.join(project_dir, "devs_similarity.csv")
    # df.to_csv(full_path, index=False, header=True)
    # print(f"Saved full similarity data to {full_path}")

    # Apply original threshold filtering (ANY condition)
    
    # 1. Define checks for Levenshtein criteria (c1, c2, c3)
    df["c1_check"] = df["c1"] >= THRESHOLD
    df["c2_check"] = df["c2"] >= THRESHOLD
    df["c3_check"] = (df["c3.1"] >= THRESHOLD) & (df["c3.2"] >= THRESHOLD)
    
    # 2. Filter: Select pairs where ANY check (c1, c2, c3, c4, c5, c6, c7) is True
    df_filtered = df[df[["c1_check", "c2_check", "c3_check", "c4", "c5", "c6", "c7"]].any(axis=1)]
    print(f"Threshold {THRESHOLD:.2f}: {len(df_filtered)} candidate pairs")
    
    # 3. Select original columns for output
    df_filtered = df_filtered[cols]

    t_path = os.path.join(project_dir, f"devs_similarity_t={THRESHOLD}.csv")
    df_filtered.to_csv(t_path, index=False, header=True)
    print(f"✅ Filtered results saved to {t_path}")

print("\n🎉 Processing complete for all repositories.")