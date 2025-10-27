import os
import csv
import re
import unicodedata
from difflib import SequenceMatcher
from collections import defaultdict

# -----------------------------
# Configuration
# -----------------------------

# The single primary threshold for determining similarity.
DUPLICATE_THRESHOLD = 0.85 

# Common nicknames mapping (can be expanded)
NICKNAME_MAP = {
    "bob": "robert",
    "rob": "robert",
    "liz": "elizabeth",
    "beth": "elizabeth",
    "mike": "michael",
    "tom": "thomas",
    "jen": "jennifer"
}

# -----------------------------
# Helper functions
# -----------------------------

def normalize_text(text):
    """Normalize names and emails: lowercase, strip accents, remove noise."""
    if not text:
        return ""
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Remove noise like commit messages or bot annotations
    text = re.split(r"[\|\(\[\:]", text)[0]
    text = re.sub(r"[^a-z0-9@._\- ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_email(email):
    """Split email into prefix and domain, and tokenize the prefix."""
    if "@" not in email:
        return "", "", [] # Fixed to always return 3 values
    prefix, domain = email.split("@", 1)
    tokens = re.split(r"[._\-\d]+", prefix)
    tokens = [t for t in tokens if t]
    return prefix, domain, tokens


def levenshtein_sim(a, b):
    """Approximate similarity using SequenceMatcher ratio (Levenshtein proxy)."""
    return SequenceMatcher(None, a, b).ratio()


def jaccard_sim(tokens1, tokens2):
    """Compute token overlap similarity."""
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def canonical_name(name):
    """Return canonicalized version of a first name using nickname map."""
    parts = name.split()
    if not parts:
        return ""
    first = parts[0]
    first = NICKNAME_MAP.get(first, first)
    return " ".join([first] + parts[1:])


# -----------------------------
# Duplicate Detection Heuristic
# -----------------------------

def check_rules_and_metrics(dev1, dev2):
    """
    Applies the original R1-R6 heuristic rules AND computes the raw C-metrics 
    (based on Levenshtein ratio) for a fair comparison.
    
    Returns: (list_of_matching_rules, dict_of_c_metrics)
    """
    
    # Define thresholds relative to the main one
    STRICT_SIM = DUPLICATE_THRESHOLD + 0.1     # e.g., 0.95
    NORMAL_SIM = DUPLICATE_THRESHOLD          # e.g., 0.85
    LAX_SIM = DUPLICATE_THRESHOLD - 0.15      # e.g., 0.70
    
    # Unpack
    name1, email1 = dev1
    name2, email2 = dev2

    prefix1, domain1, tokens1 = tokenize_email(email1)
    prefix2, domain2, tokens2 = tokenize_email(email2)

    name1_norm = normalize_text(name1)
    name2_norm = normalize_text(name2)
    prefix1_norm = normalize_text(prefix1)
    prefix2_norm = normalize_text(prefix2)

    # List to store all rules that fire for this pair
    matching_rules = []

    # Basic filters (Keep as-is)
    if any(x in email1 + email2 for x in ["bot", "ci", "auto", "build", "test", "action"]):
        return [], {}

    # --- Metrics (C1, C2, C3.1, C3.2 - all Levenshtein ratios) ---
    
    # C1 (Name Similarity)
    name_sim = levenshtein_sim(name1_norm, name2_norm)
    
    # C2 (Email Prefix Similarity)
    prefix_sim = levenshtein_sim(prefix1_norm, prefix2_norm)
    
    # C-helpers for R-rules and C3.x
    prefix_overlap = jaccard_sim(tokens1, tokens2)

    name1_tokens = name1_norm.split()
    name2_tokens = name2_norm.split()

    first1 = name1_tokens[0] if name1_tokens else ""
    first2 = name2_tokens[0] if name2_tokens else ""
    last1 = name1_tokens[-1] if len(name1_tokens) > 1 else ""
    last2 = name2_tokens[-1] if len(name2_tokens) > 1 else ""

    # C3.1 (First Name Similarity)
    c31 = levenshtein_sim(first1, first2) if first1 and first2 else 0
    
    # C3.2 (Last Name Similarity)
    c32 = levenshtein_sim(last1, last2) if last1 and last2 else 0
    
    # --- R-Rule Prerequisites (Calculated using Levenshtein ratios) ---

    same_domain = (domain1 == domain2)
    
    # R3, R4, R6 rely on same_lastname being true
    # This uses c32 (last name Levenshtein ratio)
    same_lastname = c32 >= NORMAL_SIM 
    
    # R3 relies on initial match
    firstname_initial_match = first1 and first2 and first1[0] == first2[0]

    # R6 relies on nickname match
    firstname1_canon = NICKNAME_MAP.get(first1, first1)
    firstname2_canon = NICKNAME_MAP.get(first2, first2)
    nickname_match = (firstname1_canon == firstname2_canon) and same_lastname


    # --- R-Rules (Original Logic Preserved) ---
    
    # R1: Exact email match
    if prefix1_norm == prefix2_norm and domain1 == domain2:
        matching_rules.append("R1")

    # R2: Same domain + name typo tolerance 
    if same_domain and name_sim >= STRICT_SIM: # Uses name_sim (Levenshtein)
        matching_rules.append("R2")

    # R3: Prefix variation (initials) - Must have same domain
    # Uses prefix_sim (Levenshtein) and same_lastname (derived from Levenshtein)
    if same_domain and firstname_initial_match and same_lastname and prefix_sim >= NORMAL_SIM:
        matching_rules.append("R3")

    # R4: Prefix token overlap - Requires some name evidence (e.g., last name match)
    # Uses same_lastname (derived from Levenshtein)
    if prefix_overlap >= LAX_SIM and same_lastname:
        matching_rules.append("R4")

    # R5: Cross-domain match - Requires very high name and email overlap
    # Uses name_sim (Levenshtein)
    unrelated_domains = {"github.com", "users.noreply.github.com"}
    if domain1 != domain2 and domain1 not in unrelated_domains and domain2 not in unrelated_domains:
        if name_sim >= STRICT_SIM and prefix_overlap >= NORMAL_SIM:
            matching_rules.append("R5")

    # R6: Nickname match + Email prefix evidence 
    # Uses nickname_match (derived from Levenshtein) and prefix_sim (Levenshtein)
    if nickname_match:
        if prefix_sim >= LAX_SIM or prefix_overlap >= (LAX_SIM - 0.2):
            matching_rules.append("R6")
            
    # Compile C-metrics for output, aligning with the second script's names
    c_metrics = {
        "c1": name_sim, 
        "c2": prefix_sim, 
        "c3.1": c31, 
        "c3.2": c32
        # C4-C7 are not strictly required here but can be added if needed for max comparability
    }

    return matching_rules, c_metrics


# -----------------------------
# Main Execution (Revised Output Logic)
# -----------------------------

projects = [
    "https://github.com/facebook/react",
    "https://github.com/flutter/flutter",
    "https://github.com/Homebrew/homebrew-core"
    "https://github.com/nodejs/node",
    "https://github.com/torvalds/linux",
    "https://github.com/rails/rails"
]

BASE_DIR = "src"
ALL_RULES = ["R1", "R2", "R3", "R4", "R5", "R6"] # Fixed set of rule columns
ALL_C_METRICS = ["c1", "c2", "c3.1", "c3.2"] # Levenshtein ratio metrics

for project_url in projects:
    repo_name = project_url.split("/")[-2] + "_" + project_url.split("/")[-1]
    output_dir = os.path.join(BASE_DIR, repo_name)

    # Assume commit data stored as 'authors.csv' with columns: name,email
    csv_path = os.path.join(output_dir, "devs.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping {repo_name}: authors.csv not found")
        continue

    print(f"Processing {repo_name}...")

    developers = []
    # Reverting to the original, simpler file read logic to avoid complex error handling issues
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Assuming the CSV has 'name' and 'email' columns
                developers.append((row["name"], row["email"]))
    except KeyError:
        print(f"Error: {csv_path} must contain 'name' and 'email' columns.")
        continue # Skip this project if columns are missing
    except FileNotFoundError:
        # This should be caught by os.path.exists, but here for robustness
        print(f"Error: {csv_path} not found.")
        continue


    potential_duplicates_rows = []

    for i in range(len(developers)):
        for j in range(i + 1, len(developers)):
            dev1 = developers[i]
            dev2 = developers[j]
            
            # Get the list of rules that fired AND the C-metrics
            matching_rules, c_metrics = check_rules_and_metrics(dev1, dev2)
            
            # Only process if at least one rule fired
            if matching_rules:
                # 1. Start the row with developer details
                row = [dev1[0], dev1[1], dev2[0], dev2[1]]
                
                # 2. Add a column for each R-rule (binary indicator)
                for rule in ALL_RULES:
                    # Use a binary indicator: 1 if the rule fired, 0 otherwise
                    indicator = 1 if rule in matching_rules else 0
                    row.append(indicator)
                
                # 3. Add a column for each C-metric (raw score)
                for metric in ALL_C_METRICS:
                    # Append the Levenshtein ratio
                    row.append(c_metrics.get(metric, 0.0))
                
                potential_duplicates_rows.append(row)


    # Save results
    results_path = os.path.join(output_dir, "combined_duplicates_metrics.csv")
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Create the header: name1, email1, name2, email2, R1-R6, C1-C3.2
        header = ["name1", "email1", "name2", "email2"] + ALL_RULES + ALL_C_METRICS
        writer.writerow(header) 
        
        for row in potential_duplicates_rows:
            writer.writerow(row)

    print(f"→ Saved {len(potential_duplicates_rows)} potential duplicate pairs to {results_path} (R-rules and Levenshtein ratios included)")

print("\n🎉 Processing complete for all repositories.")