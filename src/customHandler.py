import os
import csv
import re
import unicodedata
from difflib import SequenceMatcher
from collections import defaultdict

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
        return "", ""
    prefix, domain = email.split("@", 1)
    tokens = re.split(r"[._\-\d]+", prefix)
    tokens = [t for t in tokens if t]
    return prefix, domain, tokens


def levenshtein_sim(a, b):
    """Approximate similarity using SequenceMatcher ratio."""
    return SequenceMatcher(None, a, b).ratio()


def jaccard_sim(tokens1, tokens2):
    """Compute token overlap similarity."""
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


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

def is_duplicate(dev1, dev2):
    """Apply simplified improved heuristic rules R1–R6."""

    # Unpack
    name1, email1 = dev1
    name2, email2 = dev2

    prefix1, domain1, tokens1 = tokenize_email(email1)
    prefix2, domain2, tokens2 = tokenize_email(email2)

    name1_norm = normalize_text(name1)
    name2_norm = normalize_text(name2)
    prefix1_norm = normalize_text(prefix1)
    prefix2_norm = normalize_text(prefix2)

    # Basic filters
    if any(x in email1 + email2 for x in ["bot", "ci", "auto", "build", "test", "action"]):
        return False

    name_sim = levenshtein_sim(name1_norm, name2_norm)
    prefix_sim = levenshtein_sim(prefix1_norm, prefix2_norm)
    prefix_overlap = jaccard_sim(tokens1, tokens2)

    name1_tokens = name1_norm.split()
    name2_tokens = name2_norm.split()

    firstname1 = name1_tokens[0] if name1_tokens else ""
    firstname2 = name2_tokens[0] if name2_tokens else ""
    lastname1 = name1_tokens[-1] if len(name1_tokens) > 1 else ""
    lastname2 = name2_tokens[-1] if len(name2_tokens) > 1 else ""

    same_domain = (domain1 == domain2)
    same_lastname = levenshtein_sim(lastname1, lastname2) >= 0.9
    firstname_initial_match = firstname1 and firstname2 and firstname1[0] == firstname2[0]

    # Canonicalize nicknames
    firstname1_canon = NICKNAME_MAP.get(firstname1, firstname1)
    firstname2_canon = NICKNAME_MAP.get(firstname2, firstname2)
    nickname_match = (firstname1_canon == firstname2_canon) and same_lastname

    # R1: Exact email match
    if prefix1_norm == prefix2_norm and domain1 == domain2:
        return True

    # R2: Same domain + name typo tolerance
    if same_domain and name_sim >= 0.9:
        return True

    # R3: Prefix variation (initials)
    if same_domain and firstname_initial_match and same_lastname and prefix_sim >= 0.8:
        return True

    # R4: Prefix token overlap
    if prefix_overlap >= 0.75 and same_lastname:
        return True

    # R5: Cross-domain match
    unrelated_domains = {"github.com", "users.noreply.github.com"}
    if domain1 != domain2 and domain1 not in unrelated_domains and domain2 not in unrelated_domains:
        if name_sim >= 0.95 and prefix_overlap >= 0.8:
            return True

    # R6: Nickname match
    if nickname_match:
        return True

    return False


# -----------------------------
# Main Execution
# -----------------------------

projects = [
    "https://github.com/torvalds/linux",
    "https://github.com/flutter/flutter",
    "https://github.com/nodejs/node"
]

BASE_DIR = "src"

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
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            developers.append((row["name"], row["email"]))

    potential_duplicates = []

    for i in range(len(developers)):
        for j in range(i + 1, len(developers)):
            dev1 = developers[i]
            dev2 = developers[j]
            if is_duplicate(dev1, dev2):
                potential_duplicates.append((dev1, dev2))

    # Save results
    results_path = os.path.join(output_dir, "duplicates.csv")
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name1", "email1", "name2", "email2"])
        for (dev1, dev2) in potential_duplicates:
            writer.writerow([dev1[0], dev1[1], dev2[0], dev2[1]])

    print(f"→ Saved {len(potential_duplicates)} potential duplicates to {results_path}")
