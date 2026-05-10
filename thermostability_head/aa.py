"""
Download/parse the Meltome Atlas dataset that the Steering Protein Language
Models paper used, and produce train_split.csv / test_split.csv.

The paper (Long-Kai Huang et al.) reports a final dataset of 24,817 train +
3,134 test = 27,951 proteins, with the following preprocessing:

  * median melting temperature across all species per protein
  * 90/10 train/test split
  * max 90% sequence identity within the training set
  * any training sequence with >=30% identity to a test sequence is removed

The FLIP benchmark (https://github.com/J-SNACKKB/FLIP/tree/main/splits/meltome)
publishes a curated Meltome FASTA, `mixed_split.fasta`, that already contains
exactly 27,951 sequences, with TARGET= median Tm values pre-aggregated across
species and SET=train / SET=test labels matching the paper's 24,817 / 3,134
counts. Reusing it lets us replicate the paper's input data exactly, instead
of re-fetching by accession from UniProt (which loses obsolete/merged IDs)
and re-doing the cross-species median + clustering ourselves.

The header format is:
    >SequenceN TARGET=<float> SET=<train|test> VALIDATION=<True|False>
    <amino-acid-sequence>
"""

import os
import re
import urllib.request
from pathlib import Path

import pandas as pd


HERE = Path(__file__).parent
FASTA_URL = "http://data.bioembeddings.com/public/FLIP/fasta/meltome/mixed_split.fasta"
FASTA_PATH = HERE / "mixed_split.fasta"

HEADER_RE = re.compile(
    r"^>(?P<id>\S+)\s+TARGET=(?P<target>[-0-9.eE+]+)\s+SET=(?P<set>\S+)"
)


def download_fasta(path: Path, url: str = FASTA_URL):
    if path.exists() and path.stat().st_size > 0:
        print(f"Using existing FASTA at {path} ({path.stat().st_size} bytes)")
        return
    print(f"Downloading {url} -> {path}")
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    print(f"Downloaded {path.stat().st_size} bytes")


def parse_fasta(path: Path):
    """Parse FLIP-style fasta. Yields (id, target, set, sequence)."""
    cur_id = None
    cur_target = None
    cur_set = None
    cur_seq = []

    def flush():
        if cur_id is None:
            return None
        return (cur_id, cur_target, cur_set, "".join(cur_seq))

    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                rec = flush()
                if rec is not None:
                    yield rec
                m = HEADER_RE.match(line)
                if m is None:
                    raise ValueError(f"Unrecognized FASTA header: {line!r}")
                cur_id = m.group("id")
                cur_target = float(m.group("target"))
                cur_set = m.group("set").lower()
                cur_seq = []
            else:
                cur_seq.append(line.strip())

    rec = flush()
    if rec is not None:
        yield rec


def main():
    download_fasta(FASTA_PATH)

    rows = list(parse_fasta(FASTA_PATH))
    df = pd.DataFrame(rows, columns=["Protein_ID", "median_Tm", "set", "sequence"])

    # Defensive cleanup: drop empty rows. ESM2's vocab covers the 20 standard
    # amino acids plus X / B / U / Z / O, all of which appear (or could appear)
    # in this dataset; we don't filter by AA letters here.
    df = df.dropna(subset=["sequence", "median_Tm"])
    df = df[df["sequence"].str.len() > 0]

    print("Total records:", len(df))
    print("Per-set counts:")
    print(df["set"].value_counts())
    print("Tm summary:")
    print(df["median_Tm"].describe())
    print("Sequence-length summary:")
    print(df["sequence"].str.len().describe())

    train_df = df[df["set"] == "train"][["Protein_ID", "sequence", "median_Tm"]]
    test_df = df[df["set"] == "test"][["Protein_ID", "sequence", "median_Tm"]]

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Train size: {len(train_df)}  (paper: 24,817)")
    print(f"Test size:  {len(test_df)}  (paper: 3,134)")

    # Also save a unified CSV for inspection / downstream tasks.
    df[["Protein_ID", "sequence", "median_Tm", "set"]].to_csv(
        HERE / "tm_sequence_dataset.csv", index=False
    )
    train_df.to_csv(HERE / "train_split.csv", index=False)
    test_df.to_csv(HERE / "test_split.csv", index=False)

    print("Wrote tm_sequence_dataset.csv, train_split.csv, test_split.csv")


if __name__ == "__main__":
    main()
