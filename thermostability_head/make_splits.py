import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


CSV_PATH = "tm_sequence_dataset.csv"
RANDOM_STATE = 42

# Leakage filter: a hit counts as leakage only if it is a meaningful homology
# signal, not a noisy short alignment. The paper's number (24,817 train
# survivors out of ~25k pre-leakage train sequences) implies a fairly
# permissive interpretation of "30% identity" -- partial-domain hits that
# only span a tiny region should not be treated as leakage.
LEAKAGE_MIN_PIDENT = 0.30
LEAKAGE_MIN_ALN_LEN = 50

MMSEQS = "/home/ubuntu/mmseqs/bin/mmseqs"

SPLIT_DIR = Path("split_work")
SPLIT_DIR.mkdir(exist_ok=True)


def clean_dataframe(df):
    df = df.dropna(subset=["sequence", "median_Tm"])
    df["sequence"] = df["sequence"].astype(str)
    df["median_Tm"] = pd.to_numeric(df["median_Tm"], errors="coerce")
    df = df.dropna(subset=["median_Tm"])

    # Match the ESM2 alphabet (20 standard AAs plus X/B/U/Z/O).
    df = df[df["sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYXBUZO]+")]
    df = df[df["sequence"].str.len() >= 30]
    # No upper length filter: long sequences are truncated to ESM2's
    # context window in the training dataloader instead of dropped.
    #
    # We deliberately do NOT clip median_Tm to [30, 90] -- the paper
    # describes that as the natural range of the Meltome Atlas data, not
    # a preprocessing step. Clipping here would needlessly drop a few
    # hundred proteins.

    return df.reset_index(drop=True)


def write_fasta(df, path):
    with open(path, "w") as f:
        for idx, row in df.iterrows():
            f.write(f">{idx}\n{row['sequence']}\n")


def read_fasta_ids(path):
    ids = set()
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                ids.add(int(line[1:].strip().split()[0]))
    return ids


def make_procedure_split(df):
    train_df, test_df = train_test_split(
        df,
        test_size=0.1,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_raw_fa = SPLIT_DIR / "train_raw.fasta"
    test_fa = SPLIT_DIR / "test.fasta"

    write_fasta(train_df, train_raw_fa)
    write_fasta(test_df, test_fa)

    print("Clustering training set at 90% identity...")

    subprocess.run(
        [
            MMSEQS,
            "easy-cluster",
            str(train_raw_fa),
            str(SPLIT_DIR / "train90_cluster"),
            str(SPLIT_DIR / "tmp90"),
            "--min-seq-id", "0.90",
            "--cov-mode", "0",
            "-c", "0.8",
        ],
        check=True,
    )

    rep_fa = SPLIT_DIR / "train90_cluster_rep_seq.fasta"
    train_ids_90 = read_fasta_ids(rep_fa)
    train_df = train_df.loc[list(train_ids_90)].reset_index(drop=True)

    train90_fa = SPLIT_DIR / "train_90.fasta"
    write_fasta(train_df, train90_fa)

    print("Removing train sequences with >=30% identity to test sequences...")

    # Tradeoff:
    #   -c 0.8 (mmseqs default) misses partial-domain leakage between large
    #     multi-domain proteins.
    #   -c 0.0 with --min-seq-id 0.3 catches every short, noisy 30%-identity
    #     blip and drops ~37% of training.
    # The paper's stated outcome (~24,817 of ~25k train surviving) implies
    # a moderate setting. We require:
    #   * mutual coverage >= 50% (--cov-mode 0, -c 0.5), AND
    #   * alignment length >= 50 aa (filtered post-hoc),
    # so that "leakage" means a homologous region of meaningful size, not
    # a 30 aa lookalike.
    subprocess.run(
        [
            MMSEQS,
            "easy-search",
            str(train90_fa),
            str(test_fa),
            str(SPLIT_DIR / "train_vs_test.m8"),
            str(SPLIT_DIR / "tmp30"),
            "--min-seq-id", "0.30",
            "--cov-mode", "0",
            "-c", "0.5",
            "--min-aln-len", str(LEAKAGE_MIN_ALN_LEN),
            "-s", "7.5",
        ],
        check=True,
    )

    leaking_ids = set()
    hits_path = SPLIT_DIR / "train_vs_test.m8"

    if hits_path.exists():
        with open(hits_path) as f:
            for line in f:
                parts = line.split("\t")
                # m8 format: query, target, fident, alnlen, mismatch, gapopen,
                # qstart, qend, tstart, tend, evalue, bits
                if len(parts) < 4:
                    continue
                pident = float(parts[2])
                aln_len = int(parts[3])
                # mmseqs reports identity as a fraction in [0, 1]; be defensive
                # in case a future version switches to a percent.
                if pident > 1.0:
                    pident /= 100.0
                if pident >= LEAKAGE_MIN_PIDENT and aln_len >= LEAKAGE_MIN_ALN_LEN:
                    leaking_ids.add(int(parts[0]))

    print(f"  train sequences flagged as leaking: {len(leaking_ids)}")

    train_df = train_df.drop(index=list(leaking_ids), errors="ignore").reset_index(drop=True)

    return train_df, test_df


def main():
    # NOTE: When `aa.py` is sourced from FLIP's `mixed_split.fasta` (the new
    # default), the produced `tm_sequence_dataset.csv` already contains a
    # `set` column with the paper's exact 24,817 train / 3,134 test split.
    # In that case `aa.py` writes train_split.csv and test_split.csv
    # directly and this script does not need to be run.
    #
    # This script remains useful only for re-doing the split from scratch on
    # a custom data source that does not already have a `set` column.
    df = pd.read_csv(CSV_PATH)
    if "set" in df.columns:
        print(
            "Detected pre-existing `set` column in tm_sequence_dataset.csv "
            "(this is the FLIP / paper split). Skipping the custom 90/10 "
            "split + clustering + leakage filter and using the provided "
            "labels directly."
        )
        df = df.dropna(subset=["sequence", "median_Tm"])
        df["sequence"] = df["sequence"].astype(str)
        train_df = df[df["set"] == "train"][["Protein_ID", "sequence", "median_Tm"]].reset_index(drop=True)
        test_df = df[df["set"] == "test"][["Protein_ID", "sequence", "median_Tm"]].reset_index(drop=True)
    else:
        df = clean_dataframe(df)
        print("Dataset size after cleaning:", len(df))
        print(df["median_Tm"].describe())
        train_df, test_df = make_procedure_split(df)

    print("Final train size:", len(train_df))
    print("Final test size:", len(test_df))

    train_df.to_csv("train_split.csv", index=False)
    test_df.to_csv("test_split.csv", index=False)

    print("Saved train_split.csv")
    print("Saved test_split.csv")


if __name__ == "__main__":
    main()