"""
Extract all lysozyme-like UniRef50 clusters using InterPro IPR023346.

Pipeline:
  1. Query UniProt's stream API for all UniProtKB entries belonging to
     InterPro IPR023346 (Lysozyme-like domain superfamily).
  2. Map those UniProt accessions to their UniRef50 cluster IDs using
     UniProt's ID Mapping API (UniRef is NOT a returnable column on
     the UniProtKB endpoint, so we need this separate call).
  3. Fetch each UniRef50 cluster in batches via UniProt's UniRef stream
     API in FASTA format.
  4. Write a final FASTA + a metadata TSV.

Outputs:
  out_dir/
    lysozyme_uniref50.fasta     # one record per UniRef50 cluster
    lysozyme_uniref50.tsv       # cluster_id, length, name, member_uniprot_accs
    lysozyme_interpro_hits.tsv  # raw UniProt hits before dedup
"""
from __future__ import annotations
import argparse
import gzip
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

INTERPRO_ID_DEFAULT = "IPR023346"  # Lysozyme-like domain superfamily
UNIPROT_BASE = "https://rest.uniprot.org"
ID_MAPPING_BASE = f"{UNIPROT_BASE}/idmapping"


def _save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f)
    tmp.replace(path)


def _load_json(path: Path):
    with path.open() as f:
        return json.load(f)

def make_session() -> requests.Session:
    """A requests session with retries on 5xx / 429 / connection errors."""
    s = requests.Session()
    retry = Retry(
        total=8,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8))
    return s

def fetch_uniprot_hits(session: requests.Session, interpro_id: str) -> list[dict]:
    """Pull all UniProtKB entries belonging to ``interpro_id`` via /search +
    cursor pagination. Raises on partial failure rather than silently
    returning truncated data.
    """
    url = f"{UNIPROT_BASE}/uniprotkb/search"
    params = {
        "query": f"xref:interpro-{interpro_id}",
        "format": "json",
        "fields": "accession,id,length,organism_name,protein_name,reviewed",
        "size": 500,
    }

    rows: list[dict] = []
    next_url = f"{url}?{urlencode(params)}"
    page_idx = 0
    total: int | None = None
    bar: tqdm | None = None

    print(f"[1/3] Fetching UniProt hits via pagination...", file=sys.stderr)

    try:
        while next_url:
            r = session.get(next_url, timeout=(30, 120))
            r.raise_for_status()
            data = r.json()

            if total is None:
                total = int(r.headers.get("x-total-results", "0")) or None
                bar = tqdm(total=total, unit="entry", desc="  fetched")

            for entry in data.get("results", []):
                entry_type = entry.get("entryType", "")
                rows.append(
                    {
                        "Entry": entry.get("primaryAccession", ""),
                        "Entry Name": entry.get("uniProtkbId", ""),
                        "Length": entry.get("sequence", {}).get("length", 0),
                        "Organism": entry.get("organism", {}).get(
                            "scientificName", ""
                        ),
                        "Protein names": entry.get("proteinDescription", {})
                        .get("recommendedName", {})
                        .get("fullName", {})
                        .get("value", "unknown"),
                        "Reviewed": (
                            "reviewed"
                            if entry_type == "UniProtKB reviewed (Swiss-Prot)"
                            else "unreviewed"
                        ),
                    }
                )

            if bar is not None:
                bar.update(len(data.get("results", [])))

            link_header = r.headers.get("Link", "")
            if 'rel="next"' in link_header:
                next_url = link_header.split("<", 1)[1].split(">", 1)[0]
            else:
                next_url = None
            page_idx += 1
    except Exception as e:
        # Fail loudly: partial data flowing into stage 2 silently truncates
        # the final dataset and is hard to diagnose later.
        print(
            f"\n[!] Pagination failed at page {page_idx} after collecting "
            f"{len(rows):,} entries: {e}",
            file=sys.stderr,
        )
        if bar is not None:
            bar.close()
        raise

    if bar is not None:
        bar.close()
    print(
        f"      finished: {len(rows):,} entries across {page_idx} pages",
        file=sys.stderr,
    )
    return rows

def map_uniprot_to_uniref50(
    session: requests.Session,
    accessions: list[str],
    chunk_size: int = 90_000,  # UniProt ID Mapping limit is 100k per job
    poll_interval: float = 3.0,
    timeout_s: float = 1800.0,
) -> dict[str, str]:
    """Map UniProtKB accessions to their UniRef50 cluster IDs via the
    UniProt ID Mapping API. Returns {uniprot_accession: uniref50_cluster_id}.
    """
    out: dict[str, str] = {}
    if not accessions:
        return out

    print(
        f"      submitting ID mapping job(s) for {len(accessions):,} accessions ...",
        file=sys.stderr,
    )

    n_chunks = (len(accessions) + chunk_size - 1) // chunk_size
    for i in range(0, len(accessions), chunk_size):
        chunk = accessions[i : i + chunk_size]
        chunk_num = i // chunk_size + 1

        print(
            f"      chunk {chunk_num}/{n_chunks}: submitting job for {len(chunk):,} IDs ...",
            file=sys.stderr,
        )
        resp = session.post(
            f"{ID_MAPPING_BASE}/run",
            data={
                "from": "UniProtKB_AC-ID",
                "to": "UniRef50",
                "ids": ",".join(chunk),
            },
            timeout=120,
        )
        resp.raise_for_status()
        job_id = resp.json()["jobId"]
        print(f"        job_id={job_id}", file=sys.stderr)

        # Poll for completion. UniProt signals "done" by either:
        #   (a) HTTP 200 with {"jobStatus":"FINISHED"} or a results payload,
        #   (b) HTTP 303 redirect (Location: .../idmapping/<db>/results/{jobId})
        #       with possibly empty body.
        # We detect both. Without this, allow_redirects=False + 303 would
        # leave the loop polling forever.
        deadline = time.time() + timeout_s
        results_url: str | None = None
        elapsed_start = time.time()
        while True:
            s = session.get(
                f"{ID_MAPPING_BASE}/status/{job_id}",
                timeout=60,
                allow_redirects=False,
            )
            if s.status_code == 303:
                results_url = s.headers.get("Location")
                break
            s.raise_for_status()
            try:
                data = s.json()
            except ValueError:
                data = {}
            status = data.get("jobStatus")
            if status == "FINISHED" or "results" in data or "failedIds" in data:
                break
            if status == "ERROR":
                raise RuntimeError(f"ID mapping job {job_id} failed: {data}")
            if time.time() > deadline:
                raise TimeoutError(
                    f"ID mapping job {job_id} did not finish in {timeout_s}s"
                )
            time.sleep(poll_interval)
        print(
            f"        job finished after {time.time() - elapsed_start:.1f}s, "
            f"streaming results ...",
            file=sys.stderr,
        )

        # Prefer the /stream endpoint (one request, no pagination). Fall back
        # to /results with cursor pagination if /stream fails.
        n_mapped_chunk = 0
        if not results_url:
            results_url = f"{ID_MAPPING_BASE}/uniref/results/{job_id}"

        # Convert /results URL -> /stream URL when possible
        stream_url = re.sub(r"/results/", "/stream/", results_url, count=1)
        stream_url = stream_url.split("?")[0] + "?format=json"

        try:
            r = session.get(stream_url, timeout=(30, 600))
            r.raise_for_status()
            payload = r.json()
            for row in payload.get("results", []):
                target = row["to"]
                if isinstance(target, dict):
                    target = target.get("id") or target.get(
                        "representativeMember", {}
                    ).get("memberId", "")
                if target:
                    out[row["from"]] = target
                    n_mapped_chunk += 1
        except Exception as e:
            print(
                f"        /stream failed ({e}); falling back to paginated /results",
                file=sys.stderr,
            )
            url = results_url + (
                "&" if "?" in results_url else "?"
            ) + "format=json&size=500"
            while url:
                r = session.get(url, timeout=120)
                r.raise_for_status()
                payload = r.json()
                for row in payload.get("results", []):
                    target = row["to"]
                    if isinstance(target, dict):
                        target = target.get("id") or target.get(
                            "representativeMember", {}
                        ).get("memberId", "")
                    if target:
                        out[row["from"]] = target
                        n_mapped_chunk += 1
                link = r.headers.get("Link", "")
                nxt = None
                if 'rel="next"' in link:
                    nxt = link.split("<", 1)[1].split(">", 1)[0]
                url = nxt

        print(
            f"        chunk {chunk_num}: mapped {n_mapped_chunk:,}/{len(chunk):,}",
            file=sys.stderr,
        )

    return out


def group_by_cluster(
    rows: list[dict], acc_to_cluster: dict[str, str]
) -> dict[str, list[str]]:
    """Group UniProt accessions under their UniRef50 cluster id."""
    cluster_to_accs: dict[str, list[str]] = {}
    for r in rows:
        acc = r.get("Entry") or r.get("accession") or ""
        cid = acc_to_cluster.get(acc)
        if not cid:
            continue
        cluster_to_accs.setdefault(cid, []).append(acc)
    return cluster_to_accs


FASTA_HEADER_RE = re.compile(r"^>(\S+)\s*(.*)$")
def parse_fasta(text: str) -> Iterable[tuple[str, str, str]]:
    """Yield (id, description, sequence) from a FASTA blob."""
    cur_id: str | None = None
    cur_desc = ""
    cur_seq: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if cur_id is not None:
                yield cur_id, cur_desc, "".join(cur_seq)
            m = FASTA_HEADER_RE.match(line)
            cur_id = m.group(1) if m else line[1:].split()[0]
            cur_desc = m.group(2) if m else ""
            cur_seq = []
        else:
            cur_seq.append(line.strip())
    if cur_id is not None:
        yield cur_id, cur_desc, "".join(cur_seq)

def fetch_uniref50_fasta(
    session: requests.Session,
    cluster_ids: list[str],
    batch_size: int = 100,
) -> dict[str, tuple[str, str]]:
    """Fetch UniRef50 cluster representative sequences in FASTA, in batches.
    Returns {cluster_id: (description, sequence)}.
    """
    out: dict[str, tuple[str, str]] = {}
    print(
        f"[3/3] Fetching {len(cluster_ids):,} UniRef50 cluster sequences ...",
        file=sys.stderr,
    )
    bar = tqdm(total=len(cluster_ids), unit="cluster", desc="  fetched")
    for i in range(0, len(cluster_ids), batch_size):
        batch = cluster_ids[i : i + batch_size]
        query = " OR ".join(f"id:{cid}" for cid in batch)
        params = {
            "query": query,
            "format": "fasta",
            "compressed": "true",
            "size": "500",
        }
        url = f"{UNIPROT_BASE}/uniref/stream?{urlencode(params)}"
        with session.get(url, stream=True, timeout=(30, 120)) as r:
            r.raise_for_status()
            chunks = []
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    chunks.append(chunk)
            raw = b"".join(chunks)
        text = gzip.decompress(raw).decode("utf-8") if raw[:2] == b"\x1f\x8b" else raw.decode("utf-8")
        for fid, desc, seq in parse_fasta(text):
            out[fid] = (desc, seq)
        bar.update(len(batch))
        time.sleep(0.1)  # be polite to the API
    bar.close()
    print(f"      retrieved {len(out):,} sequences", file=sys.stderr)
    return out

def _load_rows_from_tsv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            rows.append(dict(zip(header, parts)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--interpro-id",
        default=INTERPRO_ID_DEFAULT,
        help="InterPro accession to filter on (default: IPR023346, Lysozyme-like).",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("./lysozyme_uniref50"))
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument(
        "--max-len",
        type=int,
        default=2000,
        help="Drop UniRef50 cluster reps longer than this (default 2000).",
    )
    ap.add_argument(
        "--reviewed-only",
        action="store_true",
        help="Restrict to SwissProt-reviewed UniProt entries before dedup.",
    )
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Ignore on-disk checkpoints and re-run all stages.",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    session = make_session()

    raw_tsv = args.out_dir / "lysozyme_interpro_hits.tsv"
    mapping_json = args.out_dir / "uniprot_to_uniref50.json"
    seqs_json = args.out_dir / "uniref50_seqs.json"

    # ------------------------------------------------------------------
    # Stage 1: UniProt hits (checkpoint -> raw_tsv)
    # ------------------------------------------------------------------
    if raw_tsv.exists() and not args.force:
        print(f"[1/3] resuming from existing {raw_tsv}", file=sys.stderr)
        rows = _load_rows_from_tsv(raw_tsv)
        print(f"      loaded {len(rows):,} entries", file=sys.stderr)
    else:
        rows = fetch_uniprot_hits(session, args.interpro_id)
        if args.reviewed_only:
            rows = [
                r for r in rows if (r.get("Reviewed") or "").lower() == "reviewed"
            ]
            print(
                f"      after reviewed-only filter: {len(rows):,}", file=sys.stderr
            )
        if rows:
            keys = list(rows[0].keys())
            with raw_tsv.open("w") as f:
                f.write("\t".join(keys) + "\n")
                for r in rows:
                    f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")
            print(f"      wrote checkpoint: {raw_tsv}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Stage 2: UniProt -> UniRef50 (checkpoint -> mapping_json)
    # ------------------------------------------------------------------
    if mapping_json.exists() and not args.force:
        print(f"[2/3] resuming from existing {mapping_json}", file=sys.stderr)
        acc_to_cluster: dict[str, str] = _load_json(mapping_json)
        print(f"      loaded {len(acc_to_cluster):,} mappings", file=sys.stderr)
    else:
        print(f"[2/3] mapping UniProt accessions -> UniRef50 ...", file=sys.stderr)
        accs = [r.get("Entry") or r.get("accession") or "" for r in rows]
        accs = [a for a in accs if a]
        acc_to_cluster = map_uniprot_to_uniref50(session, accs)
        _save_json(mapping_json, acc_to_cluster)
        print(f"      wrote checkpoint: {mapping_json}", file=sys.stderr)

    cluster_to_accs = group_by_cluster(rows, acc_to_cluster)
    n_unmapped = len(rows) - sum(len(v) for v in cluster_to_accs.values())
    print(
        f"      unique UniRef50 clusters: {len(cluster_to_accs):,} "
        f"({n_unmapped:,} accessions unmapped)",
        file=sys.stderr,
    )
    cluster_ids = sorted(cluster_to_accs.keys())

    # ------------------------------------------------------------------
    # Stage 3: UniRef50 sequences (checkpoint -> seqs_json)
    # ------------------------------------------------------------------
    if seqs_json.exists() and not args.force:
        print(f"[3/3] resuming from existing {seqs_json}", file=sys.stderr)
        raw = _load_json(seqs_json)
        seqs = {cid: (desc, seq) for cid, (desc, seq) in raw.items()}
        print(f"      loaded {len(seqs):,} sequences", file=sys.stderr)
    else:
        seqs = fetch_uniref50_fasta(
            session, cluster_ids, batch_size=args.batch_size
        )
        _save_json(seqs_json, {cid: list(v) for cid, v in seqs.items()})
        print(f"      wrote checkpoint: {seqs_json}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Stage 4: write final FASTA + metadata with length filter
    # ------------------------------------------------------------------
    fasta_out = args.out_dir / "lysozyme_uniref50.fasta"
    meta_out = args.out_dir / "lysozyme_uniref50.tsv"
    n_kept = n_missing = n_too_short = n_too_long = 0
    with fasta_out.open("w") as ff, meta_out.open("w") as mf:
        mf.write(
            "cluster_id\tlength\tdescription\tn_members\tmember_uniprot_accs\n"
        )
        for cid in cluster_ids:
            if cid not in seqs:
                n_missing += 1
                continue
            desc, seq = seqs[cid]
            L = len(seq)
            if L < args.min_len:
                n_too_short += 1
                continue
            if L > args.max_len:
                n_too_long += 1
                continue
            members = cluster_to_accs.get(cid, [])
            ff.write(f">{cid} {desc}\n{seq}\n")
            mf.write(
                f"{cid}\t{L}\t{desc}\t{len(members)}\t{','.join(members)}\n"
            )
            n_kept += 1

    print(
        f"[done] wrote {n_kept:,} clusters to {fasta_out}\n"
        f"       dropped: {n_missing:,} missing seq, "
        f"{n_too_short:,} below min_len={args.min_len}, "
        f"{n_too_long:,} above max_len={args.max_len}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()