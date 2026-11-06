"""Download and hash all admitted evidence PDFs into research/papers/.

Usage:
    python scripts/download_evidence_pdfs.py

This script is idempotent: it skips files that already exist with a matching
hash.  It records SHA-256, byte size, retrieval timestamp, and the source URL
in a per-paper manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = REPO_ROOT / "research" / "frontier_evidence.json"
PAPERS_ROOT = REPO_ROOT / "research" / "papers"
USER_AGENT = "llm-watermark-lab/0.1 (research; mailto:user@example.com)"
RETRY_LIMIT = 3
RETRY_BACKOFF = 2.0


def _slugify(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[-\s]+", "-", text)
    return text[:max_len].rstrip("-")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> bool:
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=30) as resp:
                if resp.status != 200:
                    print(f"  HTTP {resp.status} on attempt {attempt}")
                    time.sleep(RETRY_BACKOFF * attempt)
                    continue
                dest.write_bytes(resp.read())
            return True
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            print(f"  {type(exc).__name__}: {exc} (attempt {attempt}/{RETRY_LIMIT})")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_BACKOFF * attempt)
    return False


def _verify_pdf_header(path: Path) -> bool:
    if path.stat().st_size < 5:
        return False
    with path.open("rb") as f:
        header = f.read(5)
    return header == b"%PDF-"


def main() -> int:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    records = ledger.get("records", [])
    if not records:
        print("No records in evidence ledger.")
        return 1

    PAPERS_ROOT.mkdir(parents=True, exist_ok=True)
    success_count = 0
    fail_count = 0
    skip_count = 0

    for i, record in enumerate(records, 1):
        title = record.get("title", f"record-{i}")
        pdf_url = record.get("pdf_url", "")
        evidence_id = record.get("evidence_id", f"rec-{i-1}")
        if not pdf_url:
            print(f"[{i:2d}/30] SKIP (no PDF URL): {title[:60]}")
            fail_count += 1
            continue

        slug = _slugify(title)
        paper_dir = PAPERS_ROOT / slug
        paper_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = paper_dir / "paper.pdf"
        manifest_path = paper_dir / "manifest.json"

        existing_manifest: dict[str, object] = {}
        if manifest_path.is_file():
            try:
                existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass

        if pdf_path.is_file() and _verify_pdf_header(pdf_path):
            existing_hash = str(existing_manifest.get("paper_sha256", ""))
            current_hash = _sha256_file(pdf_path)
            if existing_hash == current_hash and existing_hash:
                print(f"[{i:2d}/30] CACHED: {slug}")
                skip_count += 1
                continue

        print(f"[{i:2d}/30] DOWNLOAD: {slug}")
        if not _download(pdf_url, pdf_path):
            print(f"  FAILED: {pdf_url}")
            fail_count += 1
            if pdf_path.exists():
                pdf_path.unlink()
            continue

        if not _verify_pdf_header(pdf_path):
            print(f"  NOT A PDF (header check failed): {pdf_url}")
            fail_count += 1
            if pdf_path.exists():
                pdf_path.unlink()
            continue

        sha256 = _sha256_file(pdf_path)
        size = pdf_path.stat().st_size
        manifest = {
            "evidence_id": evidence_id,
            "title": title,
            "pdf_url": pdf_url,
            "paper_sha256": sha256,
            "paper_size_bytes": size,
            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pdf_header_verified": True,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"  OK: sha256={sha256[:16]}... size={size:,} bytes")
        success_count += 1

    print(f"\nDone: {success_count} downloaded, {skip_count} cached, {fail_count} failed")
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main())