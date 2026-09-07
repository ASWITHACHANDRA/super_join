#!/usr/bin/env python3
"""
Download publicly available sample PDFs for FactLens testing.

The three macroeconomic reports about India:
  1. Economic Survey 2024-25 (Ministry of Finance, GoI)
  2. RBI Annual Report 2024-25
  3. IMF Article IV Consultation — India 2024

These are all public documents available from government / IMF websites.
This script downloads them to the pdfs/ directory.

Usage:
    python download_samples.py
"""
import os
import sys
import urllib.request
from pathlib import Path

SAMPLES = [
    {
        "name": "Economic Survey 2024-25",
        "filename": "economic_survey_2024_25.pdf",
        # Chapter 1 of the Economic Survey (publicly available from indiabudget.gov.in)
        "url": "https://www.indiabudget.gov.in/economicsurvey/doc/echapter.pdf",
    },
    {
        "name": "RBI Annual Report 2023-24",
        "filename": "rbi_annual_report_2023_24.pdf",
        # RBI Annual Report (publicly available from rbi.org.in)
        "url": "https://rbidocs.rbi.org.in/rdocs/AnnualReport/PDFs/0RBIAN2024_F5513D9E3E664AD590C7D35D09C51E53.PDF",
    },
    {
        "name": "IMF Article IV India 2024",
        "filename": "imf_article_iv_india_2024.pdf",
        # IMF Staff Report on India (publicly available)
        "url": "https://www.imf.org/en/Publications/CR/Issues/2024/02/02/India-2023-Article-IV-Consultation-Press-Release-Staff-Report-and-Statement-by-the-544755",
    },
]

def download(url: str, dest: Path, name: str) -> bool:
    if dest.exists():
        print(f"  ✓ Already exists: {dest.name}")
        return True
    
    print(f"  Downloading {name}...")
    try:
        headers = {"User-Agent": "Mozilla/5.0 FactLens-Downloader/1.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r    {pct:.0f}% ({downloaded//1024} KB / {total//1024} KB)", end="", flush=True)
        print(f"\n  ✓ Saved: {dest}")
        return True
    except Exception as e:
        print(f"\n  ✗ Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    pdfs_dir = script_dir / "pdfs"
    pdfs_dir.mkdir(exist_ok=True)

    print("FactLens Sample PDF Downloader")
    print("=" * 40)
    print(f"Saving to: {pdfs_dir.absolute()}")
    print()
    print("NOTE: If downloads fail (network issues, URL changes), manually place")
    print("PDF files in the pdfs/ directory and upload via the UI at http://localhost:8000")
    print()

    successes = 0
    for sample in SAMPLES:
        dest = pdfs_dir / sample["filename"]
        print(f"[{sample['name']}]")
        if download(sample["url"], dest, sample["name"]):
            successes += 1
        print()

    print(f"Downloaded {successes}/{len(SAMPLES)} files.")
    print()
    print("Next steps:")
    print("  1. Start FactLens:  python run.py")
    print("  2. Open browser:    http://localhost:8000")
    print("  3. Upload the PDFs from the pdfs/ directory via the UI")
    print("     (or use the API: POST /documents)")
