"""Download the public clinical guideline corpus listed in data/guidelines/sources.yaml."""

import sys
from pathlib import Path

import httpx
import yaml

GUIDELINES_DIR = Path(__file__).resolve().parents[1] / "data" / "guidelines"
WHO_API = "https://iris.who.int/server/api"
HEADERS = {"User-Agent": "Mozilla/5.0 (ai-clinician-copilot corpus fetch)"}


def who_pdf_url(client: httpx.Client, handle: str) -> str:
    """Resolve a WHO IRIS handle to the content URL of its primary PDF."""
    item = client.get(f"{WHO_API}/pid/find", params={"id": handle}).json()
    bundles = client.get(item["_links"]["bundles"]["href"]).json()["_embedded"]["bundles"]
    original = next(b for b in bundles if b["name"] == "ORIGINAL")
    streams = client.get(original["_links"]["bitstreams"]["href"]).json()["_embedded"]
    pdf = next(s for s in streams["bitstreams"] if s["name"].lower().endswith(".pdf"))
    return pdf["_links"]["content"]["href"]


def main(force: bool = False) -> int:
    entries = yaml.safe_load((GUIDELINES_DIR / "sources.yaml").read_text(encoding="utf-8"))
    failures = 0
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=120) as client:
        for entry in entries:
            if entry["source"] not in ("who_handle", "url"):
                continue
            target = GUIDELINES_DIR / entry["file"]
            if target.exists() and not force:
                print(f"skip     {entry['file']}")
                continue
            try:
                if entry["source"] == "who_handle":
                    url = who_pdf_url(client, entry["handle"])
                else:
                    url = entry["url"]
                response = client.get(url)
                response.raise_for_status()
                if not response.content.startswith(b"%PDF"):
                    raise ValueError(f"not a PDF ({response.headers.get('content-type')})")
                target.write_bytes(response.content)
                print(f"fetched  {entry['file']} ({len(response.content) // 1024} KB)")
            except Exception as exc:
                failures += 1
                print(f"FAILED   {entry['file']}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(force="--force" in sys.argv))
