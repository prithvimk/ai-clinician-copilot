"""Create a Qdrant snapshot of the guideline collection and keep only the newest N.

Snapshots land on the Qdrant snapshots volume; copy them off-cluster (object-storage sync or
Qdrant's S3 snapshot storage) for disaster recovery. The index can always be rebuilt from the
versioned corpus with the ingest job, so snapshots shorten recovery rather than replace it.

    python scripts/qdrant_snapshot.py --keep 7
    python scripts/qdrant_snapshot.py --restore <snapshot-name>   # restore in place
"""

import argparse
import sys

from qdrant_client import QdrantClient

from copilot.config import get_settings


def to_delete(snapshots, keep: int) -> list[str]:
    """Names of all but the `keep` newest snapshots (by creation time)."""
    ordered = sorted(snapshots, key=lambda s: s.creation_time or "", reverse=True)
    return [s.name for s in ordered[keep:]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", type=int, default=7)
    parser.add_argument("--restore", metavar="NAME", help="recover the collection from a snapshot")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not settings.qdrant_url:
        print("QDRANT_URL is not set (snapshots need a Qdrant server)", file=sys.stderr)
        return 2
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    collection = settings.qdrant_collection

    if args.restore:
        location = f"file:///qdrant/snapshots/{collection}/{args.restore}"
        client.recover_snapshot(collection, location=location, wait=True)
        print(f"restored {collection} from {args.restore}")
        return 0

    created = client.create_snapshot(collection, wait=True)
    print(f"created {created.name} ({created.size} bytes)")
    for name in to_delete(client.list_snapshots(collection), args.keep):
        client.delete_snapshot(collection, name, wait=True)
        print(f"deleted {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
