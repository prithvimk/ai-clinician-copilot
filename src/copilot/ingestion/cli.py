from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from copilot.ingestion.indexer import ingest_path
from copilot.rag.vector_store import get_vector_store

app = typer.Typer(help="Clinical knowledge ingestion, search and Q&A.", no_args_is_help=True)
console = Console()


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, help="File or directory of PDF/DOCX files"),
    force: bool = typer.Option(False, help="Re-index even if unchanged"),
) -> None:
    """Load, chunk, embed and index clinical documents."""
    results = ingest_path(path, get_vector_store(), force=force)
    table = Table("File", "Doc ID", "Status", "Chunks")
    for r in results:
        style = {"indexed": "green", "unchanged": "dim", "failed": "red"}[r.status]
        table.add_row(r.source_file, r.doc_id, f"[{style}]{r.status}[/]", str(r.chunks or ""))
    console.print(table)
    for r in (r for r in results if r.status == "failed"):
        console.print(f"[red]{r.source_file}: {r.error}")


@app.command()
def search(
    query: str,
    k: int = typer.Option(5, help="Number of results"),
    mode: str = typer.Option("hybrid", help="dense | sparse | hybrid"),
) -> None:
    """Search indexed chunks."""
    for i, hit in enumerate(get_vector_store().search(query, k=k, mode=mode), 1):
        console.rule(f"[{i}] {hit.chunk.citation()}  score={hit.score:.3f}")
        text = hit.chunk.text
        console.print(text[:600] + ("…" if len(text) > 600 else ""))


@app.command()
def chat() -> None:
    """Interactive cited Q&A over the indexed guidelines (follow-ups use chat history)."""
    from uuid import uuid4

    from copilot.rag.chat import get_chat_service

    service, thread_id = get_chat_service(), uuid4().hex
    console.print("[dim]Ask a question (empty line to quit).[/]")
    while question := console.input("[bold]> [/]").strip():
        answer = service.ask(question, thread_id)
        if answer.standalone_question != question:
            console.print(f"[dim]searched: {answer.standalone_question}[/]")
        console.print(answer.answer)
        for c in answer.citations:
            console.print(f"  [cyan][{c.id}][/] {c.label}  [dim]relevance={c.relevance:.2f}[/]")
        if answer.refusal_reason:
            console.print(f"[yellow]refused: {answer.refusal_reason}[/]")
        console.print()


@app.command()
def stats() -> None:
    """Chunk counts per indexed document."""
    counts = get_vector_store().stats()
    table = Table("Doc ID", "Chunks")
    for doc_id, n in sorted(counts.items()):
        table.add_row(doc_id, str(n))
    console.print(table)
    console.print(f"{len(counts)} documents, {sum(counts.values())} chunks")


if __name__ == "__main__":
    app()
