"""Turn generated documents into Arrow tables and files."""
from __future__ import annotations

from urllib.parse import urlparse

import pyarrow as pa

from common.webgen import corpus


def docs_table(n: int = 20_000, seed: int = 0) -> pa.Table:
    docs = corpus(n, seed=seed)
    return pa.table({
        "id": [d["id"] for d in docs],
        "url": [d["url"] for d in docs],
        "host": [urlparse(d["url"]).hostname for d in docs],
        "lang": [d["lang"] for d in docs],
        "kind": [d["kind"] for d in docs],
        "n_chars": [len(d["text"]) for d in docs],
        "text": [d["text"] for d in docs],
    })
