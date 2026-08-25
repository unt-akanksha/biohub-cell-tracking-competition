from __future__ import annotations

import argparse
import json
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi


def inventory_dataset(
    dataset_ref: str, *, api: Any | None = None, page_size: int = 1000
) -> dict[str, Any]:
    client = api or KaggleApi()
    if api is None:
        client.authenticate()
    files: list[dict[str, Any]] = []
    token = None
    seen_tokens: set[str] = set()
    pages = 0
    while True:
        response = client.dataset_list_files(
            dataset_ref, page_token=token, page_size=page_size
        )
        pages += 1
        if pages > 10_000:
            raise RuntimeError("dataset inventory pagination exceeded safe bound")
        for item in response.dataset_files or ():
            files.append(
                {"name": str(item.name), "total_bytes": int(item.total_bytes)}
            )
        next_token = getattr(response, "next_page_token", None)
        if not next_token:
            break
        if next_token in seen_tokens:
            raise RuntimeError("dataset inventory pagination token repeated")
        seen_tokens.add(next_token)
        token = next_token
    files.sort(key=lambda item: item["name"])
    if len({item["name"].casefold() for item in files}) != len(files):
        raise RuntimeError("dataset inventory contains duplicate paths")
    return {
        "dataset_ref": dataset_ref,
        "file_count": len(files),
        "total_bytes": sum(item["total_bytes"] for item in files),
        "pages": pages,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-ref", required=True)
    parser.add_argument("--page-size", type=int, default=1000)
    args = parser.parse_args()
    print(
        json.dumps(
            inventory_dataset(args.dataset_ref, page_size=args.page_size),
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
