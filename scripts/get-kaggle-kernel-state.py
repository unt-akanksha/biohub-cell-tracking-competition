from __future__ import annotations

import argparse
import json
from typing import Any


class KernelStateError(ValueError):
    pass


def exact_owned_kernel_refs(api: Any, kernel_slug: str) -> list[str]:
    """Resolve a 403 on an absent private kernel without guessing state."""

    owner, slug = kernel_slug.split("/", maxsplit=1)
    rows = api.kernels_list(search=slug, mine=True, page_size=100) or []
    refs = sorted(
        {
            str(getattr(row, "ref", ""))
            for row in rows
            if row is not None and str(getattr(row, "ref", "")) == f"{owner}/{slug}"
        }
    )
    if len(refs) > 1:  # pragma: no cover - a Kaggle owner/slug is unique
        raise KernelStateError("owned kernel inventory returned duplicate exact refs")
    return refs


def kernel_state_from_sdk_response(
    kernel_slug: str, response: Any | None
) -> dict[str, Any]:
    """Reduce an authenticated SDK response to compact, secret-free state."""
    parts = kernel_slug.split("/")
    if len(parts) != 2 or any(not part.strip() for part in parts):
        raise KernelStateError("expected exact owner/slug")
    if response is None:
        return {
            "schema_version": "biohub.kaggle-kernel-state.v1",
            "kernel_slug": kernel_slug,
            "present": False,
            "current_version_number": None,
            "next_version_number": 1,
            "next_kernel_ref": f"{kernel_slug}/1",
        }
    metadata = getattr(response, "metadata", None)
    if metadata is None or str(getattr(metadata, "ref", "")) != kernel_slug:
        raise KernelStateError("SDK response ref does not match exact owner/slug")
    try:
        current_version = int(getattr(metadata, "current_version_number"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise KernelStateError("invalid current_version_number") from exc
    if current_version < 1:
        raise KernelStateError("current version must be positive")
    next_version = current_version + 1
    return {
        "schema_version": "biohub.kaggle-kernel-state.v1",
        "kernel_slug": kernel_slug,
        "present": True,
        "current_version_number": current_version,
        "next_version_number": next_version,
        "next_kernel_ref": f"{kernel_slug}/{next_version}",
        "is_private": getattr(metadata, "is_private", None),
        "enable_gpu": getattr(metadata, "enable_gpu", None),
        "enable_tpu": getattr(metadata, "enable_tpu", None),
        "enable_internet": getattr(metadata, "enable_internet", None),
        "language": getattr(metadata, "language", None),
        "kernel_type": getattr(metadata, "kernel_type", None),
        "dataset_sources": list(
            getattr(metadata, "dataset_data_sources", None) or []
        ),
        "competition_sources": list(
            getattr(metadata, "competition_data_sources", None) or []
        ),
        "kernel_sources": list(
            getattr(metadata, "kernel_data_sources", None) or []
        ),
        "model_sources": list(getattr(metadata, "model_data_sources", None) or []),
    }


def inspect_owned_kernel_state(kernel_slug: str) -> dict[str, Any]:
    """Read an exact owned kernel using the authenticated Kaggle Python SDK."""
    try:
        from kaggle.api.kaggle_api_extended import (  # type: ignore[import-not-found]
            ApiGetKernelRequest,
            KaggleApi,
        )
        from requests.exceptions import HTTPError
    except ImportError as exc:  # pragma: no cover - operator environment
        raise KernelStateError("Kaggle Python SDK unavailable") from exc

    api = KaggleApi()
    try:
        api.authenticate()
        owner, slug, version = api.parse_kernel_string(kernel_slug)
        if version is not None or f"{owner}/{slug}" != kernel_slug:
            raise KernelStateError("expected exact owner/slug")
        request = ApiGetKernelRequest()
        request.user_name = owner
        request.kernel_slug = slug
        with api.build_kaggle_client() as client:
            try:
                response = client.kernels.kernels_api_client.get_kernel(request)
            except HTTPError as exc:
                status_code = getattr(
                    getattr(exc, "response", None), "status_code", None
                )
                if status_code == 404:
                    response = None
                elif status_code == 403 and not exact_owned_kernel_refs(
                    api, kernel_slug
                ):
                    # Kaggle returns 403 rather than 404 for some absent private
                    # kernels.  An authenticated, owner-scoped exact listing is
                    # the only accepted absence proof; a present ref still
                    # fails closed because its version cannot be inferred.
                    response = None
                else:
                    raise KernelStateError(
                        f"SDK get_kernel HTTP {status_code or 'unknown'}"
                    ) from exc
    except KernelStateError:
        raise
    except Exception as exc:
        raise KernelStateError(type(exc).__name__) from exc
    return kernel_state_from_sdk_response(kernel_slug, response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-slug", required=True)
    args = parser.parse_args()
    try:
        state = inspect_owned_kernel_state(args.kernel_slug)
    except KernelStateError as exc:
        parser.exit(2, f"kernel state lookup failed closed: {exc}\n")
    print(json.dumps(state, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
