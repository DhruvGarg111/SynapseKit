from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import secrets
import tempfile
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CacheKey:
    corpus_fingerprint: str
    model_id: str
    n_ctx: int
    # Optional isolation namespace (e.g. a tenant id). Two callers with an
    # identical corpus + model but different namespaces get distinct cache
    # files. Defaults to "" so existing callers are unaffected.
    namespace: str = ""


class KVCacheStore:
    """Persistent KV Cache Store for Cache-Augmented Generation (CAG).

    Writes cache blobs and metadata atomically using a temporary file and
    ``os.replace``.

    Security: the persisted ``.bin`` is a raw llama.cpp state buffer that is fed
    straight into a native C++ deserializer (``LlamaState``/``load_state``). An
    attacker who can write into the cache directory could otherwise plant a
    malformed buffer and reach that native parser. To prevent this, every blob
    is authenticated with an HMAC-SHA256 keyed by a per-install secret; on load,
    a missing/invalid HMAC is treated as a **cache miss** (returns ``None``),
    so a tampered or foreign blob never reaches ``load_state`` — the router
    simply rebuilds or falls back to RAG. The cache directory is created ``0o700``
    and the auto-generated key file ``0o600``. Pass ``hmac_key`` to supply the
    key out-of-band (recommended when the directory is shared storage).
    """

    CACHE_FORMAT_VERSION = 2
    _KEY_FILENAME = ".hmac_key"

    def __init__(
        self, cache_dir: str = ".synapsekit_cag_cache", hmac_key: bytes | None = None
    ) -> None:
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, mode=0o700, exist_ok=True)
        # makedirs won't tighten a pre-existing directory; do it explicitly so
        # the derived corpus material in here isn't world-readable.
        with contextlib.suppress(OSError):
            os.chmod(cache_dir, 0o700)
        self._hmac_key = hmac_key if hmac_key is not None else self._load_or_create_key()

    def _load_or_create_key(self) -> bytes:
        """Return the per-install HMAC key, creating a 0600 key file if needed."""
        key_path = os.path.join(self._cache_dir, self._KEY_FILENAME)
        try:
            with open(key_path, "rb") as f:
                existing = f.read()
            if existing:
                return existing
        except FileNotFoundError:
            pass
        key = secrets.token_bytes(32)
        # O_EXCL so two concurrent creators don't race; if we lose the race,
        # read back the winner's key.
        try:
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(fd, key)
            finally:
                os.close(fd)
            return key
        except FileExistsError:
            with open(key_path, "rb") as f:
                return f.read()

    def _sign(self, blob: bytes) -> str:
        return hmac.new(self._hmac_key, blob, hashlib.sha256).hexdigest()

    def _get_filename_base(self, key: CacheKey) -> str:
        # Use a safe hash of the key tuple for the filenames
        h = hashlib.sha256(
            f"{key.namespace}:{key.corpus_fingerprint}:{key.model_id}:{key.n_ctx}".encode()
        ).hexdigest()
        return os.path.join(self._cache_dir, h)

    def save(self, key: CacheKey, blob: bytes, metadata: dict[str, Any]) -> None:
        """Atomically save the cache blob and its associated metadata to disk."""
        base_path = self._get_filename_base(key)
        bin_path = f"{base_path}.bin"
        json_path = f"{base_path}.json"

        # Build full metadata dict. The HMAC authenticates the blob so a
        # tampered/foreign .bin is rejected on load before it reaches the
        # native llama.cpp state parser.
        full_meta = {
            **metadata,
            "corpus_fingerprint": key.corpus_fingerprint,
            "model_id": key.model_id,
            "n_ctx": key.n_ctx,
            "namespace": key.namespace,
            "cache_format_version": self.CACHE_FORMAT_VERSION,
            "created_at": time.time(),
            "hmac": self._sign(blob),
        }

        # Write binary blob atomically
        bin_dir = os.path.dirname(bin_path)
        with tempfile.NamedTemporaryFile(dir=bin_dir, delete=False, suffix=".tmp") as f:
            f.write(blob)
            temp_bin = f.name

        try:
            os.replace(temp_bin, bin_path)
        except Exception:
            if os.path.exists(temp_bin):
                os.remove(temp_bin)
            raise

        # Write metadata JSON atomically
        json_content = json.dumps(full_meta, indent=2).encode("utf-8")
        with tempfile.NamedTemporaryFile(dir=bin_dir, delete=False, suffix=".tmp") as f:
            f.write(json_content)
            temp_json = f.name

        try:
            os.replace(temp_json, json_path)
        except Exception:
            if os.path.exists(temp_json):
                os.remove(temp_json)
            raise

    def load(self, key: CacheKey) -> tuple[bytes, dict[str, Any]] | None:
        """Load and validate the cache blob and metadata.

        Returns None on any corruption, mismatch, or missing files.
        """
        base_path = self._get_filename_base(key)
        bin_path = f"{base_path}.bin"
        json_path = f"{base_path}.json"

        if not os.path.exists(bin_path) or not os.path.exists(json_path):
            return None

        try:
            # Load and validate metadata first
            with open(json_path, encoding="utf-8") as f:
                meta = json.load(f)

            if meta.get("corpus_fingerprint") != key.corpus_fingerprint:
                return None
            if meta.get("model_id") != key.model_id:
                return None
            if meta.get("n_ctx") != key.n_ctx:
                return None
            if meta.get("namespace", "") != key.namespace:
                return None
            if meta.get("cache_format_version") != self.CACHE_FORMAT_VERSION:
                return None

            # Read binary blob
            with open(bin_path, "rb") as f:
                blob = f.read()

            # Authenticate the blob. The metadata is attacker-writable too, so
            # the integrity fields above are not a trust boundary — the HMAC
            # (keyed by a secret not derivable from the files) is. A missing or
            # non-matching HMAC is treated as a cache miss so the tampered blob
            # never reaches the native state deserializer.
            expected = meta.get("hmac")
            if not isinstance(expected, str) or not hmac.compare_digest(
                expected, self._sign(blob)
            ):
                return None

            return blob, meta
        except Exception:
            # Treat any file error or JSON corruption as a cache invalidation
            return None
