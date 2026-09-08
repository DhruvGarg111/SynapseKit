from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile

import pytest

from synapsekit.rag.kv_cache_store import CacheKey, KVCacheStore


@pytest.fixture
def cache_dir() -> str:
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_kv_cache_store_roundtrip(cache_dir: str) -> None:
    store = KVCacheStore(cache_dir=cache_dir)
    key = CacheKey("fingerprint123", "model-a", 2048)
    blob = b"dummy kv cache bytes"
    meta = {"custom_field": "val"}

    store.save(key, blob, meta)
    loaded = store.load(key)

    assert loaded is not None
    loaded_blob, loaded_meta = loaded
    assert loaded_blob == blob
    assert loaded_meta["custom_field"] == "val"
    assert loaded_meta["corpus_fingerprint"] == "fingerprint123"
    assert loaded_meta["model_id"] == "model-a"
    assert loaded_meta["n_ctx"] == 2048
    assert "created_at" in loaded_meta


def test_kv_cache_store_missing_returns_none(cache_dir: str) -> None:
    store = KVCacheStore(cache_dir=cache_dir)
    key = CacheKey("missing", "model-a", 2048)
    assert store.load(key) is None


def test_kv_cache_store_metadata_mismatch_returns_none(cache_dir: str) -> None:
    store = KVCacheStore(cache_dir=cache_dir)
    key = CacheKey("fingerprint", "model-a", 2048)
    blob = b"dummy bytes"
    meta = {}

    store.save(key, blob, meta)

    # 1. Fingerprint mismatch
    assert store.load(CacheKey("other_fingerprint", "model-a", 2048)) is None
    # 2. Model ID mismatch
    assert store.load(CacheKey("fingerprint", "other-model", 2048)) is None
    # 3. Context size mismatch
    assert store.load(CacheKey("fingerprint", "model-a", 4096)) is None


def test_kv_cache_store_corrupted_json_returns_none(cache_dir: str) -> None:
    store = KVCacheStore(cache_dir=cache_dir)
    key = CacheKey("fingerprint", "model-a", 2048)
    blob = b"dummy bytes"
    meta = {}

    store.save(key, blob, meta)

    # Find and corrupt the json file
    base_path = store._get_filename_base(key)
    json_path = f"{base_path}.json"

    with open(json_path, "w") as f:
        f.write("{invalid json content}")

    assert store.load(key) is None


def test_kv_cache_store_version_mismatch_returns_none(cache_dir: str) -> None:
    store = KVCacheStore(cache_dir=cache_dir)
    key = CacheKey("fingerprint", "model-a", 2048)
    blob = b"dummy bytes"
    meta = {}

    store.save(key, blob, meta)

    # Manipulate JSON file version
    base_path = store._get_filename_base(key)
    json_path = f"{base_path}.json"

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    data["cache_format_version"] = 999  # Incompatible future version

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    assert store.load(key) is None


# --- Security hardening (#1024) -------------------------------------------


def test_tampered_blob_with_valid_metadata_is_rejected(cache_dir: str) -> None:
    # An attacker who can write the cache dir also writes the .json, so the
    # integrity metadata is trivially satisfiable. The HMAC (keyed by a secret
    # not in the files) must still reject a swapped .bin so it never reaches the
    # native llama.cpp deserializer.
    store = KVCacheStore(cache_dir=cache_dir)
    key = CacheKey("fp", "model-a", 2048)
    store.save(key, b"legit kv state", {})

    base_path = store._get_filename_base(key)
    with open(f"{base_path}.bin", "wb") as f:
        f.write(b"malicious payload of a different length")

    # Metadata is untouched and still "valid", but the HMAC no longer matches.
    assert store.load(key) is None


def test_forged_metadata_hmac_is_rejected(cache_dir: str) -> None:
    # Recomputing the hmac field with a wrong key (as an attacker without the
    # secret would) must not validate.
    store = KVCacheStore(cache_dir=cache_dir)
    key = CacheKey("fp", "model-a", 2048)
    store.save(key, b"legit", {})

    base_path = store._get_filename_base(key)
    with open(f"{base_path}.bin", "wb") as f:
        f.write(b"evil")
    with open(f"{base_path}.json", encoding="utf-8") as f:
        data = json.load(f)
    import hashlib
    import hmac as _hmac

    data["hmac"] = _hmac.new(b"attacker-guessed-key", b"evil", hashlib.sha256).hexdigest()
    with open(f"{base_path}.json", "w", encoding="utf-8") as f:
        json.dump(data, f)

    assert store.load(key) is None


def test_missing_hmac_field_is_rejected(cache_dir: str) -> None:
    store = KVCacheStore(cache_dir=cache_dir)
    key = CacheKey("fp", "model-a", 2048)
    store.save(key, b"legit", {})

    base_path = store._get_filename_base(key)
    with open(f"{base_path}.json", encoding="utf-8") as f:
        data = json.load(f)
    data.pop("hmac", None)
    with open(f"{base_path}.json", "w", encoding="utf-8") as f:
        json.dump(data, f)

    assert store.load(key) is None


def test_explicit_hmac_key_roundtrip_and_wrong_key_misses(cache_dir: str) -> None:
    key = CacheKey("fp", "model-a", 2048)
    KVCacheStore(cache_dir=cache_dir, hmac_key=b"key-one").save(key, b"state", {})

    # Same key: hit.
    assert KVCacheStore(cache_dir=cache_dir, hmac_key=b"key-one").load(key) is not None
    # Different key: the blob won't authenticate -> miss (no crash).
    assert KVCacheStore(cache_dir=cache_dir, hmac_key=b"key-two").load(key) is None


def test_namespace_isolates_cache_entries(cache_dir: str) -> None:
    store = KVCacheStore(cache_dir=cache_dir)
    a = CacheKey("fp", "model-a", 2048, namespace="tenant-a")
    b = CacheKey("fp", "model-a", 2048, namespace="tenant-b")
    store.save(a, b"a-state", {})

    assert store.load(a) is not None
    # Identical corpus+model, different namespace -> distinct file, miss.
    assert store.load(b) is None


def test_default_namespace_is_backward_compatible(cache_dir: str) -> None:
    store = KVCacheStore(cache_dir=cache_dir)
    key = CacheKey("fp", "model-a", 2048)  # no namespace arg
    store.save(key, b"state", {})
    loaded = store.load(key)
    assert loaded is not None
    assert loaded[1]["namespace"] == ""


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_cache_dir_is_private(cache_dir: str) -> None:
    KVCacheStore(cache_dir=cache_dir)
    mode = stat.S_IMODE(os.stat(cache_dir).st_mode)
    assert mode == 0o700
    key_path = os.path.join(cache_dir, KVCacheStore._KEY_FILENAME)
    assert stat.S_IMODE(os.stat(key_path).st_mode) == 0o600
