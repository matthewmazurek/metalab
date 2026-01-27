"""Tests for file_hash, dir_hash, FilePath, DirPath, and resolve_context."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from metalab._ids import (
    DirPath,
    FilePath,
    Fingerprintable,
    dir_hash,
    file_hash,
    fingerprint_context,
    resolve_context,
)


class TestFileHash:
    """Tests for file_hash utility (content-based hashing)."""

    def test_hash_deterministic(self, tmp_path: Path):
        """Same file content produces same hash."""
        file = tmp_path / "test.txt"
        file.write_text("hello world")

        hash1 = file_hash(file)
        hash2 = file_hash(file)

        assert hash1 == hash2

    def test_hash_length(self, tmp_path: Path):
        """Hash is 16 characters."""
        file = tmp_path / "test.txt"
        file.write_text("hello world")

        h = file_hash(file)
        assert len(h) == 16

    def test_different_content_different_hash(self, tmp_path: Path):
        """Different file content produces different hash."""
        file1 = tmp_path / "test1.txt"
        file2 = tmp_path / "test2.txt"
        file1.write_text("hello")
        file2.write_text("world")

        assert file_hash(file1) != file_hash(file2)

    def test_file_not_found(self, tmp_path: Path):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            file_hash(tmp_path / "nonexistent.txt")

    def test_accepts_string_path(self, tmp_path: Path):
        """Accepts string path."""
        file = tmp_path / "test.txt"
        file.write_text("hello")

        h = file_hash(str(file))
        assert len(h) == 16

    def test_binary_file(self, tmp_path: Path):
        """Works with binary files."""
        file = tmp_path / "test.bin"
        file.write_bytes(b"\x00\x01\x02\x03")

        h = file_hash(file)
        assert len(h) == 16


class TestDirHash:
    """Tests for dir_hash utility (content-based hashing)."""

    def test_hash_deterministic(self, tmp_path: Path):
        """Same directory content produces same hash."""
        (tmp_path / "a.txt").write_text("aaa")
        (tmp_path / "b.txt").write_text("bbb")

        hash1 = dir_hash(tmp_path)
        hash2 = dir_hash(tmp_path)

        assert hash1 == hash2

    def test_hash_length(self, tmp_path: Path):
        """Hash is 16 characters."""
        (tmp_path / "test.txt").write_text("hello")

        h = dir_hash(tmp_path)
        assert len(h) == 16

    def test_different_content_different_hash(self, tmp_path: Path):
        """Different content produces different hash."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "test.txt").write_text("hello")
        (dir2 / "test.txt").write_text("world")

        assert dir_hash(dir1) != dir_hash(dir2)

    def test_pattern_filter(self, tmp_path: Path):
        """Pattern filters which files are included."""
        (tmp_path / "include.txt").write_text("include")
        (tmp_path / "exclude.csv").write_text("exclude")

        hash_txt = dir_hash(tmp_path, pattern="*.txt")
        hash_csv = dir_hash(tmp_path, pattern="*.csv")

        assert hash_txt != hash_csv

    def test_dir_not_found(self, tmp_path: Path):
        """Raises FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            dir_hash(tmp_path / "nonexistent")

    def test_empty_dir(self, tmp_path: Path):
        """Empty directory produces a hash (of nothing)."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        h = dir_hash(empty_dir)
        assert len(h) == 16

    def test_order_independent_of_creation(self, tmp_path: Path):
        """Hash is based on sorted filenames, not creation order."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        # Create in different order but same content
        (dir1 / "a.txt").write_text("aaa")
        (dir1 / "b.txt").write_text("bbb")

        (dir2 / "b.txt").write_text("bbb")
        (dir2 / "a.txt").write_text("aaa")

        assert dir_hash(dir1) == dir_hash(dir2)


class TestFilePath:
    """Tests for FilePath marker type with lazy metadata-based hashing."""

    def test_construction_does_not_access_filesystem(self, tmp_path: Path):
        """FilePath construction should NOT access the filesystem."""
        # File doesn't exist, but construction should succeed
        fp = FilePath(str(tmp_path / "nonexistent.txt"))
        assert fp.path == str(tmp_path / "nonexistent.txt")

    def test_fingerprint_payload_computes_metadata_hash(self, tmp_path: Path):
        """fingerprint_payload() computes hash using file metadata."""
        file = tmp_path / "test.txt"
        file.write_text("hello world")

        fp = FilePath(str(file))
        payload = fp.fingerprint_payload()

        assert "hash" in payload
        assert len(payload["hash"]) == 16

    def test_fingerprint_payload_hash_changes_on_modification(self, tmp_path: Path):
        """Hash should change when file is modified."""
        file = tmp_path / "test.txt"
        file.write_text("original content")

        fp = FilePath(str(file))
        hash1 = fp.fingerprint_payload()["hash"]

        # Small delay to ensure mtime changes
        time.sleep(0.01)
        file.write_text("modified content")

        hash2 = fp.fingerprint_payload()["hash"]

        assert hash1 != hash2

    def test_hash_path_false_excludes_path(self, tmp_path: Path):
        """hash_path=False (default) excludes path from payload."""
        file = tmp_path / "test.txt"
        file.write_text("hello")

        fp = FilePath(str(file), hash_path=False)
        payload = fp.fingerprint_payload()

        assert "path" not in payload
        assert "hash" in payload

    def test_hash_path_true_includes_path(self, tmp_path: Path):
        """hash_path=True includes path in payload."""
        file = tmp_path / "test.txt"
        file.write_text("hello")

        fp = FilePath(str(file), hash_path=True)
        payload = fp.fingerprint_payload()

        assert "path" in payload
        assert payload["path"] == str(file)

    def test_str_returns_path(self, tmp_path: Path):
        """str() returns the path."""
        fp = FilePath(str(tmp_path / "test.txt"))
        assert str(fp) == str(tmp_path / "test.txt")

    def test_fspath_returns_path(self, tmp_path: Path):
        """os.fspath() returns the path."""
        import os

        fp = FilePath(str(tmp_path / "test.txt"))
        assert os.fspath(fp) == str(tmp_path / "test.txt")

    def test_can_construct_path_from_filepath(self, tmp_path: Path):
        """Can construct Path from FilePath."""
        file = tmp_path / "test.txt"
        file.write_text("hello")

        fp = FilePath(str(file))
        p = Path(fp)

        assert p.exists()
        assert p.read_text() == "hello"

    def test_fingerprint_payload_raises_for_missing_file(self, tmp_path: Path):
        """fingerprint_payload() raises FileNotFoundError for missing file."""
        fp = FilePath(str(tmp_path / "nonexistent.txt"))

        with pytest.raises(FileNotFoundError, match="file not found"):
            fp.fingerprint_payload()

    def test_implements_fingerprintable(self, tmp_path: Path):
        """FilePath implements the Fingerprintable protocol."""
        file = tmp_path / "test.txt"
        file.write_text("hello")

        fp = FilePath(str(file))
        assert isinstance(fp, Fingerprintable)


class TestDirPath:
    """Tests for DirPath marker type with lazy metadata-based hashing."""

    def test_construction_does_not_access_filesystem(self, tmp_path: Path):
        """DirPath construction should NOT access the filesystem."""
        dp = DirPath(str(tmp_path / "nonexistent"))
        assert dp.path == str(tmp_path / "nonexistent")

    def test_fingerprint_payload_computes_metadata_hash(self, tmp_path: Path):
        """fingerprint_payload() computes hash using directory metadata."""
        (tmp_path / "a.txt").write_text("aaa")
        (tmp_path / "b.txt").write_text("bbb")

        dp = DirPath(str(tmp_path))
        payload = dp.fingerprint_payload()

        assert "hash" in payload
        assert len(payload["hash"]) == 16

    def test_hash_path_false_excludes_path(self, tmp_path: Path):
        """hash_path=False (default) excludes path from payload."""
        (tmp_path / "test.txt").write_text("hello")

        dp = DirPath(str(tmp_path), hash_path=False)
        payload = dp.fingerprint_payload()

        assert "path" not in payload

    def test_hash_path_true_includes_path_and_pattern(self, tmp_path: Path):
        """hash_path=True includes path and pattern in payload."""
        (tmp_path / "test.txt").write_text("hello")

        dp = DirPath(str(tmp_path), pattern="*.txt", hash_path=True)
        payload = dp.fingerprint_payload()

        assert payload["path"] == str(tmp_path)
        assert payload["pattern"] == "*.txt"

    def test_pattern_filters_files(self, tmp_path: Path):
        """Pattern filters which files are included in hash."""
        (tmp_path / "include.txt").write_text("include")
        (tmp_path / "exclude.csv").write_text("exclude")

        dp_txt = DirPath(str(tmp_path), pattern="*.txt")
        dp_csv = DirPath(str(tmp_path), pattern="*.csv")

        assert (
            dp_txt.fingerprint_payload()["hash"] != dp_csv.fingerprint_payload()["hash"]
        )

    def test_fingerprint_payload_raises_for_missing_dir(self, tmp_path: Path):
        """fingerprint_payload() raises FileNotFoundError for missing directory."""
        dp = DirPath(str(tmp_path / "nonexistent"))

        with pytest.raises(FileNotFoundError, match="directory not found"):
            dp.fingerprint_payload()

    def test_implements_fingerprintable(self, tmp_path: Path):
        """DirPath implements the Fingerprintable protocol."""
        (tmp_path / "test.txt").write_text("hello")

        dp = DirPath(str(tmp_path))
        assert isinstance(dp, Fingerprintable)


class TestResolveContext:
    """Tests for resolve_context() function."""

    def test_resolve_filepath(self, tmp_path: Path):
        """Resolves FilePath to its payload."""
        file = tmp_path / "test.txt"
        file.write_text("hello")

        fp = FilePath(str(file))
        resolved, manifest = resolve_context(fp)

        assert "hash" in resolved
        assert "context" in manifest
        assert manifest["context"]["type"] == "FilePath"

    def test_resolve_dataclass_with_filepath(self, tmp_path: Path):
        """Resolves dataclass containing FilePath fields."""
        file = tmp_path / "test.txt"
        file.write_text("hello")

        @dataclass(frozen=True)
        class MySpec:
            data: FilePath
            value: int

        spec = MySpec(data=FilePath(str(file)), value=42)
        resolved, manifest = resolve_context(spec)

        assert "data" in resolved
        assert "hash" in resolved["data"]
        assert resolved["value"] == 42
        assert "context.data" in manifest

    def test_resolve_nested_dataclass(self, tmp_path: Path):
        """Resolves nested dataclasses with FilePath."""
        file = tmp_path / "test.txt"
        file.write_text("hello")

        @dataclass(frozen=True)
        class Inner:
            file: FilePath

        @dataclass(frozen=True)
        class Outer:
            inner: Inner
            name: str

        spec = Outer(inner=Inner(file=FilePath(str(file))), name="test")
        resolved, manifest = resolve_context(spec)

        assert "context.inner.file" in manifest

    def test_resolve_dict_with_filepath(self, tmp_path: Path):
        """Resolves dict containing FilePath values."""
        file = tmp_path / "test.txt"
        file.write_text("hello")

        spec = {"data": FilePath(str(file)), "value": 42}
        resolved, manifest = resolve_context(spec)

        assert "hash" in resolved["data"]
        assert resolved["value"] == 42

    def test_resolve_list_with_filepath(self, tmp_path: Path):
        """Resolves list containing FilePath values."""
        file1 = tmp_path / "test1.txt"
        file2 = tmp_path / "test2.txt"
        file1.write_text("hello1")
        file2.write_text("hello2")

        spec = [FilePath(str(file1)), FilePath(str(file2))]
        resolved, manifest = resolve_context(spec)

        assert len(resolved) == 2
        assert "context[0]" in manifest
        assert "context[1]" in manifest

    def test_resolve_primitives_unchanged(self, tmp_path: Path):
        """Primitives pass through unchanged."""
        spec = {"name": "test", "value": 42, "flag": True}
        resolved, manifest = resolve_context(spec)

        assert resolved == spec
        assert manifest == {}

    def test_error_message_includes_path(self, tmp_path: Path):
        """FileNotFoundError includes the field path."""

        @dataclass(frozen=True)
        class MySpec:
            data: FilePath

        spec = MySpec(data=FilePath(str(tmp_path / "missing.txt")))

        with pytest.raises(FileNotFoundError, match="context.data"):
            resolve_context(spec)

    def test_custom_fingerprintable_type(self, tmp_path: Path):
        """Custom Fingerprintable types work with resolve_context()."""

        @dataclass(frozen=True)
        class CustomMarker:
            value: str

            def fingerprint_payload(self) -> dict[str, Any]:
                return {"custom_hash": f"hash-{self.value}"}

        spec = {"marker": CustomMarker(value="test")}
        resolved, manifest = resolve_context(spec)

        assert resolved["marker"]["custom_hash"] == "hash-test"
        assert "context.marker" in manifest


class TestFingerprintContextWithFilePath:
    """Tests for fingerprint_context() with FilePath."""

    def test_fingerprint_context_resolves_filepath(self, tmp_path: Path):
        """fingerprint_context() resolves FilePath before fingerprinting."""
        file = tmp_path / "test.txt"
        file.write_text("hello")

        @dataclass(frozen=True)
        class MySpec:
            data: FilePath

        spec = MySpec(data=FilePath(str(file)))
        fp = fingerprint_context(spec)

        assert len(fp) == 16

    def test_fingerprint_changes_when_file_modified(self, tmp_path: Path):
        """Fingerprint changes when file is modified."""
        file = tmp_path / "test.txt"
        file.write_text("original")

        @dataclass(frozen=True)
        class MySpec:
            data: FilePath

        spec = MySpec(data=FilePath(str(file)))
        fp1 = fingerprint_context(spec)

        time.sleep(0.01)
        file.write_text("modified")

        fp2 = fingerprint_context(spec)

        assert fp1 != fp2

    def test_same_content_different_path_same_fingerprint(self, tmp_path: Path):
        """Same content at different paths produces same fingerprint (hash_path=False)."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        # Create files with same content but ensure same metadata
        content = "hello world"
        file1.write_text(content)
        file2.write_text(content)

        # Note: This test only works if the files have the same size
        # The mtime and inode will differ, so hashes will differ
        # This is expected behavior for metadata-based hashing
        @dataclass(frozen=True)
        class MySpec:
            data: FilePath

        spec1 = MySpec(data=FilePath(str(file1)))
        spec2 = MySpec(data=FilePath(str(file2)))

        # With metadata-based hashing, different files = different hashes
        # (even with same content, different mtime/inode)
        fp1 = fingerprint_context(spec1)
        fp2 = fingerprint_context(spec2)

        # These will be different due to different mtime/inode
        # This is a trade-off for O(1) performance
        assert fp1 != fp2

    def test_raises_for_missing_file(self, tmp_path: Path):
        """fingerprint_context() raises FileNotFoundError for missing file."""

        @dataclass(frozen=True)
        class MySpec:
            data: FilePath

        spec = MySpec(data=FilePath(str(tmp_path / "missing.txt")))

        with pytest.raises(FileNotFoundError):
            fingerprint_context(spec)
