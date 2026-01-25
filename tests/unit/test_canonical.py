"""Tests for canonicalization and fingerprinting."""

from __future__ import annotations

import pytest

from metalab._canonical import CanonicalizeError, canonical, fingerprint


class TestCanonical:
    """Tests for canonical() function."""

    def test_canonical_dict_ordering(self):
        """Dict keys should be sorted regardless of insertion order."""
        d1 = {"b": 1, "a": 2, "c": 3}
        d2 = {"a": 2, "c": 3, "b": 1}
        assert canonical(d1) == canonical(d2)

    def test_canonical_nested_dict(self):
        """Nested dicts should also have sorted keys."""
        d = {"z": {"b": 1, "a": 2}, "y": 3}
        result = canonical(d)
        # Keys should appear in sorted order (y before z alphabetically)
        assert result.index('"y"') < result.index('"z"')
        assert '"a"' in result
        assert '"b"' in result

    def test_canonical_list_preserves_order(self):
        """List order should be preserved."""
        lst = [3, 1, 2]
        result = canonical(lst)
        assert result == "[3,1,2]"

    def test_canonical_set_becomes_sorted_list(self):
        """Sets should become sorted lists."""
        s = {3, 1, 2}
        result = canonical(s)
        assert result == "[1,2,3]"

    def test_canonical_float_full_precision(self):
        """Floats should use repr() for full precision."""
        # These are different values at full precision
        f1 = 1.0000000000000001
        f2 = 1.0000000000000002

        c1 = canonical({"x": f1})
        c2 = canonical({"x": f2})

        # They should produce different canonical strings
        # (unless Python collapses them to the same float)
        if f1 != f2:
            assert c1 != c2

    def test_canonical_nan_raises(self):
        """NaN should raise CanonicalizeError."""
        import math

        with pytest.raises(CanonicalizeError, match="NaN"):
            canonical({"x": float("nan")})

    def test_canonical_inf_raises(self):
        """Inf should raise CanonicalizeError."""
        with pytest.raises(CanonicalizeError, match="Inf"):
            canonical({"x": float("inf")})

        with pytest.raises(CanonicalizeError, match="Inf"):
            canonical({"x": float("-inf")})

    def test_canonical_none(self):
        """None should be preserved."""
        assert canonical(None) == "null"
        assert canonical({"x": None}) == '{"x":null}'

    def test_canonical_bool(self):
        """Booleans should be preserved."""
        assert canonical(True) == "true"
        assert canonical(False) == "false"

    def test_canonical_bytes(self):
        """Bytes should be encoded as hex."""
        result = canonical(b"\x00\xff")
        assert "__bytes__" in result
        assert "00ff" in result

    def test_canonical_dataclass(self):
        """Dataclasses should be converted to dicts."""
        from dataclasses import dataclass

        @dataclass
        class Point:
            x: int
            y: int

        p = Point(x=1, y=2)
        result = canonical(p)
        assert '"x":1' in result
        assert '"y":2' in result


class TestFingerprint:
    """Tests for fingerprint() function."""

    def test_fingerprint_deterministic(self):
        """Same input should produce same fingerprint."""
        d = {"a": 1, "b": 2}
        assert fingerprint(d) == fingerprint(d)
        assert fingerprint(d) == fingerprint({"b": 2, "a": 1})

    def test_fingerprint_length(self):
        """Fingerprint should be 16 hex characters."""
        fp = fingerprint({"test": "data"})
        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)

    def test_fingerprint_different_values(self):
        """Different values should produce different fingerprints."""
        fp1 = fingerprint({"a": 1})
        fp2 = fingerprint({"a": 2})
        assert fp1 != fp2

    def test_fingerprint_empty(self):
        """Empty objects should fingerprint successfully."""
        fp1 = fingerprint({})
        fp2 = fingerprint([])
        fp3 = fingerprint("")
        assert len(fp1) == 16
        assert len(fp2) == 16
        assert len(fp3) == 16
        # They should all be different
        assert fp1 != fp2 != fp3
