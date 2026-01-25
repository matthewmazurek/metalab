"""
Serializers for artifact capture.

Available serializers:
- JSONSerializer: JSON (stdlib) - DEFAULT, always available
- PickleSerializer: Pickle - OPT-IN ONLY
- NumpySerializer: NumPy arrays (optional, requires numpy)
"""

from metalab.capture.serializers.json_ import JSONSerializer

__all__ = ["JSONSerializer"]

# Conditionally export other serializers
try:
    from metalab.capture.serializers.numpy_ import NumpySerializer

    __all__.append("NumpySerializer")
except ImportError:
    pass

# Pickle is always available but not exported by default
# Users must explicitly import it
