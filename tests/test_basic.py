"""Basic tests to verify the setup is working."""

from wat_mmt import __version__


def test_version():
    """Test that the version is defined."""
    assert __version__ == "0.1.0"


def test_import():
    """Test that the package can be imported."""
    import wat_mmt  # noqa: F401
