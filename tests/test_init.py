import torchfoo


class TestVersion:
    def test_version_is_string():
        assert isinstance(torchfoo.__version__, str)

    def test_version_is_not_unknown():
        assert torchfoo.__version__ != "unknown"

    def test_version_format():
        parts = torchfoo.__version__.split(".")
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts)
