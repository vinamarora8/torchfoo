import torchfoo


class TestVersion:
    def test_version_is_string(self):
        assert isinstance(torchfoo.__version__, str)

    def test_version_is_not_unknown(self):
        assert torchfoo.__version__ != "unknown"

    def test_version_format(self):
        parts = torchfoo.__version__.split(".")
        assert len(parts) >= 2
        assert parts[0].isdigit() and parts[1].isdigit()
