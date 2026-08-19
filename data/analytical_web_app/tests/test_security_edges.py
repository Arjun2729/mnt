"""Security guard paths: resolution failures, odd URLs, unparseable SQL."""
import pytest

from groundtruth import security
from groundtruth.security import SecurityError, assert_read_only, validate_api_url


def test_unresolvable_host_is_refused(monkeypatch):
    import socket

    def boom(*args, **kwargs):
        raise socket.gaierror("nope")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    with pytest.raises(SecurityError, match="does not resolve"):
        validate_api_url("https://nowhere.invalid/x", allowed_hosts=["nowhere.invalid"])


def test_public_address_is_permitted(monkeypatch):
    monkeypatch.setattr(security, "_resolves_to_private_address", lambda host: False)
    assert validate_api_url("https://api.test/x", allowed_hosts=["api.test"])


def test_private_check_can_be_waived(monkeypatch):
    """Explicitly allowed for local development, never by default."""
    assert validate_api_url("http://localhost/x", allowed_hosts=["localhost"], allow_private=True)


def test_private_detection_reads_resolution(monkeypatch):
    import socket

    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    assert security._resolves_to_private_address("example.test") is False
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("10.0.0.5", 0))])
    assert security._resolves_to_private_address("internal.test") is True


@pytest.mark.parametrize("url", ["ws://host/x", "mailto:a@b.c", "//host/x", "just-a-string"])
def test_non_http_schemes_are_refused(url):
    with pytest.raises(SecurityError):
        validate_api_url(url, allowed_hosts=["host"])


def test_url_without_a_host_is_refused():
    with pytest.raises(SecurityError, match="no host"):
        validate_api_url("http:///path", allowed_hosts=["anything"])


def test_unparseable_sql_is_refused():
    with pytest.raises(SecurityError, match="Could not parse|read-only"):
        assert_read_only("SELECT FROM WHERE ((((")


@pytest.mark.parametrize("statement", [
    "CREATE TABLE t (a INT)", "ALTER TABLE t ADD COLUMN b INT",
    "ATTACH 'x.db'", "COPY t TO 'x.csv'", "INSTALL httpfs", "LOAD httpfs",
])
def test_side_effecting_statements_are_refused(statement):
    with pytest.raises(SecurityError):
        assert_read_only(statement)


def test_trailing_semicolon_is_tolerated():
    assert assert_read_only("SELECT 1;") == "SELECT 1"


def test_whitespace_only_is_refused():
    with pytest.raises(SecurityError, match="Empty query"):
        assert_read_only("   \n  ")
