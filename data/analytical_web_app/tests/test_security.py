"""The guards that stand between user input and the outside world."""
import pytest

from groundtruth.security import SecurityError, assert_read_only, validate_api_url


@pytest.mark.parametrize("query", ["SELECT 1", "  select * from t  ", "WITH a AS (SELECT 1) SELECT * FROM a"])
def test_reads_are_allowed(query):
    assert assert_read_only(query)


@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE t",
        "DELETE FROM t",
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET a = 1",
        "SELECT 1; DROP TABLE t",
        "WITH a AS (SELECT 1) DELETE FROM t",  # prefix matching would wave this through
        "",
    ],
)
def test_writes_are_blocked(query):
    with pytest.raises(SecurityError):
        assert_read_only(query)


def test_api_is_deny_by_default():
    with pytest.raises(SecurityError, match="allowlist"):
        validate_api_url("https://example.com/data", allowed_hosts=[])


def test_api_rejects_unlisted_host():
    with pytest.raises(SecurityError, match="not in the allowlist"):
        validate_api_url("https://evil.test/data", allowed_hosts=["good.test"])


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host/x", "gopher://host"])
def test_api_rejects_non_http_schemes(url):
    with pytest.raises(SecurityError):
        validate_api_url(url, allowed_hosts=["host"])


def test_api_rejects_private_addresses():
    # Allowlisted, but resolves to loopback — the DNS-rebinding case.
    with pytest.raises(SecurityError, match="private address"):
        validate_api_url("http://localhost:8000/x", allowed_hosts=["localhost"])
