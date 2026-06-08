from auxiliary_agents import classify_error


def test_api_key_error():
    r = classify_error(Exception("unauthorized 401 api_key invalid"))
    assert r["error_type"] == "api_key"
    assert r["recoverable"] is True

def test_authentication_error():
    r = classify_error(Exception("Authentication failed"))
    assert r["error_type"] == "api_key"

def test_timeout_error():
    r = classify_error(Exception("Connection timed out after 30s"))
    assert r["error_type"] == "timeout"
    assert r["recoverable"] is True

def test_connection_refused():
    r = classify_error(Exception("connection refused"))
    assert r["error_type"] == "timeout"

def test_rate_limit_error():
    r = classify_error(Exception("rate limit exceeded, 429 too many requests"))
    assert r["error_type"] == "rate_limit"
    assert r["recoverable"] is True

def test_model_not_found():
    r = classify_error(Exception("model not found: invalid model name"))
    assert r["error_type"] == "model"
    assert r["recoverable"] is False

def test_unknown_error():
    r = classify_error(Exception("something completely unexpected"))
    assert r["error_type"] == "unknown"
    assert r["recoverable"] is False

def test_raw_contains_exception_text():
    r = classify_error(ValueError("test message"))
    assert "test message" in r["raw"]
    assert "ValueError" in r["raw"]
