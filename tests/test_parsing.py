import pytest
from gabriel.utils.parsing import safe_json


def test_safe_json_with_comments():
    raw = (
        '[\n'
        '  {\n'
        '    "name": "Example",\n'
        '    "links": [\n'
        '      "https://example.com", // comment\n'
        '      "s3://bucket/file"  // another comment\n'
        '    ]\n'
        '  }\n'
        ']'
    )
    parsed = safe_json(raw)
    assert isinstance(parsed, list)
    assert parsed[0]["links"][0] == "https://example.com"

