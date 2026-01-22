from app import _extract_json_object


def test_extract_json_object_plain():
    s = '{"a":1,"b":2}'
    assert _extract_json_object(s) == s

def test_extract_json_object_fenced():
    s = "```json\n{ \"x\": 1 }\n```"
    assert _extract_json_object(s).strip().startswith("{")

def test_extract_json_object_embedded():
    s = "Here is JSON:\n{ \"k\": \"v\" }\nThanks"
    j = _extract_json_object(s)
    assert '"k"' in j
