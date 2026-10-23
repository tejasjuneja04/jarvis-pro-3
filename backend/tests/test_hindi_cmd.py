import pytest

import hindi_cmd as HC
import skills

CASES = [
    ("पाँच मिनट का टाइमर लगाओ", "5 minutes timer"),
    ("दिल्ली का मौसम कैसा है", "weather in Delhi"),
    ("bacchon ko tab remind karo", None),
]


@pytest.mark.parametrize("raw,expect", CASES)
def test_normalize(raw, expect):
    out = HC.normalize(raw)
    if expect:
        assert expect.lower() in out.lower()


def test_plain_english_is_untouched():
    for t in ("set a 10 minute timer for the pasta", "what is the weather in Paris", "give me two examples"):
        assert HC.normalize(t) == t


def test_timer_reaches_the_real_parser():
    out = HC.normalize("दस मिनट का टाइमर लगाओ")
    assert skills.parse_timer(out) is not None


def test_loanwords():
    assert "forecast" in HC.normalize("सेल्स का फोरकास्ट दिखाओ").lower()
