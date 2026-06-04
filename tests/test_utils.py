import pytest

from youtube_spam_detection.utils import clean_comment


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Check out my channel!!!", "check out my channel"),
        ("Free at https://example.com now", "free at urltoken now"),
        (None, ""),
    ],
)
def test_clean_comment(raw, expected):
    assert clean_comment(raw) == expected
