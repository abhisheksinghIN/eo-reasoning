from analysis.temporal import summarize_indicator


def test_declining_indicator():
    result = summarize_indicator([0.7, 0.6, 0.5])
    assert result["absolute_change"] < 0
    assert result["slope"] < 0
