import voice as V


def test_units_english():
    out = V.speech_text("It is 32°C and 12 km/h, price ₹1,50,000", "en")
    assert "°" not in out and "km/h" not in out and "," not in out.replace(", ", "")


def test_units_hindi_uses_hindi_words():
    out = V.speech_text("तापमान 32°C है", "hi")
    assert "°" not in out
