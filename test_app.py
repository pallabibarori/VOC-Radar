"""Runs the Streamlit app headlessly on the bundled sample data."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_demo_flow():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    at.toggle[0].set_value(True).run()                       # sample data on
    at.button[0].click().run()                                # Gather feedback
    assert not at.exception, at.exception
    assert any("Voice of customer" in s.value for s in at.subheader)
    gen = next(b for b in at.button if b.label == "Generate ICPs")
    gen.click().run()
    assert not at.exception, at.exception
    names = [m.value for m in at.markdown if m.value.startswith("### ")]
    assert len(names) == 5, names


def test_example_button_prefills_query(monkeypatch):
    from voc import collect
    monkeypatch.setattr(collect, "collect", lambda *a, **k: collect.CollectionResult(query="Figma", reviews=[]))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    next(b for b in at.button if b.label == "Figma").click().run()
    assert not at.exception, at.exception
    assert at.text_input[0].value == "Figma"
