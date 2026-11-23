"""End-to-end UI test: render the whole dashboard headlessly with Streamlit's AppTest."""
import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

from f1cc.config import CIRCUITS

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _app() -> AppTest:
    os.environ["F1CC_OFFLINE"] = "1"
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def _by_label(elements, label):
    return next(e for e in elements if e.label == label)


def test_dashboard_renders_without_errors():
    at = _app()
    assert not at.exception, at.exception
    assert len(at.tabs) == 6
    assert any("Optimal strategy" in m.label for m in at.metric)


def test_switching_circuit_and_drivers():
    at = _app()
    _by_label(at.toggle, "Persist run to SQLite warehouse").set_value(False)
    _by_label(at.selectbox, "Circuit").select_index(list(CIRCUITS).index("monaco")).run()
    assert not at.exception, at.exception
    assert "Monaco" in at.markdown[1].value or any("Monaco" in m.value for m in at.markdown)

    _by_label(at.selectbox, "Driver A").select_index(0)
    _by_label(at.selectbox, "Driver B (reference)").select_index(0).run()
    assert not at.exception
    assert any("two different drivers" in i.value for i in at.info)


def test_undercut_controls_update_verdict():
    at = _app()
    _by_label(at.number_input, "Gap to car ahead (s)").set_value(0.0).run()
    assert any("UNDERCUT WORKS" in s.value for s in at.success)
    _by_label(at.number_input, "Gap to car ahead (s)").set_value(10.0).run()
    assert any("falls short" in e.value for e in at.error)
