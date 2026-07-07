"""corpus — the open-book context builder: ordering, cap, skip-reporting, PDF text, retrieval."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus  # noqa: E402


def _fake_building(tmp_path: Path) -> Path:
    b = tmp_path / "bldg"
    b.mkdir()
    (b / "model.ttl").write_text("ex:AHU1 a s223:AirHandlingUnit .")
    (b / "notes.md").write_text("# notes\nsupply air setpoint is 18 C")
    (b / "trend.csv").write_text("t,value\n1,17.9\n2,18.1")
    (b / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return b


def test_text_files_in_density_order_and_binaries_reported(tmp_path):
    b = _fake_building(tmp_path)
    text, skipped = corpus.building_corpus(b)
    assert text.index("model.ttl") < text.index("trend.csv") < text.index("notes.md")
    assert "AHU1" in text and "18.1" in text
    assert skipped == ["photo.png"]              # never silently dropped


def test_cap_is_respected(tmp_path):
    b = _fake_building(tmp_path)
    text, _ = corpus.building_corpus(b, max_chars=50)
    assert len(text) == 50


def test_pdf_text_layer_extracted(tmp_path):
    fitz = pytest.importorskip("fitz")           # pymupdf
    b = _fake_building(tmp_path)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Control card: GT41 frost guard limit 2 C")
    doc.save(b / "card.pdf")
    doc.close()
    text, skipped = corpus.building_corpus(b)
    assert "frost guard limit 2 C" in text
    assert "card.pdf" not in skipped


def test_imageonly_pdf_is_skipped_not_silent(tmp_path):
    fitz = pytest.importorskip("fitz")
    b = _fake_building(tmp_path)
    doc = fitz.open()
    doc.new_page()                               # blank page, no text layer
    doc.save(b / "scan.pdf")
    doc.close()
    _, skipped = corpus.building_corpus(b)
    assert "scan.pdf" in skipped


def test_retrieve_picks_relevant_chunk():
    text = "\n\n".join([f"chunk about topic {i}" for i in range(50)]
                       + ["the GT41_sensor frost guard threshold is 2 C"])
    out = corpus.retrieve(text, 'What is the "GT41_sensor" threshold?', max_chars=200)
    assert "GT41_sensor" in out and len(out) <= 200
