"""Run every Streamlit page headlessly and assert it renders without exceptions."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from config import DB_PATH

ROOT = Path(__file__).resolve().parents[1]
PAGES = [ROOT / "app" / "Home.py", *sorted((ROOT / "app" / "pages").glob("*.py"))]


@pytest.mark.parametrize("page", PAGES, ids=[p.stem for p in PAGES])
def test_page_renders(page: Path) -> None:
    if not DB_PATH.exists():
        pytest.skip("warehouse not built")
    at = AppTest.from_file(str(page), default_timeout=120).run()
    assert not at.exception, at.exception[0].value if at.exception else None
    assert at.title, "page has no title"


def test_class_switch_on_overview() -> None:
    if not DB_PATH.exists():
        pytest.skip("warehouse not built")
    overview = ROOT / "app" / "pages" / "0_Overview.py"
    at = AppTest.from_file(str(overview), default_timeout=120).run()
    at.sidebar.radio[0].set_value("yellow").run()
    assert not at.exception
    assert any("YMAZ" in m.label for m in at.metric)


def test_home_blotter_lists_every_market() -> None:
    """The monitor must render one row per configured market, with no unhandled exception."""
    if not DB_PATH.exists():
        pytest.skip("warehouse not built")
    from app.data.blotter import MARKETS

    at = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=120).run()
    assert not at.exception, at.exception[0].value if at.exception else None
    html = " ".join(m.body for m in at.markdown)
    for _, _, symbol, _, _ in MARKETS:
        assert symbol in html, f"missing market row: {symbol}"
