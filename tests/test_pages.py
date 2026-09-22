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
    for _, _, _, symbol, _, _ in MARKETS:
        assert symbol in html, f"missing market row: {symbol}"
    for group in {m[0] for m in MARKETS}:
        assert group in html, f"missing group band: {group}"


def test_home_replays_point_in_time_and_shades_signals() -> None:
    """At the 2016 drought the board must show shaded SELL signals, not the neutral present."""
    if not DB_PATH.exists():
        pytest.skip("warehouse not built")
    import re

    import pandas as pd

    at = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=180).run()
    target = next(o for o in at.selectbox[0].options
                  if pd.Timestamp(o).strftime("%Y-%m") == "2016-07")
    at.selectbox[0].set_value(target).run()
    assert not at.exception

    html = " ".join(m.body for m in at.markdown)
    assert html.count(">SELL<") >= 3, "expected several rich signals during the drought"
    alphas = {float(a) for r, g, b, a in re.findall(r"rgba\((\d+),(\d+),(\d+),([\d.]+)\)", html)
              if (r, g, b) == ("255", "92", "92") and float(a) < 0.55}
    assert len(alphas) > 1, "shading should vary with severity, not be a single flat tone"

    caption = " ".join(c.value for c in at.caption)
    assert "replayed at release" in caption and "26 Jul 2016" in caption


def test_home_table_headers_match_body_columns() -> None:
    """Regression: a silent no-op string replace once left the Flow column unheaded."""
    src = (ROOT / "app" / "Home.py").read_text(encoding="utf-8")
    header = src[src.index('html = ['):src.index('current_group')]
    body = src[src.index('current_group'):src.index('html.append("</table>")')]
    n_head = header.count("<th")
    n_body = body.count("<td") - 1          # the group band row contributes one spanning cell
    assert n_head == n_body, f"{n_head} headers vs {n_body} body cells"
    assert f"colspan='{n_head}'" in body, "group band must span every column"


def test_home_shows_a_flow_column() -> None:
    if not DB_PATH.exists():
        pytest.skip("warehouse not built")
    at = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=180).run()
    assert not at.exception
    html = " ".join(m.body for m in at.markdown)
    assert "Flow 1w" in html
    assert "pp" in html, "flow values should be quoted in position points"
