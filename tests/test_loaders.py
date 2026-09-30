from app.data.loaders import load_demo


def test_load_demo_shape() -> None:
    df = load_demo(n=10)
    assert list(df.columns) == ["date", "price", "ret"]
    assert len(df) == 10


def test_constant_maturity_blends_across_the_roll() -> None:
    """Contracts at 60 and 120 days bracket the 90-day tenor: log-linear halfway = geometric mean.
    When every contract is beyond the tenor, the nearest one is used."""
    import numpy as np
    import pandas as pd

    from app.data import features as F

    rows = [("2020-01-02", "2020-03-02", 60, 100.0), ("2020-01-02", "2020-05-01", 120, 400.0),
            ("2020-01-03", "2020-05-01", 95, 200.0), ("2020-01-03", "2020-07-01", 150, 300.0)]
    p = pd.DataFrame(rows, columns=["trade_date", "expiry_date", "days_to_expiry", "close"])
    p["trade_date"], p["expiry_date"], p["symbol"] = pd.to_datetime(p.trade_date), pd.to_datetime(p.expiry_date), "YMAZ"
    cm = F.constant_maturity(p).set_index("trade_date")["close_cm"]
    assert np.isclose(cm.iloc[0], 200.0)
    assert np.isclose(cm.iloc[1], 200.0)
