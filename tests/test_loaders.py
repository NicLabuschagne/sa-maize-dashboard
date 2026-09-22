from app.data.loaders import load_demo


def test_load_demo_shape() -> None:
    df = load_demo(n=10)
    assert list(df.columns) == ["date", "price", "ret"]
    assert len(df) == 10
