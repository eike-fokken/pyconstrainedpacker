from pyconstrainedpacker import ArticleShipment, PackingGroup


def test_simple_run() -> None:
    ArticleShipment([250, 500], [8, 20])
    PackingGroup("first", 4000, 0.8, 2)
