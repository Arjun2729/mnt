"""Store paths that only appear when names collide or objects change kind."""
import pandas as pd
import pytest

from groundtruth.store import Store, safe_table_name


@pytest.fixture
def store():
    s = Store()
    s.register_frame("t", pd.DataFrame({"a": [1, 2, 3], "b": list("xyz")}))
    yield s
    s.close()


def test_table_names_are_sanitised():
    assert safe_table_name("Q3 sales!") == "q3_sales"
    assert safe_table_name("2024 report").startswith("t_")   # cannot begin with a digit
    assert safe_table_name("") == "dataset"
    assert len(safe_table_name("x" * 200)) <= 60


def test_filtered_view_switches_between_view_and_table(store):
    """Unfiltered is a view; filtered must materialise. Swapping kinds must work."""
    assert store.count(store.create_filtered_view("v", "t")) == 3
    assert store.count(store.create_filtered_view("v", "t", '"a" > ?', [1])) == 2
    assert store.count(store.create_filtered_view("v", "t")) == 3          # table -> view
    assert store.count(store.create_filtered_view("v", "t", '"a" > ?', [2])) == 1


def test_dropping_removes_either_kind(store):
    store.create_filtered_view("v", "t")
    store._drop_any("v")
    store.create_filtered_view("v", "t", '"a" > ?', [1])
    store._drop_any("v")
    assert store.con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'v'"
    ).fetchone()[0] == 0


def test_dropping_something_absent_is_harmless(store):
    store._drop_any("never_existed")


def test_drop_deregisters_the_dataset(store):
    store.drop("t")
    assert "t" not in store.datasets


def test_dropping_an_unknown_dataset_is_harmless(store):
    store.drop("nope")


def test_ordered_paging(store):
    page = store.page("t", order_by='"a" DESC', limit=2)
    assert list(page["a"]) == [3, 2]
