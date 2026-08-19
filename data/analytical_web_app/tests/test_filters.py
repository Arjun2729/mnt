"""Filter trees must compile to parameterised SQL with real nesting."""
from groundtruth.filters import Condition, Group, compile_tree, describe, from_dict, to_dict


def test_single_condition():
    where, params = compile_tree(Condition("revenue", ">", 100))
    assert where == '"revenue" > ?'
    assert params == [100]


def test_nested_and_or():
    tree = Group("AND", [
        Condition("region", "in", ["North", "East"]),
        Group("OR", [Condition("revenue", ">", 150000), Condition("channel", "=", "Paid")]),
    ])
    where, params = compile_tree(tree)
    assert "IN (?, ?)" in where
    assert " OR " in where and " AND " in where
    assert params == ["North", "East", 150000, "Paid"]


def test_empty_tree_produces_no_clause():
    assert compile_tree(Group("AND", [])) == ("", [])
    assert compile_tree(None) == ("", [])


def test_empty_in_matches_nothing():
    assert compile_tree(Condition("region", "in", [])) == ("FALSE", [])
    assert compile_tree(Condition("region", "not in", [])) == ("TRUE", [])


def test_negation_wraps_the_group():
    where, _ = compile_tree(Group("AND", [Condition("cost", ">", 1)], negate=True))
    assert where.startswith("NOT (")


def test_contains_is_parameterised_not_interpolated():
    where, params = compile_tree(Condition("region", "contains", "'; DROP TABLE t --"))
    assert "DROP" not in where
    assert params == ["%'; DROP TABLE t --%"]


def test_quotes_in_column_names_are_escaped():
    where, _ = compile_tree(Condition('we"ird', "=", 1))
    assert where == '"we""ird" = ?'


def test_roundtrip_serialisation():
    tree = Group("OR", [Condition("a", "=", 1), Group("AND", [Condition("b", "<", 2)])])
    assert to_dict(from_dict(to_dict(tree))) == to_dict(tree)


def test_describe_is_readable():
    text = describe(Group("AND", [Condition("a", "=", 1)]))
    assert "AND" in text and "a = 1" in text


def test_filters_actually_reduce_rows(store):
    tree = Group("AND", [
        Condition("region", "in", ["North", "East"]),
        Group("OR", [Condition("revenue", ">", 150000), Condition("channel", "=", "Paid")]),
    ])
    where, params = compile_tree(tree)
    assert 0 < store.count("sample", where, params) < store.count("sample")


# ---------------- one-line rendering for the header ----------------


def test_inline_description_unwraps_single_child_groups():
    from groundtruth.filters import describe_inline

    tree = Group("AND", [Group("AND", [Condition("region", "in", ["East"])])])
    assert describe_inline(tree) == "region in East"


def test_inline_description_keeps_real_structure():
    from groundtruth.filters import describe_inline

    tree = Group("AND", [
        Group("AND", [Condition("region", "in", ["North", "East"])]),
        Group("OR", [Condition("revenue", ">", 150000), Condition("channel", "=", "Paid")]),
    ])
    text = describe_inline(tree)
    assert " AND " in text and " OR " in text
    assert text.count("AND") == 1  # the redundant inner AND is dropped


def test_inline_description_truncates_long_value_lists():
    from groundtruth.filters import describe_inline

    text = describe_inline(Condition("region", "in", ["a", "b", "c", "d", "e", "f"]))
    assert "+2" in text


def test_inline_description_of_empty_tree():
    from groundtruth.filters import describe_inline

    assert describe_inline(Group("AND", [])) == "no filters"
    assert describe_inline(None) == "no filters"
