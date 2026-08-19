"""Every operator, and the rendering helpers."""
import pytest

from groundtruth.filters import Condition, Group, compile_tree, describe, describe_inline


def test_unknown_operator_is_refused():
    with pytest.raises(ValueError, match="Unknown operator"):
        compile_tree(Condition("a", "≈", 1))


@pytest.mark.parametrize("operator,sql_fragment", [
    ("=", '"a" = ?'),
    ("!=", '"a" IS DISTINCT FROM ?'),
    (">", '"a" > ?'),
    (">=", '"a" >= ?'),
    ("<", '"a" < ?'),
    ("<=", '"a" <= ?'),
])
def test_comparison_operators(operator, sql_fragment):
    where, params = compile_tree(Condition("a", operator, 5))
    assert where == sql_fragment and params == [5]


@pytest.mark.parametrize("operator", ["is null", "is not null"])
def test_null_operators_bind_nothing(operator):
    where, params = compile_tree(Condition("a", operator))
    assert params == []
    assert "IS" in where


def test_between_binds_both_bounds():
    where, params = compile_tree(Condition("a", "between", [1, 9]))
    assert "BETWEEN ? AND ?" in where and params == [1, 9]


@pytest.mark.parametrize("operator,expected", [
    ("contains", "%x%"),
    ("starts with", "x%"),
    ("ends with", "%x"),
])
def test_text_operators_build_the_right_pattern(operator, expected):
    where, params = compile_tree(Condition("a", operator, "x"))
    assert "ILIKE" in where and params == [expected]


def test_not_in_negates():
    where, params = compile_tree(Condition("a", "not in", [1, 2]))
    assert "NOT IN" in where and params == [1, 2]


def test_groups_holding_only_empty_children_vanish():
    tree = Group("AND", [Group("AND", []), Group("OR", [])])
    assert compile_tree(tree) == ("", [])


def test_describe_renders_a_tree():
    text = describe(Group("AND", [Condition("a", "is null"), Condition("b", ">", 2)]))
    assert "a is null" in text and "b > 2" in text


def test_describe_of_an_empty_group():
    assert describe(Group("AND", [])) == "(no filters)"


def test_describe_inline_null_operator():
    assert describe_inline(Condition("a", "is not null")) == "a is not null"


def test_describe_inline_between():
    assert describe_inline(Condition("a", "between", [1, 5])) == "a 1–5"


def test_describe_inline_negation():
    assert describe_inline(Group("AND", [Condition("a", "=", 1)], negate=True)) == "NOT (a = 1)"
