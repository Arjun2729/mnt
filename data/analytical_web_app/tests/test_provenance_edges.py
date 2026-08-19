"""Every event kind must render as valid Python."""
from groundtruth.provenance import Provenance


def test_clear_empties_the_log():
    log = Provenance()
    log.record("load", "x")
    log.clear()
    assert log.events == [] and log.frame().empty


def test_every_event_kind_compiles():
    log = Provenance()
    log.record("load", "From an API", detail="https://api.test/x", rows=100)
    log.record("load", "From parquet", path="data.parquet", rows=10)
    log.record("filter", "No conditions", where="", params=[])
    log.record("transform", "Added margin", sql="ALTER TABLE t ADD COLUMN m DOUBLE")
    log.record("query", "Counted", sql="SELECT COUNT(*) FROM t")
    log.record("agent_sql", "Agent query", sql="SELECT 1")
    log.record("stat_test", "Correlation", test="Pearson", result="p=0.01")
    log.record("model", "Trained", target="y", features=["a"], best_model="Ridge", metrics={})
    log.record("chart", "Bar chart", spec="Bar x=region y=revenue")
    log.record("question", "Asked something", question="why?", queries=["SELECT 1", "SELECT 2"])
    log.record("scan", "Insight scan surfaced 3 findings")
    log.record("view", "Saved a view")
    compile(log.to_script("dataset"), "<lineage>", "exec")


def test_parquet_load_uses_the_right_reader():
    log = Provenance()
    log.record("load", "Loaded", path="data.parquet", rows=5)
    assert "read_parquet" in log.to_script()


def test_api_load_leaves_a_placeholder():
    log = Provenance()
    log.record("load", "Loaded", detail="https://api.test/x", rows=5)
    script = log.to_script()
    assert "https://api.test/x" in script
    compile(script, "<lineage>", "exec")


def test_unfiltered_step_selects_everything():
    log = Provenance()
    log.record("filter", "Cleared", where="", params=[])
    assert "SELECT * FROM dataset" in log.to_script("dataset")


def test_unknown_kind_becomes_a_comment():
    log = Provenance()
    log.record("something_new", "A future event kind")
    script = log.to_script()
    assert "# A future event kind" in script
    compile(script, "<lineage>", "exec")


def test_question_replays_its_queries():
    log = Provenance()
    log.record("question", "Asked", question="q", queries=["SELECT 1"])
    assert "SELECT 1" in log.to_script()
