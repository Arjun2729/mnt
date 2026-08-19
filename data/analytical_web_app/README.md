# Groundtruth

An analytical workspace where every number is traceable to the query that produced it.

Load data from a file, a database or an API; explore it with nested filters; chart it;
ask questions in English and get answers computed by SQL rather than estimated; run
statistics that report effect sizes; train models that must beat a baseline; forecast;
export a report that carries its charts; and monitor it with alerts that notify on
transitions instead of on every run.

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then load `sample_data.csv` from the sidebar — 288 rows of monthly revenue across four
regions and three channels, with seasonality, trend and a little missingness, enough to
exercise every tab.

## The six pillars

| Pillar | What it does |
|---|---|
| 📂 **Multi-format & DB/API** | CSV, TSV, Excel, JSON, JSONL, Parquet; SQLAlchemy databases; allowlisted JSON APIs. Everything lands in DuckDB — Parquet is scanned in place. |
| 🔍 **Exploration & filtering** | Column roles inferred at load; nested AND/OR filter groups compiled to parameterised SQL; saved views; derived columns from SQL expressions. |
| 📊 **Visualization & reporting** | Charts suggested from column roles; eight chart types; **click or drag a chart to filter every tab**; pin anything to a report and export it as self-contained HTML with live charts, or as Excel. |
| 🤖 **AI analytics** | A tool-calling agent with read-only SQL, charting and statistics tools, **streaming its answer as it works**. It shows the query behind every answer. Plus a proactive scan that surfaces findings before you ask. |
| 🧠 **Statistics & ML** | Correlation, chi-square, ANOVA/Kruskal, t-test/Mann-Whitney, normality — all with effect sizes, intervals and FDR correction. Models cross-validate against a baseline with permutation importance and leakage detection. |
| 📤 **Export & alerts** | CSV, Excel, HTML report, runnable Python script. Alerts hold state: they fire on transition, report recovery, and honour a cooldown. |

## Interaction

The interface is built to be driven, not just read. Everything below is native
Streamlit — no custom components, nothing that breaks on the next release.

**Click a chart to filter the workspace.** Charts are selection-enabled: click
marks or drag a box, and the selection compiles into filter-tree conditions that
every other tab then respects. Clicking the East bar in "revenue by region" drops
288 rows to 72, and the models, statistics, forecasts and agent all follow.
Categorical marks become `in` conditions; box-drags over numeric axes become
`between` conditions; a coloured series contributes its own condition.

**Distributions live inside the tables.** The column profile renders a real
histogram per measure, category frequencies per dimension, and records-per-period
for time columns, alongside a completeness bar — so the shape of the data is
visible without plotting anything.

**Answers stream.** The analyst reassembles tool calls from streamed deltas, so
each query appears in the status panel as it runs and the final answer arrives
token by token rather than after a long pause.

**Time is scrubbable.** Temporal charts get a range slider and quick-range
buttons; the header carries a per-measure trend strip.

**Panels rerun independently.** The Visualize and Data panels are
`st.fragment`s, so changing a chart type or a page does not re-profile the
dataset or re-run any other tab.

Motion is used where it carries meaning: tiles rise in a short stagger on load,
cards lift under the pointer, and the cross-filter badge pulses while a selection
is driving the rest of the app. All of it respects `prefers-reduced-motion`.

## Architecture

Six layers. Data flows down; nothing above L1 keeps its own copy.

```
L0  connectors.py            files · databases · APIs
L1  store.py                 DuckDB — one engine for every read
L2  semantic.py              column roles · time grain · keys
L3  filters · stats · ml · timeseries
L4  agent.py · insights.py   tools over the filtered view
L5  report · alerts · provenance
```

The point of the layering is that a filter set means the same thing everywhere. The grid,
the charts, the models, the agent and the alerts all read the same filtered view, because
they all read through L1.

## The analyst

The agent is given tools, not a sample of rows:

- `run_sql` — read-only DuckDB SELECT against the filtered view
- `describe_columns` — the semantic layer's schema, no row data
- `make_chart` — renders a Plotly figure from a query
- `run_stat_test` — correlation, chi-square or group comparison, with effect sizes

Ask "what was median revenue for Paid in Q3?" and it computes the exact figure and shows
you the SQL. Every query it runs is displayed beneath the answer and recorded in the
lineage. Errors are returned to the model so it can correct itself rather than failing.

### Rate limits

The analyst spends **one request per tool round**, so a single question costs several
requests against a per-minute quota. On a tight free tier that matters more than the
per-request price.

When a 429 arrives the loop waits out the delay the provider itself specifies and retries,
rather than failing; only an exhausted retry budget surfaces as an error, and that message
names the quota and what to do about it. The round ceiling is capped at 6, and the agent is
instructed to batch lookups into as few queries as it can.

Gemini's free tier meters **per day as well as per minute**, and the daily budget is the
binding one. Measured on a free key: `gemini-3.6-flash` allows 5 requests/minute, and
`gemini-2.5-flash-lite` allows **20 requests per day** — which at one request per tool
round is roughly four to six questions in total. Enough to look around; not enough to work.

**Groq is the default.** Its free tier comfortably carries an agent that spends several
requests per question, and the whole loop is verified against it end to end — a
multi-round question with SQL and a significance test, and a chart-generating question,
both answered correctly through the running UI.

Model ids move under you. `openai/gpt-oss-120b`, `openai/gpt-oss-20b` and
`qwen/qwen3.6-27b` were confirmed to support tool calling on a live free key;
`groq/compound` does not. If one 404s, **List models** asks the endpoint what your key can
reach rather than trusting a name baked into this repo.

The two limits need different handling, so the app distinguishes them: a per-minute ceiling
is waited out and retried, while a daily exhaustion is reported immediately — it reports a
short `retryDelay` too, but waiting cannot help.

Keys are per-provider: pasting a Groq key while "Google Gemini" is selected sends it to
Google and gets a 401. Change the provider first — the endpoint changes with it.

If limits still bite: move to **Groq**, whose free tier is far more generous for an agent
that makes several calls per question, or run **Ollama** locally for no limit at all.

### Choosing a provider

The analyst talks to any OpenAI-compatible chat-completions endpoint, so the provider is a
setting rather than a dependency. **Google Gemini has a free tier and needs no payment
method**, and is the default.

| Provider | Free | Get a key |
|---|---|---|
| Google Gemini | free, but only ~20 requests/day | https://aistudio.google.com/apikey |
| **Groq** (default) | free, far larger budget | https://console.groq.com/keys |
| OpenRouter | models ending `:free` | https://openrouter.ai/keys |
| Ollama | yes, runs locally | https://ollama.com/download |
| OpenAI | no, billing required | https://platform.openai.com/api-keys |

```bash
export LLM_PROVIDER="Google Gemini"
export GEMINI_API_KEY="..."
export LLM_MODEL="gemini-2.5-flash-lite"
```

Keys belong in a **`.env`** beside the app — it is gitignored and loaded at startup.
`.env.example` is the template and must stay empty; a real key there is what GitHub's push
protection blocks, and `tests/test_config.py` fails the build if one appears.

```bash
cp .env.example .env      # then put the real key in .env, never in .env.example
```

Or just pick a provider and paste the key in the Analyst tab. Three buttons there check the
setup before you rely on it: **Test connection**, **Check tool calling** (the analyst needs
it, and not every model has it) and **List models** (ids change often — this asks the
endpoint what your key can actually reach).

The tool layer is provider-independent — `ToolBox` executes calls and knows nothing about
the model, which is why the whole loop is tested offline against a scripted fake client.

Provider messages are echoed back as the provider produced them rather than rebuilt field
by field. Some vendors attach data the conversation must carry unchanged — Gemini 3.x
returns a `thought_signature` on every function call and rejects the next request without
it — and reconstructing the message silently drops anything outside the OpenAI schema.

## Security

Three boundaries, all enforced in `groundtruth/security.py`:

**The API connector is deny-by-default.** It fetches nothing until you allowlist hosts,
and it re-checks that an allowlisted name does not resolve to a private address, which
covers DNS rebinding.

```bash
export GT_ALLOWED_API_HOSTS="api.example.com,data.example.org"
```

**SQL is validated by a parser, not a prefix match.** DuckDB parses the statement and the
guard rejects anything that is not exactly one SELECT. This catches cases prefix matching
misses, such as `WITH a AS (SELECT 1) DELETE FROM t`. Still use a read-only database
credential — the guard is the second line of defence, not the first.

**Untrusted queries are row-capped** before results are returned to the model.

## Alerts

Rules hold state between runs, so a metric that crosses its threshold notifies once
rather than on every cycle:

| Transition | Notifies |
|---|---|
| `fired` | yes — first breach |
| `still_firing` | no — already known |
| `recovered` | yes — back to normal |
| `suppressed` | no — inside the cooldown |

Two kinds: **threshold** rules on an aggregate, and **anomaly** rules that fire when the
latest period falls outside its forecast's prediction interval — no hand-picked number.

Schedule the worker:

```bash
export ALERT_SOURCE=file ALERT_DATA_PATH=sample_data.csv
export ALERT_RULES_FILE=alerts.json ALERT_STATE_FILE=alert_state.json
export ALERT_WEBHOOK_URL="https://hooks.example.com/..."
python alert_worker.py
```

Exit code `10` means a rule newly fired, `0` means nothing changed, `2` is a config error.
The state file is what makes that distinction possible — keep it on durable storage.

```cron
*/15 * * * * cd /path/to/analytical_web_app && /path/to/.venv/bin/python alert_worker.py
```

## Reproducibility

Every action appends to a lineage log, exportable as a Python script that re-runs the
session, as JSON, or as Markdown. Trace any number back through the filters, transforms
and queries that produced it.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q                                  # 496 tests
python -m pytest tests --cov=groundtruth --cov-report=term-missing
```

Coverage is **100% of the library, statements and branches**, enforced by
`fail_under = 100` in `pyproject.toml` so a regression fails the build rather than passing
quietly. Network boundaries are stubbed at `requests` and at the model client, so the suite
needs no API key and makes no outbound calls.

Reaching it was worth more than the number: chasing the last uncovered lines exposed a
NaN p-value that silently blanked the correlation scan, a leakage check disabled by a dtype
comparison, a changepoint detector that could not see a perfect step, and a segment scan
whose scale the outlier itself inflated. One unreachable branch turned out to be dead code
and was removed rather than tested.

The suite runs entirely offline — no API key, no network. Statistics are checked against
constructed data with known answers, the agent loop against a scripted fake client, `test_integration.py`
walks all six layers in one pass, and `test_interaction.py` covers the
selection-to-SQL path end to end.

## Layout

```
app.py                  Streamlit UI
alert_worker.py         unattended alert runner
groundtruth/            the library — 14 modules
  security.py           URL allowlist, SQL parser guard
  store.py              L1 DuckDB engine
  connectors.py         L0 file / database / API
  semantic.py           L2 column roles, grain, keys
  filters.py            L3 filter tree -> SQL
  stats.py              L3 tests with effect sizes
  ml.py                 L3 leaderboard, permutation importance, leakage
  timeseries.py         L3 STL, changepoints, forecasting
  charts.py             role-driven charts, cross-filtering, time controls
  llm.py                provider presets for any OpenAI-compatible endpoint
  theme.py              typography, motion, composite components
  agent.py              L4 tool-calling loop
  insights.py           L4 proactive scan
  report.py             L5 report builder
  alerts.py             L5 state machine
  provenance.py         L5 lineage and script export
tests/                  496 tests, 100% coverage
legacy/                 the original MVP, still runnable
```

## Known limits

- **Single-user.** State lives in a Streamlit session. Concurrent users would need real
  sessions and a shared metadata store.
- **No authentication.** Put an authenticating proxy in front of any shared deployment.
- **The API key is entered in the browser.** Move it to server-side configuration or a
  secret manager before this leaves your machine.
- **Seasonal decomposition needs three cycles** to be stable. It warns below that rather
  than reporting confident nonsense.
- **Derived columns modify the loaded table** rather than being stored as a reusable
  recipe, so they are lost when the dataset is reloaded.
