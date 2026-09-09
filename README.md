# Fuzzy Characterization Framework

Privacy-compliant **user characterization framework** for enterprise social
networks, implemented as a Python package. It is the reference implementation
of the framework designed and evaluated in the PhD thesis *Privacy-Compliant
User Characterization in Enterprise Social Networks* (chapters "Design of the
Privacy-Compliant User Characterization Framework" and "Framework
Implementation and Evaluation"; see also Mancera Andrade, Portmann and Terán,
*Privacy-Compliant User Characterization Framework in Enterprise Social
Networks*, SSRN 10.2139/ssrn.6544138).

The framework turns raw platform interaction metadata into **fuzzy
behavioural representations** and groups users into overlapping behavioural
segments, without ever reading message content or personal identifiers:

```
 ESN data ──► data compliance ──► characterized user vector ──► fuzzification engine ──► fuzzy clustering ──► performance & metrics
             (consent, pseudonyms,   (API-event indicators +     (membership functions,     (FCM, PFCM, GK      (FSI / silhouette, stability,
              generalised bins)       heuristic attributes +      alpha-cuts, fuzzy          + K-Means / AHC      interpretability, privacy,
                                      compliant demographics)     rule-based layer)          baselines)           profiles, figures)
```

## Contents

| Path | What it is |
|---|---|
| `src/fuzzy_characterization/` | The package (see [Architecture](#architecture)) |
| `configs/` | YAML configurations: membership functions, fuzzy rules, retail case |
| `data/synthetic/` | Coherent synthetic demo data shaped like the retail case (`data/README.md`) |
| `tests/` | pytest suite (membership functions, alpha-cuts, engine, rules, clustering, evaluation, pipeline) |
| `docs/thesis_mapping.md` | Table linking every thesis element to its implementation |
| `legacy/` | The original exploratory scripts, kept for reference (`legacy/README.md`) |
| `outputs/` | Pipeline artefacts (git-ignored) |

## Installation

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # package + CLI (`fcf`) + pytest / kneed
```

## Quick start

```bash
# 1. (optional) regenerate the synthetic demo data
fcf generate-data --out data/synthetic --n-users 400 --days 42 --seed 7

# 2. run the complete framework on the retail-case configuration
fcf run --config configs/retail_case.yaml

# 3. look at the results
cat outputs/retail_case/report.md
open outputs/retail_case/figures/membership_contours.png
```

The run takes a few seconds and writes ~45 artefacts to
`outputs/retail_case/` (tables as CSV/JSON, figures as PNG, a Markdown
report). Everything reported is **group-level**; the only per-user files are
the pseudonymised membership matrix and fuzzy vectors, which are the
anonymised representation intended for downstream components.

Other commands:

```bash
fcf describe --config configs/retail_case.yaml      # membership functions, shift patterns, rules
fcf plot-mf  --config configs/retail_case.yaml      # figures of the membership functions
fcf select-k --config configs/retail_case.yaml --k-min 2 --k-max 10
fcf compare  --config configs/retail_case.yaml --k 4 --methods fcm,pfcm,gk,kmeans,ahc
fcf run --config configs/retail_case_with_rules.yaml # same case, fuzzy rule layer enabled
fcf run --config configs/retail_case.yaml --method gk --k 5 --restarts 10
fcf unpack-events legacy_export.csv                  # expand a serialised event_type_counts column
python -m fuzzy_characterization run -c configs/quick_test.yaml   # equivalent module entry point
```

Or from Python:

```python
from fuzzy_characterization.config import load_config
from fuzzy_characterization.pipeline import run_pipeline

cfg = load_config("configs/retail_case.yaml")
res = run_pipeline(cfg, k=4)
res.profiles["labels"]          # archetype per cluster
res.comparison                  # Table "segmentation quality"
res.fuzzification.fuzzy_vectors # the anonymised fuzzy attribute vector (users x attributes)
res.result.memberships          # FCM membership matrix (users x k)
```

## What the pipeline does

1. **Data compliance** (`compliance/`). Users without consent are excluded
   from every table, identifiers are replaced by salted pseudonyms, direct
   identifiers (`full_name`, `username`, e-mail, phone) are dropped, message
   payload columns are never read, and demographics are generalised (age
   bands, region, role category). A compliance report checks that the feature
   space contains no identifier or content column and that every reported
   group reaches `compliance.min_group_size`.
2. **Characterized user vector** (`data/indicators.py`, `heuristics/`). The
   API event log is aggregated per user into the behavioural indicators of
   the thesis: posting, commenting, reactions, chat, logins by channel,
   topic/stream access, working vs non-working hours, weekly rhythm, per-hour
   and per-weekday activity profiles. Heuristic attributes from the third
   design cycle are added: hierarchy score from the job title, accessible
   streams from the permission table, broadcast share, group memberships and
   post engagement.
3. **Fuzzification engine** (`fuzzification/`). Each indicator is normalised
   and mapped through its membership-function family (trapezoidal,
   triangular, Gaussian, sigmoidal, plus wrap-around and split-shift windows
   for working hours) to linguistic terms; the result is the fuzzy attribute
   vector `mu_1 … mu_n`. Alpha-cuts (0.2 / 0.5 / 0.8) isolate characteristic
   behaviour and are reported as segment sizes. The **fuzzy rule-based layer**
   encodes organisational assumptions (top-down communication, opportunity to
   interact, operational rhythm) as fuzzy IF–THEN rules that re-weight
   attributes, rescale indicators or switch the active-hours window.
4. **Reduction** (`preprocessing/`). Optional normalisation of the membership
   variables and PCA keeping the components that explain ~90 % of the
   variance (at most 20). PCA is also the plane used to draw the clusters.
5. **Clustering** (`clustering/`). Validity indices over a range of k (PC,
   CE, DBI, Xie–Beni, silhouette, elbow of the FCM objective) and a majority
   recommendation; then Fuzzy C-Means with m = 2, 30 seeded restarts, 300
   iterations and ΔJ < 1e-5, keeping the best objective. PFCM,
   Gustafson–Kessel, K-Means and Ward AHC are available with the same
   interface.
6. **Performance and metrics** (`evaluation/`, `profiling/`,
   `visualization/`). The comparative evaluation runs the fuzzy methods on
   the anonymised representation and the baselines on the full-resolution
   behavioural features, scoring separation (fuzzy silhouette index or
   silhouette), stability (ARI across restarts, membership agreement),
   interpretability (feature dominance clarity, membership transparency,
   optional expert ratings) and privacy compliance. Profiles give the average
   behavioural feature values per cluster, match clusters to the four
   archetypes of the thesis (high communicators, reactive consumers,
   operational browsers, low-engagement users), and quantify overlaps.

## Results on the synthetic data

`fcf run --config configs/retail_case.yaml` on the shipped demo data (400
users, 42 days, 387 with consent) retains 8 principal components (90 % of the
variance), recommends k = 4 unanimously (DBI, Xie–Beni and both elbows) and
recovers the four planted archetypes:

| Cluster | Posts / month | Comments / month | Reactions / month | Chat events / month | Topic accesses / month | Working-hours activity |
|---|---|---|---|---|---|---|
| High Communicators | 6.0 | 9.1 | 48.6 | 155.8 | 153.2 | 58.8 % |
| Reactive Consumers | 1.2 | 3.7 | 41.9 | 58.0 | 89.9 | 50.9 % |
| Operational Browsers | 0.6 | 1.3 | 15.5 | 21.5 | 205.1 | 65.9 % |
| Low-Engagement Users | 0.1 | 0.2 | 2.8 | 4.1 | 31.1 | 59.7 % |

| Method | Representation | Silhouette / FSI | Stability | Interpretability | Privacy |
|---|---|---|---|---|---|
| FCM | fuzzy | 0.40 (FSI) | 1.00 | 0.30 | High |
| PFCM | fuzzy | 0.40 (FSI) | 1.00 | 0.28 | High |
| Gustafson–Kessel | fuzzy | 0.27 (FSI) | 1.00 | 0.44 | High |
| K-Means | full | 0.24 (silhouette) | 0.90 | 0.20 | Low |
| AHC (Ward) | full | 0.24 (silhouette) | 0.73 | 0.20 | Low |

The interpretability score here lacks its semantic-consistency component
(no expert ratings are supplied for synthetic data). Absolute values are
specific to the demo data and are not those of the thesis' retail case.

## Configuration

A run is described by one YAML file; files can `include` others.

* `configs/membership_functions.yaml` – the fuzzy features (Table
  "membership functions"): indicator, family, linguistic terms and their
  parameters, normalisation, alpha-cut, plus the shift-pattern windows.
* `configs/fuzzy_rules.yaml` – the rule layer: context terms and rules.
* `configs/retail_case.yaml` – data location, compliance settings, PCA,
  clustering (k = 4, m = 2, 30 restarts) and evaluation options.
* `configs/retail_case_with_rules.yaml`, `configs/quick_test.yaml` – variants.

Adding a behavioural indicator is a matter of computing the column (in
`data/indicators.py` or by joining your own table) and adding a feature block:

```yaml
- name: search
  indicator: searches_per_month
  family: triangular
  terms:
    low:  {family: trapezoidal, params: [0, 0, 0.2, 0.45]}
    high: {family: trapezoidal, params: [0.55, 0.8, 1, 1]}
```

Adding an organisational rule:

```yaml
- name: regional_boundary
  if: [{attribute: region, equals: north_america}]
  then: [{action: scale_weight, features: [chat], factor: 0.8}]
```

## Outputs of a run (`outputs/<run_name>/`)

| Artefact | Content |
|---|---|
| `report.md` | Group-level summary: clusters, archetypes, feature averages, comparison table, alpha-cut segments |
| `compliance_report.json` | Identifiers/content excluded, consent coverage, pseudonymisation, feature-space check, level |
| `user_indicators.csv`, `user_heuristic_attributes.csv`, `user_compliant_demographics.csv` | The characterized user vector (pseudonymised) |
| `membership_functions.csv/json`, `fuzzy_rules.csv`, `rule_firing_strengths.csv` | The fuzzification engine as configured and the rule activations |
| `user_fuzzy_vectors.csv`, `alpha_cut_segments.csv` | Fuzzy attribute vector and alpha-cut segment sizes |
| `pca_explained_variance.csv`, `pca_loadings.csv`, `user_principal_components.csv` | Reduction |
| `k_selection_metrics.csv`, `k_selection_suggestions.json` | Validity indices and recommendation |
| `user_memberships.csv`, `cluster_centers_*.csv`, `clustering_result.json` | Main clustering result |
| `cluster_feature_averages.csv`, `cluster_archetypes.csv`, `cluster_distribution.csv`, `cluster_dominant_attributes.csv`, `cluster_pairwise_overlap.csv` | Profiles |
| `method_comparison.csv`, `method_evaluations.json` | Comparative evaluation (Table "segmentation quality") |
| `figures/*.png` | Membership functions, shift patterns, alpha-cuts, explained variance, k selection, membership contours in the PCA plane, scatter, membership heatmap, cluster profiles, method comparison |

## Architecture

```
src/fuzzy_characterization/
├── config.py            PipelineConfig dataclasses + YAML loading (with includes)
├── pipeline.py          run_pipeline(): orchestration and artefact writing
├── cli.py               `fcf` command line
├── data/                schema, loaders, synthetic generator, behavioural indicators
├── heuristics/          hierarchy score, stream permissions, post engagement/similarity
├── compliance/          pseudonymisation, consent, generalisation, compliance checks
├── fuzzification/       membership functions, alpha-cuts, engine, fuzzy rule layer
├── preprocessing/       scaling transforms, full-resolution features, PCA
├── clustering/          FCM, PFCM, Gustafson–Kessel, K-Means, AHC, k selection
├── evaluation/          validity indices, stability, interpretability, privacy, comparison
├── profiling/           cluster feature averages, archetype labelling, overlaps
└── visualization/       matplotlib figures
```

`docs/thesis_mapping.md` lists, for every element of the two thesis chapters,
where it is implemented.

## Using your own data

Export the tables described in `data/README.md` into a directory (only
interaction metadata and generalised profile attributes are needed; message
payloads are never read) and point the configuration at it:

```bash
fcf run --config configs/retail_case.yaml --data-root data/raw/my_case --run-name my_case
```

Membership-function parameters live on the normalised [0, 1] domain, so the
default configuration transfers across organisations; shift windows,
alpha-cut levels and organisational rules are the parts expected to be
adapted with the customer (the thesis' customer-feedback cycle). Expert
ratings of the resulting segments can be supplied through
`evaluation.expert_ratings_file` (CSV with `cluster,rating`) to complete the
interpretability score with its semantic-consistency component.

## Testing

```bash
pytest
```

The suite checks the membership-function equations against the thesis
examples, alpha-cut nesting, the engine and the rule layer, that every
clustering method recovers planted clusters, the validity/stability/
interpretability/privacy metrics, and that the full pipeline recovers the
four archetypes planted in the synthetic data.

## Privacy notes

* Real exports belong in `data/raw/` (git-ignored). Never commit them.
* The framework consumes counts, timestamps and normalised paths only; a
  content column found in any table is dropped and reported.
* Alpha-cut segments and clusters smaller than `compliance.min_group_size`
  are flagged as not reportable.
* Pseudonyms are salted hashes; change `compliance.pseudonym_salt` per case.
