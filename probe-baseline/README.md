# Training the baseline probe
- chose layer 28 (the last hidden layer) as the one to probe,
   - note that the hidden states the probe was trained on doesn't have to be the one we steer with. In fact, this approach is more common -- train on last hidden layer, and steer with middle-late layers. 
- paragraph breaks (\\n\\n) as the positions to probe, and
   - if there are more than MAX_HIDDEN_PER_PROBLEM (default=100) such positions, we sample a subset uniformly at random.
- It's entirely possible that we don't have enough compute to train the probe for all 1000 problems + run COT + extract hidden states for 100 paragraphs per problem/COT. 
    - If that's the case, we can train on a subset of the problems, and/or reduce the number of paragraphs we probe per problem. Shouldn't be too bad though, with a small model and a good GPU.

## Split workflow

Data collection on 1000 problems is too expensive for one person's compute budget. We split the generation step across teammates, then one person merges and trains.

Three notebooks, in order:

| Notebook | Who runs it | GPU? | Output |
|---|---|---|---|
| `shard_collect.ipynb` | **Each teammate**, on their assigned problem range | Yes (H100/A100) | `shard_<NAME>.pt` |
| `shard_merge.ipynb` | One person, after all shards are in | No (CPU) | `X.pt`, `y.pt`, `problem_ids.pt` |
| `probe_train.ipynb` | Same person | No (CPU) | `linear_probe.pt`, `probe_metrics.json` |

### Step 1: Assign ranges

Pick a split up front. For 4 teammates:

| Person | `START_IDX` | `END_IDX` |
|---|---|---|
| A | 0   | 250 |
| B | 250 | 500 |
| C | 500 | 750 |
| D | 750 | 1000 |

For 2 or 3 teammates, see the table inside `shard_collect.ipynb`. Pick a unique `NAME`.

### Step 2: Each teammate runs `shard_collect.ipynb`

- Set `NAME`, `START_IDX`, `END_IDX` in the config cell. Don't change anything else.
- Runtime → GPU (H100 or A100). Not T4 at default `MAX_NEW_TOKENS=16384`.
- Produces `shard_<NAME>.pt` in their Google Drive (`MyDrive/cs639-outputs/`).
- Cost: ~40–75 units per 250 problems on H100, depending on how many traces hit the 16k token cap. Later problem ranges cost more (harder problems → longer traces).

### Step 3: Put all shards in one shared folder

Drop your `shard_<NAME>.pt` into [shared Google Drive](https://drive.google.com/drive/folders/1Z5bn-IZusVL1dLgX1zvf9417A6t0do0T?usp=share_link)

### Step 4: Merge runner runs `shard_merge.ipynb`

- Point `SHARDS_DIR` at the folder with all the shards.
- Runs consistency checks: same `MODEL_ID`, `PROBE_LAYER`, `MAX_NEW_TOKENS`, and no overlapping problem ranges.
- Outputs `X.pt`, `y.pt`, `problem_ids.pt` + a `merge_meta.json` audit trail.
- CPU runtime is fine. Free.

### Step 5: Merge runner runs `probe_train.ipynb`

- Loads the merged dataset.
- Splits at the **problem level** (not position level — same-problem positions share a label, so splitting at position level leaks).
- Class-weighted BCE, best-val-AUC checkpointing.
- Reports **ROC-AUC, ECE, Brier** (the proposal's primary metrics) plus accuracy / precision / recall / F1 on the test set.
- Saves `linear_probe.pt` and `probe_metrics.json`.

## What goes in git vs. Drive

- **Git**: notebooks, `linear_probe.pt` (tiny), `probe_metrics.json`, `merge_meta.json`.
- **Drive (shared folder)**: `shard_*.pt`, merged `X.pt` / `y.pt` / `problem_ids.pt`.

Shards are ~360 MB each (fp32). GitHub blocks files >100 MB on push, so don't try to commit them.

## Cost summary

At 8.71 units/hr on H100, current config (1 greedy trace per problem, `MAX_NEW_TOKENS=16384`):

| Split | Per person | Total |
|---|---|---|
| 4 people × 250 problems | ~48–72 units | ~190–285 |
| 3 people × 334 problems | ~64–95 units | ~190–285 |
| 2 people × 500 problems | ~95–145 units | ~190–285 |

Lower bound uses the measured 78.6 s/problem from a DEBUG run on early (easier) AIMEs; upper bound adds ~50% for harder later problems hitting the token cap more often.
