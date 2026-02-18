# Stage-2 Release Manifest

This manifest defines the canonical Stage-2 release scope and how to reproduce/verify it.

## Canonical Tag

- `stage2_release_v1_2_refactor`

## Claim Scope

- **Fixed:** non-flicker `delay=10` + warmup(both20k) gives `selfmodel - learned` final > 0 with CI95 lower bound > 0 (`n_pairs=30`).
- **Trend:** warmup delay dependence by group contrast `{10,20} - 0` is not significant at `n=30`.
- **Limitation:** with `env.flicker_prob=0.1`, warmup `delay=10` gain does not hold at `n=5`; AUC is significantly negative.

## Primary Artifacts

- Narrative: `docs/stage2_results_2026-02-16.md`
- Main table: `docs/stage2_main_table_2026-02-16.csv`
- Main table (TeX): `docs/stage2_main_table_2026-02-16.tex`
- Warmup paired: `docs/stage2_delay_warmup_both20k_paired_2026-02-16.csv`
- Group difference: `docs/stage2_delay_warmup_groupdiff_0_vs_10_20_2026-02-16.csv`
- Flicker paired: `docs/stage2_delay_warmup_flicker01_paired_2026-02-16.csv`

## Regeneration

From repository root:

```cmd
scripts\reproduce_stage2_release_v1.cmd
```

If run data are stored outside the repo, pass explicit roots:

```cmd
scripts\reproduce_stage2_release_v1.cmd ^
  --warmup-delay10-root <PATH> ^
  --warmup-sweep-root <PATH> ^
  --nonwarmup-root <PATH> ^
  --flicker-root <PATH>
```

## Verification

From repository root:

```cmd
scripts\verify_stage2_release_v1.cmd
```

Guard conditions checked:

- document head contains `Fixed:`, `Trend:`, `Limitation:`
- warmup delay10 fixed claim: `n_pairs >= 30`, `ci95_low > 0`
- non-warmup causal claim: `n_pairs >= 15`, `ci95_low > 0`
- warmup groupdiff coverage: `n_seeds >= 30`
- flicker limitation: `n_pairs >= 5`, `auc ci95_high < 0`

