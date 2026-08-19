# Project agent instructions

## CARVE-FL experiment implementation

Any agent implementing or running the CARVE-FL feasibility experiment must read these files completely, in this order:

1. `experiment/experiment_01_causal_colored_mnist/HERMES_IMPLEMENTATION_CONTRACT.md` — normative engineering contract.
2. `experiment/experiment_01_causal_colored_mnist/configs/carve_cmnist_v1.yaml` — locked experiment configuration; read but do not edit.
3. `experiment/experiment_01_causal_colored_mnist/PROTOCOL_LOCK.json` — expected hashes for the design artifacts.
4. `experiment/experiment_01_causal_colored_mnist/carve_fl_causal_colored_mnist_plan.md` — scientific rationale, hypotheses, thresholds, and kill switches.
5. `ideaspark_run/federated-relation-unlearning/phase3_revise/final_candidate.json` — provenance of the research mechanism.

When implementation details conflict, the implementation contract controls code behavior. The locked YAML controls numerical configuration. A current user instruction always has higher authority. Do not silently repair a scientific ambiguity, change a seed or threshold, replace a metric, add a baseline, or tune on test data. Record any necessary deviation in `experiment/experiment_01_causal_colored_mnist/DEVIATION_REQUEST.md` and stop before running the affected stage.

## Role split

- **Hermes** owns code creation, configuration creation, unit/integration tests, experiment execution, raw result collection, and factual implementation/run reports.
- **Codex/design owner** owns protocol interpretation, implementation audit, gate approval after the debug seed, statistical validation, scientific conclusions, and later manuscript/reviewer work.
- Neither role may convert a failed run, infeasible edit, invalid data seed, or failed precondition into a successful result.

## Mandatory gate

Hermes must not start the eight main seeds until it has produced the debug audit packet specified in the implementation contract and the design owner has reviewed it. A crash must be reported; do not silently retry it with changed settings.

## Experiment 02 representation readiness

Any agent implementing or running Experiment 02 must read, in order:

1. `experiment/experiment_02_representation_readiness/HERMES_IMPLEMENTATION_CONTRACT.md`
2. `experiment/experiment_02_representation_readiness/configs/readiness_cmnist_v1.yaml`
3. `experiment/experiment_02_representation_readiness/PROTOCOL_LOCK.json`
4. `experiment/experiment_02_representation_readiness/REPRESENTATION_READINESS_PLAN.md`
5. `experiment/experiment_02_representation_readiness/DESIGN_OWNER_AUTHORIZATION.md`

Experiment 02 is train/validation-only and is not a CARVE-FL experiment.
Development implementation and CPU execution are approved. Confirmatory seeds
remain prohibited until the design owner writes `CONFIRMATORY_GATE.md`. MNIST
test/client-test roles and all unlearning solvers remain prohibited throughout
Experiment 02.

## Experiment 03 coexistence window

Any agent implementing or running Experiment 03 must read, in order:

1. `experiment/experiment_03_coexistence_window/HERMES_IMPLEMENTATION_CONTRACT.md`
2. `experiment/experiment_03_coexistence_window/configs/coexistence_window_v1.yaml`
3. `experiment/experiment_03_coexistence_window/PROTOCOL_LOCK.json`
4. `experiment/experiment_03_coexistence_window/COEXISTENCE_WINDOW_PLAN.md`
5. `experiment/experiment_03_coexistence_window/DESIGN_OWNER_AUTHORIZATION.md`

Experiment 03 is train/validation-only and is not a CARVE-FL experiment.
Development implementation and the three locked CPU trajectories are approved.
Confirmatory seeds remain prohibited until the design owner writes
`CONFIRMATORY_GATE.md`. MNIST test/client-test and every unlearning solver remain
prohibited.

## Experiment 04 CARVE-FL at the coexistence checkpoint

Any agent implementing or running Experiment 04 must read, in order:

1. `experiment/experiment_04_carve_fl_coexistence/HERMES_IMPLEMENTATION_CONTRACT.md`
2. `experiment/experiment_04_carve_fl_coexistence/configs/carve_cmnist_coexistence_v1.yaml`
3. `experiment/experiment_04_carve_fl_coexistence/PROTOCOL_LOCK.json`
4. `experiment/experiment_04_carve_fl_coexistence/CARVE_FL_FEASIBILITY_PLAN.md`
5. `experiment/experiment_04_carve_fl_coexistence/DESIGN_OWNER_AUTHORIZATION.md`
6. `experiment/experiment_03_coexistence_window/DESIGN_OWNER_CONFIRMATORY_REVIEW.md`
7. `ideaspark_run/federated-relation-unlearning/phase3_revise/final_candidate.json`

Experiment 04 is a new CARVE-FL feasibility experiment. Implementation, tests,
and debug seed 5101 are approved. All eight main seeds and all MNIST
test/client-test access remain prohibited until the design owner writes
`MAIN_GATE.md`. No prior experiment seed, checkpoint, or output may be reused as
an Experiment 04 result.

## Experiment 04 search diagnostics (04-1 / 04-2 / 04-3)

Any agent implementing or running the Experiment 04 search-diagnostic
continuation must read, in order:

1. `experiment/experiment_04_carve_fl_coexistence/HERMES_DIAGNOSTIC_IMPLEMENTATION_CONTRACT.md`
2. `experiment/experiment_04_carve_fl_coexistence/configs/carve_search_diagnostic_v1.yaml`
3. `experiment/experiment_04_carve_fl_coexistence/DIAGNOSTIC_PROTOCOL_LOCK.json`
4. `experiment/experiment_04_carve_fl_coexistence/CARVE_FL_SEARCH_DIAGNOSTIC_PLAN.md`
5. `experiment/experiment_04_carve_fl_coexistence/DIAGNOSTIC_INPUT_LOCK.json`
6. `experiment/experiment_04_carve_fl_coexistence/DESIGN_OWNER_DIAGNOSTIC_AUTHORIZATION.md`
7. `experiment/experiment_04_carve_fl_coexistence/MAIN_RESULT_REVIEW_01.md`
8. the parent Experiment 04 contract/config/lock named by the diagnostic contract.

This is an Experiment 04 continuation, not Experiment 05. Parent artifacts are
read-only hash-locked inputs. 04-1 and 04-2 are Oracle-blind and may not access
MNIST test/client-test or opened-test artifacts. Implementation, tests, and
debug seed 5101 are approved. The eight diagnostic confirmatory instances are
prohibited until the design owner writes `DIAGNOSTIC_MAIN_GATE.md`.

## Experiment 05 solution transfer

Any agent implementing or running Experiment 05 must read, in order:

1. `experiment/experiment_05_solution_transfer/HERMES_IMPLEMENTATION_CONTRACT.md`
2. `experiment/experiment_05_solution_transfer/configs/solution_transfer_v1.yaml`
3. `experiment/experiment_05_solution_transfer/PROTOCOL_LOCK.json`
4. `experiment/experiment_05_solution_transfer/SOLUTION_TRANSFER_PLAN.md`
5. `experiment/experiment_05_solution_transfer/INPUT_LOCK.json`
6. `experiment/experiment_05_solution_transfer/DESIGN_OWNER_AUTHORIZATION.md`
7. the Experiment 04 parent contracts/configs/locks named by the Experiment 05
   contract.

Experiment 05 is a train/validation-only cross-seed transfer experiment. It
reuses the hash-locked seed-5101 primal-feasible, non-certified candidate as a
discovery artifact and never counts seed 5101 as target evidence.
Implementation, tests, and source-only replay are approved. Evaluation on
target seeds 6101–6801 remains prohibited until the design owner writes
`MAIN_GATE.md`. Target diagnostic outputs, Oracle heads/distances, MNIST
test/client-test roles, opened-test artifacts, and all parent mutations are
prohibited.

## Experiment 06 search-prior placebo validity

Any agent implementing or running Experiment 06 must read, in order:

1. `experiment/experiment_06_search_prior_validity/HERMES_IMPLEMENTATION_CONTRACT.md`
2. `experiment/experiment_06_search_prior_validity/configs/search_prior_validity_v1.yaml`
3. `experiment/experiment_06_search_prior_validity/PROTOCOL_LOCK.json`
4. `experiment/experiment_06_search_prior_validity/SEARCH_PRIOR_VALIDITY_PLAN.md`
5. `experiment/experiment_06_search_prior_validity/INPUT_LOCK.json`
6. `experiment/experiment_06_search_prior_validity/DESIGN_OWNER_AUTHORIZATION.md`
7. `experiment/experiment_06_search_prior_validity/HERMES_DELIVERY_CHECKLIST.md`
8. the Experiment 04/05 parent files named by the Experiment 06 contract and
   input lock.

Experiment 06 is a train/validation-only mechanistic placebo validation. It
generates 16 new target models and tests a fixed seed-5101 constraint-transfer
search prior against an equal-budget block-permuted matched placebo. Old
Experiment 05 target outputs are prohibited evidence. Implementation, tests,
source replay, and debug seed 7001 are approved. Confirmatory seeds 7101–8601
remain prohibited until the design owner writes `MAIN_GATE.md`. MNIST
test/client-test roles, opened-test/Phase-2 artifacts, Oracle artifacts, target
diagnostics, multi-source tuning, replacement seeds, and parent mutations are
prohibited.

## Experiment 07 functional-signature reproducibility

Any agent implementing or running Experiment 07 must read, in order:

1. `experiment/experiment_07_functional_signature_reproducibility/HERMES_IMPLEMENTATION_CONTRACT.md`
2. `experiment/experiment_07_functional_signature_reproducibility/configs/functional_signature_reproducibility_v1.yaml`
3. `experiment/experiment_07_functional_signature_reproducibility/PROTOCOL_LOCK.json`
4. `experiment/experiment_07_functional_signature_reproducibility/FUNCTIONAL_SIGNATURE_REPRODUCIBILITY_PLAN.md`
5. `experiment/experiment_07_functional_signature_reproducibility/INPUT_LOCK.json`
6. `experiment/experiment_07_functional_signature_reproducibility/DESIGN_OWNER_AUTHORIZATION.md`
7. `experiment/experiment_07_functional_signature_reproducibility/HERMES_DELIVERY_CHECKLIST.md`
8. the Experiment 04/05/06 parent files named by the Experiment 07 contract and
   input lock.

Experiment 07 is a train/validation-only function-space mechanism-readiness
experiment and is not a CARVE-FL execution. It tests whether the fixed seed-5101
functional edit signature can be reconstructed on sealed held-out anchors in
new target models. Implementation, synthetic tests, source/anchor freeze, and
debug seed 8701 are approved. Confirmatory seeds 8801–9501 remain prohibited
until the design owner writes `MAIN_GATE.md`. Every unlearning/search solver,
post-edit `gD/gP` or certificate evaluation, MNIST test/client-test,
opened-test/Phase-2, Oracle, historical target output, replacement seed, and
parent mutation is prohibited.

## Experiment 07-01 exploratory translation and diagnostics

Any agent implementing or running Experiment 07-01 must read, in order:

1. `experiment/experiment_07_01_exploratory_translation_diagnostics/HERMES_IMPLEMENTATION_CONTRACT.md`
2. `experiment/experiment_07_01_exploratory_translation_diagnostics/configs/exploratory_followup_v1.yaml`
3. `experiment/experiment_07_01_exploratory_translation_diagnostics/PROTOCOL_LOCK.json`
4. `experiment/experiment_07_01_exploratory_translation_diagnostics/EXPLORATORY_FOLLOWUP_PLAN.md`
5. `experiment/experiment_07_01_exploratory_translation_diagnostics/INPUT_LOCK.json`
6. `experiment/experiment_07_01_exploratory_translation_diagnostics/DESIGN_OWNER_AUTHORIZATION.md`
7. `experiment/experiment_07_01_exploratory_translation_diagnostics/HERMES_DELIVERY_CHECKLIST.md`
8. the parent Experiment 07 files named by the input lock.

Experiment 07-01 is an exploratory, train/validation-only continuation. It
translates exactly the seven frozen ready targets and diagnoses frozen target
9501 without translating it. Parent files are read-only. Implementation,
synthetic tests, and hash-only preflight are approved; real Track A/B execution
requires design-owner `RUN_GATE.md`. No confirmatory claim, retraining,
replacement, tuning, test/Oracle/Phase-2 access, unlearning/search outcome, or
parent mutation is permitted.

## Experiment 08 B3-anchored functional calibration

Any agent implementing or running Experiment 08 must read, in order:

1. `experiment/experiment_08_b3_functional_calibration/HERMES_IMPLEMENTATION_CONTRACT.md`
2. `experiment/experiment_08_b3_functional_calibration/configs/b3_functional_calibration_v1.yaml`
3. `experiment/experiment_08_b3_functional_calibration/PROTOCOL_LOCK.json`
4. `experiment/experiment_08_b3_functional_calibration/B3_FUNCTIONAL_CALIBRATION_PLAN.md`
5. `experiment/experiment_08_b3_functional_calibration/INPUT_LOCK.json`
6. `experiment/experiment_08_b3_functional_calibration/DESIGN_OWNER_AUTHORIZATION.md`
7. `experiment/experiment_08_b3_functional_calibration/HERMES_DELIVERY_CHECKLIST.md`
8. parent Experiment 07 files named by the input lock.

Experiment 08 is a fresh-target, train/validation-only mechanism experiment.
It tests B3-AFRC: an unchanged B3 backbone plus a calibration-only functional
residual correction. Implementation, tests, source replay, and debug seed 9601
are approved. Confirmatory seeds 9701–11201 require design-owner
`MAIN_GATE.md`. Experiment 07/07-01 target outputs are prohibited runtime
evidence. No replacement, tuning, test/Oracle/Phase-2 access, unlearning/search
outcome, or parent mutation is permitted.

## Experiment 08.02 B3 retention-budget discovery

Any agent implementing or running Experiment 08.02 must read, in order:

1. `experiment/experiment_08_02_b3_retention_budget_discovery/HERMES_IMPLEMENTATION_CONTRACT.md`
2. `experiment/experiment_08_02_b3_retention_budget_discovery/configs/b3_retention_budget_discovery_v1.yaml`
3. `experiment/experiment_08_02_b3_retention_budget_discovery/PROTOCOL_LOCK.json`
4. `experiment/experiment_08_02_b3_retention_budget_discovery/B3_RETENTION_BUDGET_DISCOVERY_PLAN.md`
5. `experiment/experiment_08_02_b3_retention_budget_discovery/INPUT_LOCK.json`
6. `experiment/experiment_08_02_b3_retention_budget_discovery/DESIGN_OWNER_AUTHORIZATION.md`
7. `experiment/experiment_08_02_b3_retention_budget_discovery/HERMES_DELIVERY_CHECKLIST.md`
8. parent files named by the Experiment 08.02 input lock.

Experiment 08.02 is a train/validation-only exploratory budget-selection study,
not a confirmatory AFRC experiment. It estimates one global B3 energy fraction
using a locked discovery grid on fresh targets, then checks the mechanically
selected fraction on a disjoint fresh-target stability group. Implementation,
tests, source replay, and debug seed 13001 are approved. Discovery requires
design-owner `DISCOVERY_GATE.md`; stability construction/evaluation requires a
later `STABILITY_GATE.md`. No historical target runtime evidence, replacement,
target-specific fraction, adaptive or continuous search, grid/threshold/ridge/
radius tuning, MNIST test/client-test, Oracle/Phase-2, unlearning/search outcome,
parent mutation, or confirmatory efficacy claim is permitted.

## Experiment 09 B3-SAFE gated AFRC

Any agent implementing or running Experiment 09 must read, in order:

1. `experiment/experiment_09_b3_safe_gated_afrc/HERMES_IMPLEMENTATION_CONTRACT.md`
2. `experiment/experiment_09_b3_safe_gated_afrc/configs/b3_safe_gated_afrc_v1.yaml`
3. `experiment/experiment_09_b3_safe_gated_afrc/PROTOCOL_LOCK.json`
4. `experiment/experiment_09_b3_safe_gated_afrc/B3_SAFE_GATED_AFRC_PLAN.md`
5. `experiment/experiment_09_b3_safe_gated_afrc/INPUT_LOCK.json`
6. `experiment/experiment_09_b3_safe_gated_afrc/DESIGN_OWNER_AUTHORIZATION.md`
7. `experiment/experiment_09_b3_safe_gated_afrc/HERMES_DELIVERY_CHECKLIST.md`
8. parent files named by the Experiment 09 input lock.

Experiment 09 is a fresh-target, train/validation-only functional mechanism
experiment. It tests B3-SAFE-v1: eta-conditioned residual relinearization plus
a four-fold calibration-only target gate and byte-identical B3 fallback.
Implementation, tests, source replay, and debug seed 15501 are approved.
Development seeds 15601-17101 require design-owner `DEVELOPMENT_GATE.md`;
confirmation seeds 17201-18701 require a later `CONFIRMATORY_GATE.md` bound to
the unchanged source and frozen development audit. Historical target runtime
evidence, replacement, holdout-driven gate/eta, continuous search, tuning,
MNIST test/client-test, Oracle/Phase-2, target diagnostics, unlearning/search
outcomes, parent mutation, and claims outside the locked functional population
are prohibited.

## Experiment 09-01 AFRC value diagnostic

Any agent implementing or running Experiment 09-01 must read, in order:

1. `experiment/experiment_09_01_afrc_value_diagnostic/HERMES_IMPLEMENTATION_CONTRACT.md`
2. `experiment/experiment_09_01_afrc_value_diagnostic/configs/afrc_value_diagnostic_v1.yaml`
3. `experiment/experiment_09_01_afrc_value_diagnostic/PROTOCOL_LOCK.json`
4. `experiment/experiment_09_01_afrc_value_diagnostic/AFRC_VALUE_DIAGNOSTIC_PLAN.md`
5. `experiment/experiment_09_01_afrc_value_diagnostic/INPUT_LOCK.json`
6. `experiment/experiment_09_01_afrc_value_diagnostic/PARENT_TARGET_SNAPSHOT.json`
7. `experiment/experiment_09_01_afrc_value_diagnostic/DESIGN_OWNER_AUTHORIZATION.md`
8. `experiment/experiment_09_01_afrc_value_diagnostic/HERMES_DELIVERY_CHECKLIST.md`
9. parent Experiment 09 files named by the input lock.

Experiment 09-01 is a train/validation-only exploratory continuation that asks
whether AFRC has source-specific residual signal and pre-holdout-selectable net
value sufficient to justify further algorithm research. It does not repair or
confirm Experiment 09. It may retrospectively parse the frozen development JSON
and, after a separate design-owner `RUN_GATE.md`, consume exactly the 15 sealed
eligible parent targets 17301-18701 once as exploratory value-diagnostic
evidence. Seed 17201 remains sealed and ineligible. Implementation, tests, and
hash-only preflight are approved; all real Track A/B data access is prohibited
until `RUN_GATE.md`. Parent mutation, retraining, replacement, tuning, new eta,
holdout-driven selection, MNIST test/client-test, Oracle/Phase-2,
unlearning/search outcomes, Experiment 09 `CONFIRMATORY_GATE.md`, and any
confirmatory or deployment claim are prohibited.

## Experiment 10 B3 functional-gradient rotation

Any agent implementing or running Experiment 10 must read, in order:

1. `experiment/experiment_10_b3_functional_gradient_rotation/HERMES_IMPLEMENTATION_CONTRACT.md`
2. `experiment/experiment_10_b3_functional_gradient_rotation/configs/b3_functional_gradient_rotation_v1.yaml`
3. `experiment/experiment_10_b3_functional_gradient_rotation/PROTOCOL_LOCK.json`
4. `experiment/experiment_10_b3_functional_gradient_rotation/B3_FUNCTIONAL_GRADIENT_ROTATION_PLAN.md`
5. `experiment/experiment_10_b3_functional_gradient_rotation/INPUT_LOCK.json`
6. `experiment/experiment_10_b3_functional_gradient_rotation/DESIGN_OWNER_AUTHORIZATION.md`
7. `experiment/experiment_10_b3_functional_gradient_rotation/HERMES_DELIVERY_CHECKLIST.md`
8. parent Experiment 09/09-01 files named by the input lock.

Experiment 10 is a staged, train/validation-only functional-mechanism study. It
tests B3-FGR-v1: a full-radius spherical B3 rotation toward the matched
functional gradient after projecting out the B3 radial and local deletion/
preservation constraint-gradient span. Current authorization covers
implementation, tests, source-only replay, hash-only preflight, and synthetic
smoke only. Every real target remains prohibited. Seed 17301 requires a later
`DISCOVERY_DEBUG_GATE.md`; discovery seeds 17401-18701 require
`DISCOVERY_GATE.md`; fresh debug/development/confirmation require their own
later gates. Stage A may open calibration only and a negative/invalid leverage
label permanently prohibits fresh work under v1. Experiment 09-01 outputs,
algorithm holdouts, seed 17201, replacement, tuning, continuous angles,
post-edit scalar gD/gP, unlearning/search solvers, MNIST test/client-test,
Oracle/Phase-2, parent mutation, and claims beyond the locked functional
population are prohibited.

## Experiment 10-01 B3-FGR numerical-feasibility and curvature diagnostic

Any agent implementing or running Experiment 10-01 must read, in order:

1. `experiment/experiment_10_01_b3_fgr_curvature_diagnostic/HERMES_IMPLEMENTATION_CONTRACT.md`
2. `experiment/experiment_10_01_b3_fgr_curvature_diagnostic/configs/b3_fgr_curvature_diagnostic_v1.yaml`
3. `experiment/experiment_10_01_b3_fgr_curvature_diagnostic/PROTOCOL_LOCK.json`
4. `experiment/experiment_10_01_b3_fgr_curvature_diagnostic/B3_FGR_CURVATURE_DIAGNOSTIC_PLAN.md`
5. `experiment/experiment_10_01_b3_fgr_curvature_diagnostic/INPUT_LOCK.json`
6. `experiment/experiment_10_01_b3_fgr_curvature_diagnostic/DESIGN_OWNER_AUTHORIZATION.md`
7. `experiment/experiment_10_01_b3_fgr_curvature_diagnostic/HERMES_DELIVERY_CHECKLIST.md`
8. parent Experiment 10 files named by the input lock.

Experiment 10-01 is an exploratory, calibration-only continuation. It first
reproduces the frozen Experiment 10 failure census and then, only after a
design-owner `REPLAY_GATE.md`, may replay the same consumed calibration targets
to compare one-shot FGR with fixed two-step and four-step sequential
relinearization. Initial authorization covers implementation, tests, synthetic
smoke, hash-only preflight, and Track A parsing of locked JSON/manifests only.
Parent target tensors and Track B replay remain prohibited. No new target,
fresh seed, replacement, algorithm holdout, test/client-test, Oracle/Phase-2,
Experiment 09-01 runtime evidence, target-specific schedule, tuning, parent
mutation, or retrospective relabeling of Experiment 10 is permitted. A positive
result may justify a separate Experiment 11 design but never authorizes fresh
execution by itself.

## Experiment 11 mechanism selection

Any agent implementing or running Experiment 11 must read, in order:

1. `experiment/experiment_11_mechanism_selection/HERMES_IMPLEMENTATION_CONTRACT.md`
2. `experiment/experiment_11_mechanism_selection/configs/mechanism_selection_v1.yaml`
3. `experiment/experiment_11_mechanism_selection/PROTOCOL_LOCK.json`
4. `experiment/experiment_11_mechanism_selection/MECHANISM_SELECTION_PLAN.md`
5. `experiment/experiment_11_mechanism_selection/INPUT_LOCK.json`
6. `experiment/experiment_11_mechanism_selection/DESIGN_OWNER_AUTHORIZATION.md`
7. `experiment/experiment_11_mechanism_selection/HERMES_DELIVERY_CHECKLIST.md`
8. the Experiment 10 parent files named by the input lock.

Experiment 11 is a train/validation-only, calibration-only replay that compares
soft-protection FGR and free-residual FGR against hard FGR on the fourteen
already-consumed targets 17401-18701. Implementation, tests, source-only replay,
hash-only preflight, and synthetic smoke are approved. The replay requires a
design-owner `REPLAY_GATE.md` bound to the reviewed source tree and preflight
audit. Fresh targets, holdouts, MNIST test/client-test, Oracle/Phase-2, seed
17201, Experiment 09-01 runtime evidence, target-specific lambda/angle search,
parent mutation, and any confirmatory claim are prohibited. Post-edit gD/gP
evaluation is allowed only for the mechanism-A feasibility check.

## Proposed next research direction: minibatch-consensus sparse relation editing

### Motivation

The current CARVE-FL / functional-editing line has progressively tested whether
a client-specific spurious relation can be removed by finding a single globally
useful edit direction while preserving unrelated behavior.

The latest Experiment 10 / 10-01 evidence suggests that this formulation may be
too restrictive for a substantial fraction of targets. In particular, useful
functional directions can overlap strongly with deletion/preservation
constraint directions, so hard projection may remove not only harmful
components but also the directions carrying the desired edit signal.

This does **not** establish that no globally feasible edit exists. It establishes
only that the current family of global constrained-direction mechanisms has low
applicability on the locked population.

A possible next research direction is therefore to replace the requirement of
finding one globally valid edit direction with a staged procedure:

> local evidence discovery -> cross-minibatch consensus -> bounded sparse edit
> -> held-out validation -> accept or rollback

The conceptual inspiration comes from validation-gated iterative optimization:
rather than trusting one globally derived direction, repeatedly propose a small
candidate modification, test its real functional effect, retain it only when it
improves the deletion-preservation trade-off, and otherwise restore the previous
accepted model.

---

### Research question

The central question is:

> Does a client-specific spurious relation depend on a small set of backbone
> functional units that can be identified consistently across independent local
> minibatches and edited without materially damaging unrelated behavior?

The proposed method is intended for **client-specific relation unlearning**.

A requesting client receives the current server model and performs local
relation editing. The resulting client-specific unlearned model is conceptually
separate from the unchanged server/global model unless a future protocol
explicitly studies how such edits should be aggregated.

The first version should therefore avoid claiming that the edited backbone can
be safely uploaded and globally aggregated.

---

### Core representation

Let the current model be

    M_t = H_t(B_t(x))

where:

- `B_t` is the current backbone;
- `H_t` is the decision head;
- `M_t` is the last accepted client model.

The target is not merely to reduce ordinary classification loss.

The target is to reduce dependence on a specified spurious relation while
preserving unrelated client behavior.

The framework should distinguish three roles of data:

1. **discovery data**
   - used to identify candidate relation-relevant functional units;

2. **edit data**
   - used to optimize only the selected units;

3. **validation data**
   - never used to rank candidate units;
   - used only to accept or reject the resulting candidate model.

The exact split and reuse policy must be locked before any confirmatory
experiment.

---

### Stage 1: minibatch-level relation attribution

Partition the local relation-evidence set into multiple minibatches:

    D_relation
        |
        +--> B_1
        +--> B_2
        +--> ...
        +--> B_M

Each minibatch is analyzed independently.

For every minibatch `B_m`, compute the importance of candidate backbone
functional units with respect to the target spurious relation:

    B_m
      |
      v
    relation attribution
      |
      v
    {g_i : s_m(g_i)}

where:

- `g_i` is a candidate functional unit;
- `s_m(g_i)` is its relation-attribution score on minibatch `m`.

The initial unit of editing should **not** be an individual scalar weight.

Candidate units should preferentially be structured objects such as:

- convolutional channels;
- neurons;
- attention heads;
- layer blocks;
- low-rank parameter directions;
- other predefined functional parameter groups.

The attribution mechanism should be deterministic and model-based rather than
LLM-based. Candidate choices include gradient-, activation-, Jacobian-,
intervention-, or ablation-based scores.

The purpose of minibatching is not merely computational.

It provides repeated independent evidence about whether the same functional
units remain important across different subsets of relation examples.

---

### Stage 2: cross-minibatch consensus

A functional unit should not be selected only because it receives a large score
on one minibatch.

For every candidate unit `g`, aggregate evidence across minibatches using at
least the following factors:

- **importance**:
  how strongly changing `g` is associated with the target relation;

- **frequency**:
  in how many minibatches `g` is identified as important;

- **stability**:
  how consistent the attribution score or ranking of `g` is across minibatches.

Conceptually:

    minibatch 1 -> candidate units
    minibatch 2 -> candidate units
    ...
    minibatch M -> candidate units
                       |
                       v
               consensus aggregation
                       |
                       v
               ranked candidate pool

A unit that is moderately important in most minibatches may be more trustworthy
than a unit that is extremely important in only one minibatch.

The intended signal is therefore:

> repeated cross-minibatch evidence of relation relevance

rather than:

> maximum single-minibatch attribution.

The exact aggregation score should be treated as a design choice and must be
locked before confirmatory use.

---

### Stage 3: bounded edit budget

Introduce an explicit edit budget `K`.

At one optimization step, only the top-ranked `K` functional units may be
modified:

    ranked units
        |
        v
    top-K selection
        |
        v
    editable set G_t

All non-selected backbone units remain frozen for that step.

The role of `K` is analogous to a bounded optimization step:

- prevent unrestricted backbone drift;
- reduce collateral damage;
- improve interpretability;
- make every accepted change attributable to a small candidate set;
- permit rollback to a well-defined previous state.

The candidate should normally contain the **joint edit of the selected top-K
units**, rather than validating each unit independently.

This avoids assuming that useful relation edits are strictly additive or that a
single unit must improve the objective in isolation.

---

### Stage 4: local sparse editing

Given the selected set `G_t`, construct a candidate model:

    M_t
      |
      v
    freeze all parameters outside G_t
      |
      v
    optimize G_t on local edit data
      |
      v
    M_candidate

The optimization objective should explicitly separate:

- target relation removal;
- preservation of unrelated behavior.

The framework should not assume in advance that a mathematically promising
update direction is safe.

The result of local optimization is only a **candidate**.

It becomes the new client model only after held-out validation.

---

### Stage 5: deletion-preservation validation gate

Every candidate must be evaluated on held-out validation evidence.

The gate contains two conceptually separate components.

#### Deletion validation

Measure whether the target relation has actually weakened.

Possible quantities include:

- targeted association strength;
- counterfactual response gap;
- targeted flip rate;
- relation-specific prediction dependence;
- representation/decision dependence measures.

Denote the deletion objective by:

    D(M)

where a larger improvement means stronger removal of the target relation.

#### Preservation validation

Measure damage to unrelated behavior.

Possible quantities include:

- clean accuracy;
- Macro-F1;
- worst-group accuracy;
- prediction divergence;
- retention-set JS divergence;
- other locked preservation metrics.

Denote preservation damage by:

    P(M)

The gate may eventually use either:

1. a constrained rule

       deletion improvement >= tau_D
       AND
       preservation damage <= tau_P

or

2. a pre-registered scalar selection score

       S(M) = deletion benefit - lambda * preservation damage

The validation rule must be fixed before evaluation and may not be adjusted
after observing the held-out result.

---

### Stage 6: accept or rollback

Let `M_t` be the last accepted model.

After evaluating `M_candidate`:

    if Gate(M_candidate, M_t) == PASS:
        M_{t+1} = M_candidate
    else:
        M_{t+1} = M_t

A rejected candidate does not become part of the client model.

The previous accepted state is retained byte-for-byte where practical.

This converts relation unlearning from a one-shot global search problem into an
iterative validation-gated editing process:

    accepted model
        |
        v
    minibatch attribution
        |
        v
    consensus
        |
        v
    top-K bounded edit
        |
        v
    candidate model
        |
        v
    held-out validation
       / \
    PASS   FAIL
     |       |
    keep   rollback
     |
     v
    next iteration

---

### Relation to head-only editing

The current proposal should not assume that backbone modification is always
necessary.

Two distinct hypotheses must remain separate:

#### Decision-level suppression

A spurious relation may be removable by freezing the backbone and modifying
only the head.

In that case, the backbone may still encode the spurious feature, but the
client decision function no longer relies on it.

#### Representation-level relation removal

If the research goal requires reducing the relation inside the learned
representation itself, head-only retraining may be insufficient and selective
backbone editing may be necessary.

Therefore a future feasibility study should include a locked head-only baseline.

The sparse-backbone method is justified only if it provides measurable value
beyond head-only editing.

---

### Federated interpretation

The first version of this mechanism should be interpreted as **local
client-specific unlearning**, not global federated re-optimization.

For a requesting client `c`:

    global model M
          |
          v
    client c local relation editing
          |
          v
    M_c^u

while other clients continue to use the unchanged global model:

    M_j = M,    j != c

This provides structural isolation from other clients during the first
feasibility stage.

Any future proposal that uploads or aggregates the edited backbone must be
treated as a separate research problem because local relation removal may
interfere with relations that remain useful to other clients.

---

### Primary feasibility hypothesis

Before designing a full unlearning algorithm, the following hypothesis should
be tested directly:

> Independent minibatches exposing the same spurious relation will repeatedly
> identify a non-random subset of common backbone functional units.

The first experiment should therefore measure:

1. cross-minibatch overlap of selected units;
2. rank/score stability across minibatches;
3. whether consensus-selected units outperform equally sized random units;
4. whether consensus-selected units outperform single-minibatch-selected units;
5. whether sparse editing improves deletion while preserving unrelated behavior;
6. whether sparse backbone editing adds value beyond a head-only baseline.

A particularly important placebo is:

    consensus-selected Top-K units

versus

    matched random Top-K units

under the same optimization budget.

If the consensus-selected units do not reliably outperform the matched random
placebo, the mechanism should be considered unsupported.

---

### Kill criteria

This research direction should be stopped or substantially revised if any of
the following occur:

- cross-minibatch functional-unit overlap is near random;
- rankings are unstable across minibatches;
- consensus-selected units do not outperform matched random units;
- useful deletion requires modifying a large fraction of the backbone;
- preservation damage rises at approximately the same rate as deletion benefit;
- head-only editing performs equally well or better;
- validation acceptance is highly seed-specific and fails to reproduce.

The objective of the first study is mechanism feasibility, not benchmark
maximization.

---

### Conceptual contribution

The proposed mechanism can be summarized as:

> **Minibatch-Consensus Validation-Gated Sparse Relation Editing**

Its central idea is to replace one-shot global constrained-direction search
with repeated local evidence aggregation and bounded model intervention.

The method contains four essential components:

    minibatch relation attribution
              +
    cross-minibatch consensus
              +
    bounded sparse functional editing
              +
    deletion-preservation validation gate

The intended scientific question is not whether a large model can be forcibly
changed, but whether client-specific spurious relations expose stable,
repeatable functional support that can be edited incrementally under explicit
validation control.

---

### Current status and authorization

This section records a **proposed future research direction only**.

It does not modify, reinterpret, or supersede any locked result, experiment
protocol, input lock, authorization, or gate belonging to Experiments 01-11.

In particular:

- Experiment 10 / 10-01 results remain unchanged;
- Experiment 11 remains the currently locked mechanism-selection study;
- no new target, holdout, seed, replay, attribution analysis, sparse edit, or
  head-only experiment is authorized by this proposal;
- any real execution of this new direction requires a separately designed,
  reviewed, and locked experiment with its own contract, configuration,
  protocol lock, input lock, and design-owner gate.

No implementer may treat this proposal as execution authorization.