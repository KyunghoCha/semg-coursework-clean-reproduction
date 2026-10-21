#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/semg-coursework-matplotlib}"
SEMG_PYTHON="${SEMG_PYTHON:-python3}"
phase="${1:-test}"
task_output="${SEMG_RUN_DIR:-$PWD}"
for stage in diagnostic_v2 mechanisms_v3 capacity_v4 geometry_probe latent_subclass_v5 cross_spectrum_v6 cross_capacity_v7 fine_spectrum_v8 spectral_controls_v9 nested_selection_v10 line_projection_v11 reference_phase_probe record_reference_control probability_blend_v12 forest_transfer_v13 fit_gap_probe cost_benchmark; do
  mkdir -p "$task_output/private/$stage" "$task_output/results/$stage"
done
run_python() { "$SEMG_PYTHON" "$@"; }
case "$phase" in
  test) run_python -m pytest -q tests experiments/diagnostic_v2/test_features_v2.py experiments/diagnostic_v2/test_causal_context.py experiments/mechanisms_v3/test_features_v3.py experiments/latent_subclass_v5/test_collapse.py experiments/cross_spectrum_v6/test_cross.py experiments/fine_spectrum_v8/test_fine.py experiments/cost_benchmark/test_fast.py experiments/spectral_controls_v9/test_design.py experiments/nested_selection_v10/test_selection.py experiments/line_projection_v11/test_projection.py experiments/record_reference_control/test_splits.py experiments/probability_blend_v12/test_blending.py experiments/forest_transfer_v13/test_forest_selection.py experiments/gain_invariance_v14/test_features.py experiments/ar_residual_v15/test_ar_features.py experiments/factorized_targets_v16/test_factorization.py ;;
  features)
    run_python experiments/diagnostic_v2/features_v2.py
    run_python experiments/mechanisms_v3/features_v3.py
    run_python experiments/cross_spectrum_v6/features_cross.py
    run_python experiments/fine_spectrum_v8/features_fine.py ;;
  diagnose)
    run_python experiments/diagnostic_v2/diagnose.py
    run_python experiments/diagnostic_v2/spectrum_diagnostic.py
    run_python experiments/geometry_probe/describe.py ;;
  nested)
    run_python experiments/diagnostic_v2/run_nested.py
    run_python experiments/diagnostic_v2/causal_context.py
    run_python experiments/diagnostic_v2/verify_and_report.py
    run_python experiments/diagnostic_v2/group_checks.py ;;
  mechanisms)
    run_python experiments/mechanisms_v3/run.py
    run_python experiments/mechanisms_v3/verify.py ;;
  capacity|subclass|cross|cross-capacity|fine-spectrum|spectral-controls|nested-selection|line-projection)
    case "$phase" in
      capacity) stage=capacity_v4 ;;
      subclass) stage=latent_subclass_v5 ;;
      cross) stage=cross_spectrum_v6 ;;
      cross-capacity) stage=cross_capacity_v7 ;;
      fine-spectrum) stage=fine_spectrum_v8 ;;
      spectral-controls) stage=spectral_controls_v9 ;;
      nested-selection) stage=nested_selection_v10 ;;
      line-projection) stage=line_projection_v11 ;;
    esac
    if [[ "$phase" == line-projection && ! -f "$task_output/private/line_projection_v11/features.npy" ]]; then run_python experiments/line_projection_v11/features.py; fi
    run_python "experiments/$stage/run.py" "${@:2}"
    run_python "experiments/$stage/verify.py"
    if [[ "$phase" == cross-capacity ]]; then run_python experiments/cross_capacity_v7/report.py; fi
    if [[ "$phase" == fine-spectrum ]]; then run_python experiments/fine_spectrum_v8/report.py; run_python experiments/fine_spectrum_v8/diagnose.py; fi
    if [[ "$phase" == spectral-controls ]]; then run_python experiments/spectral_controls_v9/report.py; fi
    if [[ "$phase" == nested-selection ]]; then run_python experiments/nested_selection_v10/report.py; fi
    if [[ "$phase" == line-projection ]]; then run_python experiments/line_projection_v11/report.py; fi
    if [[ "$phase" == subclass ]]; then run_python experiments/latent_subclass_v5/report.py; fi
    if [[ "$phase" == cross ]]; then run_python experiments/cross_spectrum_v6/diagnose.py; run_python experiments/cross_spectrum_v6/report.py; fi ;;
  fit-gap)
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 run_python experiments/fit_gap_probe/run.py
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 run_python experiments/fit_gap_probe/verify.py ;;
  forest-transfer)
    run_python experiments/forest_transfer_v13/run.py "${@:2}"
    run_python experiments/forest_transfer_v13/verify.py ;;
  factorized-targets)
    run_python experiments/factorized_targets_v16/run.py
    run_python experiments/factorized_targets_v16/verify.py
    run_python experiments/factorized_targets_v16/cost.py
    run_python experiments/factorized_targets_v16/report.py ;;
  ar-residual)
    run_python experiments/ar_residual_v15/ar_features.py
    run_python experiments/ar_residual_v15/run.py
    run_python experiments/ar_residual_v15/verify.py
    run_python experiments/ar_residual_v15/cost.py
    run_python experiments/ar_residual_v15/report.py ;;
  probability-blend)
    run_python experiments/probability_blend_v12/run.py "${@:2}"
    run_python experiments/probability_blend_v12/verify.py ;;
  record-reference)
    run_python experiments/record_reference_control/run.py "${@:2}"
    run_python experiments/record_reference_control/verify.py
    run_python experiments/record_reference_control/report.py
    run_python experiments/record_reference_control/label_support.py ;;
  reference-phase)
    run_python experiments/reference_phase_probe/run.py "${SEMG_RAW:?Set SEMG_RAW to the original CSV directory}"
    run_python experiments/reference_phase_probe/verify.py "$SEMG_RAW" ;;
  benchmark)
    run_python experiments/cost_benchmark/refit.py
    run_python experiments/cost_benchmark/run_all.py
    run_python experiments/cost_benchmark/inventory.py
    run_python experiments/cost_benchmark/report.py ;;
  verify)
    run_python experiments/diagnostic_v2/verify_and_report.py
    run_python experiments/mechanisms_v3/verify.py
    run_python experiments/capacity_v4/verify.py
    run_python experiments/latent_subclass_v5/verify.py
    run_python experiments/latent_subclass_v5/report.py
    run_python experiments/cross_spectrum_v6/verify.py
    run_python experiments/cross_capacity_v7/verify.py
    run_python experiments/fine_spectrum_v8/verify.py
    run_python experiments/spectral_controls_v9/verify.py
    run_python experiments/nested_selection_v10/verify.py
    run_python experiments/line_projection_v11/verify.py
    run_python experiments/record_reference_control/verify.py ;;
  *) echo 'Usage: bash run_extensions.sh {test|features|diagnose|nested|mechanisms|capacity|subclass|cross|cross-capacity|fine-spectrum|spectral-controls|nested-selection|line-projection|reference-phase|record-reference|probability-blend|forest-transfer|ar-residual|factorized-targets|fit-gap|benchmark|verify} [--resume]' >&2; exit 2 ;;
esac
