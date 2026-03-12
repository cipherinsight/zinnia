from zinnia import *


def test_mixed_chained_nested_arithmetic_ops():
    @zk_circuit
    def foo(shape_flag: Integer):
        rows = 1 if shape_flag > 0 else 2
        cols = 6 if shape_flag > 0 else 3
        dyn = np.eye(rows, cols, int)
        stat = np.asarray([3])

        expr = (dyn + stat) * 2
        assert expr.size == 6
        assert expr.sum() == dyn.sum() * 2 + 36

    assert foo(1)
    assert foo(0)


def test_mixed_compare_and_logical_ops():
    @zk_circuit
    def foo(shape_flag: Integer):
        rows = 1 if shape_flag > 0 else 2
        cols = 6 if shape_flag > 0 else 3
        dyn = np.eye(rows, cols, int)
        stat = np.asarray([1, 2, 3, 4, 5, 6])

        cmp_dyn = dyn > 0
        cmp_stat = stat > 3
        mask = np.logical_or(cmp_dyn, cmp_stat)

        # Union of {dynamic eye ones} and {indices >= 3} has 4 ones for both shapes.
        assert mask.sum() == 4

    assert foo(1)
    assert foo(0)


def test_mixed_filtering_static_by_dynamic_mask():
    @zk_circuit
    def foo(shape_flag: Integer):
        rows = 1 if shape_flag > 0 else 2
        cols = 6 if shape_flag > 0 else 3
        dyn = np.eye(rows, cols, int)
        stat = np.asarray([10, 20, 30, 40, 50, 60])

        picked = stat[dyn > 0]
        assert picked.sum() == (10 if shape_flag > 0 else 60)

    assert foo(1)
    assert foo(0)


def test_mixed_concat_and_stack_ops():
    @zk_circuit
    def foo(shape_flag: Integer):
        rows = 1 if shape_flag > 0 else 2
        cols = 6 if shape_flag > 0 else 3
        dyn = np.eye(rows, cols, int)
        stat = np.asarray([1, 2, 3, 4, 5, 6])

        joined = np.concatenate([dyn, stat], axis=0)
        stacked = np.stack([dyn, stat], axis=0)

        assert joined.sum() == dyn.sum() + stat.sum()
        assert stacked.sum() == dyn.sum() + stat.sum()

    assert foo(1)
    assert foo(0)


def test_mixed_control_flow_with_arithmetic_ops():
    @zk_circuit
    def foo(shape_flag: Integer, op_flag: Integer):
        rows = 1 if shape_flag > 0 else 2
        cols = 6 if shape_flag > 0 else 3
        dyn = np.eye(rows, cols, int)
        stat = np.asarray([10, 20, 30, 40, 50, 60])

        scale = 1 if op_flag > 0 else 2
        out = dyn + stat * scale
        expected = dyn.sum() + stat.sum() * scale

        assert out.sum() == expected

    assert foo(1, 1)
    assert foo(1, 0)
    assert foo(0, 1)
    assert foo(0, 0)


def test_mixed_nested_control_flow_and_logical_pipeline():
    @zk_circuit
    def foo(shape_flag: Integer, branch_flag: Integer):
        rows = 1 if shape_flag > 0 else 2
        cols = 6 if shape_flag > 0 else 3
        dyn = np.eye(rows, cols, int)
        stat = np.asarray([2])

        threshold = 0 if branch_flag > 0 else 2
        base = dyn + stat
        mask = np.logical_and(base > threshold, stat > 0)

        assert mask.sum() == (6 if branch_flag > 0 else (1 if shape_flag > 0 else 2))

    assert foo(1, 1)
    assert foo(1, 0)
    assert foo(0, 1)
    assert foo(0, 0)


def test_ds_like_rule_based_screening_pipeline():
    @zk_circuit
    def foo(shape_flag: Integer, policy_flag: Integer):
        rows = 1 if shape_flag > 0 else 2
        cols = 6 if shape_flag > 0 else 3
        dyn = np.eye(rows, cols, int)

        # DS-like feature pipeline: baseline + bump + policy-specific threshold.
        baseline = np.asarray([2])
        bump = np.asarray([3])
        score = (dyn + baseline) * bump
        threshold = 8 if policy_flag > 0 else 5
        alert = score > threshold

        # policy_flag>0 => only eye-positions exceed 8; else all exceed 6
        assert alert.sum() == (dyn.sum() if policy_flag > 0 else 6)

    assert foo(1, 1)
    assert foo(1, 0)
    assert foo(0, 1)
    assert foo(0, 0)


def test_ds_like_feature_stacking_and_fusion():
    @zk_circuit
    def foo(shape_flag: Integer):
        rows = 1 if shape_flag > 0 else 2
        cols = 6 if shape_flag > 0 else 3
        dyn = np.eye(rows, cols, int)

        # Simulate feature fusion from dynamic and static feature channels.
        static_channel = np.asarray([4])
        channel2 = dyn + static_channel
        fused_concat = np.concatenate([dyn, static_channel], axis=0)
        fused_stack = np.stack([dyn, channel2], axis=0)

        assert fused_concat.sum() == dyn.sum() + 4
        assert fused_stack.sum() == dyn.sum() + (dyn.sum() + 24)

    assert foo(1)
    assert foo(0)


def test_ds_like_mask_then_weighted_aggregation():
    @zk_circuit
    def foo(shape_flag: Integer):
        rows = 1 if shape_flag > 0 else 2
        cols = 6 if shape_flag > 0 else 3
        dyn = np.eye(rows, cols, int)

        # "records" vector and mask from dynamic signal.
        records = np.asarray([10, 20, 30, 40, 50, 60])
        kept = records[dyn > 0]

        weight = np.asarray([2])
        weighted = kept * weight
        assert weighted.sum() == (20 if shape_flag > 0 else 120)

    assert foo(1)
    assert foo(0)


def test_ds_like_branching_scorecards():
    @zk_circuit
    def foo(shape_flag: Integer, model_flag: Integer):
        rows = 1 if shape_flag > 0 else 2
        cols = 6 if shape_flag > 0 else 3
        dyn = np.eye(rows, cols, int)

        # Two scorecards selected by control-flow-equivalent scalar factor,
        # avoiding branch-local variable declarations.
        base = np.asarray([5])
        factor = 1 if model_flag > 0 else 2
        pred = (dyn + base) * np.asarray([factor])
        expected = (dyn.sum() + 30) * factor

        assert pred.sum() == expected

    assert foo(1, 1)
    assert foo(1, 0)
    assert foo(0, 1)
    assert foo(0, 0)


def test_ds_like_nested_boolean_composition():
    @zk_circuit
    def foo(shape_flag: Integer, hard_flag: Integer):
        rows = 1 if shape_flag > 0 else 2
        cols = 6 if shape_flag > 0 else 3
        dyn = np.eye(rows, cols, int)

        base = dyn + np.asarray([2])
        primary = base > (2 if hard_flag > 0 else 1)
        secondary = np.logical_or(dyn > 0, np.asarray([1]) > 0)
        final = np.logical_and(primary, secondary)

        # secondary is always true due to scalar true branch; final == primary.
        assert final.sum() == (dyn.sum() if hard_flag > 0 else 6)

    assert foo(1, 1)
    assert foo(1, 0)
    assert foo(0, 1)
    assert foo(0, 0)


def test_ds_like_multistage_risk_pipeline_with_filtering():
    @zk_circuit
    def foo(shape_flag: Integer):
        rows = 1 if shape_flag > 0 else 2
        cols = 6 if shape_flag > 0 else 3
        dyn = np.eye(rows, cols, int)

        records = np.asarray([10, 20, 30, 40, 50, 60])

        # Stage 1: nonlinear-ish score transform on dynamic feature.
        score = ((dyn + np.asarray([2])) * np.asarray([3])) - np.asarray([4])
        gate = score > np.asarray([3])

        # Stage 2: filter then re-weight selected records.
        kept = records[gate]
        weighted = kept * np.asarray([2]) + np.asarray([1])

        assert score.sum() > dyn.sum()
        assert weighted.sum() > kept.sum()
        assert kept.sum() <= records.sum()

    assert foo(1)
    assert foo(0)


def test_ds_like_dual_mask_fusion_and_aggregation():
    @zk_circuit
    def foo(shape_flag: Integer, strict_flag: Integer):
        rows = 1 if shape_flag > 0 else 2
        cols = 6 if shape_flag > 0 else 3
        dyn = np.eye(rows, cols, int)

        stat = np.asarray([1, 2, 3, 4, 5, 6])

        mask_a = (dyn + np.asarray([1])) > np.asarray([2])
        threshold = 4 if strict_flag > 0 else 2
        mask_b = stat > threshold
        fused = np.logical_or(mask_a, mask_b)

        agg = fused.sum() * np.asarray([3]) + mask_a.sum() * np.asarray([5]) + mask_b.sum() * np.asarray([7])

        assert fused.sum() >= mask_a.sum()
        assert fused.sum() >= mask_b.sum()
        assert agg > fused.sum()

    assert foo(1, 1)
    assert foo(0, 1)
    assert foo(1, 0)
    assert foo(0, 0)


def test_ds_like_concat_score_then_gate_pipeline():
    @zk_circuit
    def foo(shape_flag: Integer):
        rows = 1 if shape_flag > 0 else 2
        cols = 6 if shape_flag > 0 else 3
        dyn = np.eye(rows, cols, int)

        feature = dyn * np.asarray([3]) + np.asarray([1])
        joined = np.concatenate([feature, np.asarray([7])], axis=0)

        score = joined * np.asarray([2]) - np.asarray([1])
        hot = np.logical_and(score > np.asarray([5]), joined > np.asarray([0]))

        assert hot.sum() == dyn.sum() + 1
        assert score.sum() == (25 if shape_flag > 0 else 31)

    assert foo(1)
    assert foo(0)


def test_ds_like_three_stage_normalization_and_rules():
    @zk_circuit
    def foo(shape_flag: Integer):
        rows = 1 if shape_flag > 0 else 2
        cols = 6 if shape_flag > 0 else 3
        dyn = np.eye(rows, cols, int)

        stage1 = dyn + np.asarray([2])
        stage2 = stage1 * np.asarray([2])
        stage3 = np.concatenate([stage2, np.asarray([1])], axis=0)

        rules = np.logical_and(stage3 > np.asarray([3]), stage3 > np.asarray([0]))
        assert rules.sum() == 6

    assert foo(1)
    assert foo(0)
