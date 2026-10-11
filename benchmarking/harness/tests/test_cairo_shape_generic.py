"""Cairo shaped I/O and discovery, without running a prover."""
import argparse
import json
from pathlib import Path

import pytest

from benchmarking.baselines.cairo import cairo_baseline as cairo
from benchmarking.baselines.zkvm_common import check_outputs, sync_cases
from benchmarking.harness.adapters import get_adapter
from benchmarking.harness.case_sets import load_perf_cases
from benchmarking.harness.tests.test_shape_generic_cases import in_place_case
from benchmarking.harness.tests.test_protocol import fake_baselines, install_fake  # noqa: F401


def fixture(parameters, values):
    return {"parameters": parameters, "inputs": {"A": values}}


def test_shaped_tuple_complex_and_unused_encoding():
    f = fixture([
        {"name": "ignored", "kind": "scalar", "dtype": "int"},
        {"name": "pair", "kind": "tuple", "items": [
            {"kind": "array", "dtype": "complex", "shape": [1, 2]},
            {"kind": "scalar", "dtype": "bool"}]},
        {"name": "n", "kind": "scalar", "dtype": "int"}],
        {"ignored": 999, "pair": [[[[1.5, -2.0], [0., .25]]], True], "n": -3})
    words, shapes = cairo.encode_shaped_args(f, "A", ["ignored"])
    assert words == [3, 6, 6442450944, -8589934592, 0, 1073741824, 1, 2, 1, 1, 1, -3]
    assert shapes == [1, 2]


@pytest.mark.parametrize("value,shape", [([[1], [2, 3]], [2, 1]), ([[1, 2]], [2, 1]), ([2**63], [1])])
def test_reject_bad_shapes_or_word_domain(value, shape):
    with pytest.raises(cairo.Failure):
        cairo.encode_shaped_args(fixture([{"name": "x", "kind": "array", "dtype": "int", "shape": shape}],
                                         {"x": value}), "A", [])


def test_fixed_encoding_is_unchanged():
    f = fixture([{"name": "x", "kind": "array", "dtype": "float"},
                 {"name": "n", "kind": "scalar", "dtype": "int"}], {"x": [[.5, -1]], "n": 4})
    assert cairo.encode_args(f, "A", []) == [2, 2**31, -2**32, 4]


def test_decode_exact_words_shapes_tensor_complex_and_option():
    schema, rest = cairo.parse_type("(tensor<2,c32>,option<i64>)")
    assert not rest
    words = [2, 1, 1, 1, 1, 2**31, -2**32, 1, -7]
    report = {}
    cairo.decode_output(report, ("shaped", schema), [len(words)] + [w % cairo.STARK_PRIME for w in words])
    assert report == {"public_words": words, "input_shapes": [1, 1], "output_shapes": [[1, 1], []],
                      "publics": [.5, -1., -7]}
    schema, _ = cairo.parse_type("(tensor<2,q32>,option<i64>,())")
    report = {}
    cairo.decode_output(report, ("shaped", schema), [4, 0, 0, 3, 0])
    assert report["publics"] == []
    assert report["output_shapes"] == [[0, 3], [], []]


@pytest.mark.parametrize("raw", [[2, 0], [1, -1], [3, 5, 1, 1]])
def test_reject_bad_shape_envelope(raw):
    with pytest.raises(cairo.Failure):
        cairo.decode_output({}, ("shaped", "i64"), raw)


def test_headers_are_per_function(tmp_path):
    (tmp_path / "cairo").mkdir()
    (tmp_path / "cairo/a.cairo").write_text("// unused parameters: x\n// output: i64\n")
    (tmp_path / "cairo/b.cairo").write_text("// unused parameters: y, z\n// output: tensor<2,q32>\n")
    assert cairo.unused_params(tmp_path, "a") == ["x"]
    assert cairo.unused_params(tmp_path, "b") == ["y", "z"]
    assert cairo.output_schema(tmp_path, "a") == "i64"
    assert cairo.output_schema(tmp_path, "b") == ("tensor", (2, "q32"))


def test_sync_adds_only_available_entries_and_removes_stale_links(tmp_path, monkeypatch):
    baseline = tmp_path / "baselines"
    package = baseline / "cairo"
    package.mkdir(parents=True)
    (package / "Scarb.toml.in").write_text("# @EXECUTABLES@\n")
    shaped = tmp_path / "shaped"
    prog = shaped / "prog"
    (prog / "cairo").mkdir(parents=True)
    (prog / "zkvm.rs").write_text("pub mod a {\n}\npub mod b {\n}\n")
    (prog / "cairo/b.cairo").write_text("// output: i64\n")
    monkeypatch.setattr(sync_cases, "BASELINES", baseline)
    monkeypatch.setattr(sync_cases, "FIXED_ROOTS", [])
    monkeypatch.setattr(sync_cases, "SHAPED_ROOTS", [shaped])
    sync_cases.sync_cairo()
    link = package / "src/cases/prog__b.cairo"
    assert link.resolve() == prog / "cairo/b.cairo"
    assert 'name = "prog__b"' in (package / "Scarb.toml").read_text()
    assert 'name = "prog__a"' not in (package / "Scarb.toml").read_text()
    (prog / "cairo/b.cairo").unlink()
    sync_cases.sync_cairo()
    assert not link.is_symlink()


def test_harness_uses_entry_source_and_passes_function(tmp_path, fake_baselines):
    install_fake(fake_baselines, "cairo", "honest")
    case = in_place_case(tmp_path)
    adapter = get_adapter("cairo")
    assert "cairo/scale.cairo" in adapter.missing_program(case)
    (case.directory / "cairo").mkdir()
    source = case.directory / "cairo/scale.cairo"
    source.write_text("// outputs: mutated a\n// output: tensor<1,q32>\n")
    assert adapter.missing_program(case) is None
    assert source in adapter.program_files(case)
    ctx = argparse.Namespace(case=case)
    argv = adapter._argv(ctx, "execute", "A", tmp_path, tmp_path / "out.json", [])
    assert argv[argv.index("--function") + 1] == "scale"
    assert argv[argv.index("--fixture") + 1] == str(case.fixture_path)
    # The other entry is unavailable even though this folder has one Cairo file.
    assert not check_outputs.Case("prog/other", case.directory, "other").has_program("cairo")
    assert check_outputs.Case("prog/scale", case.directory, "scale").has_program("cairo")


def test_real_perf_set_pilots_are_available():
    root = Path(__file__).resolve().parents[2]
    cases = load_perf_cases(root / "datasets/case_sets/npbench_pythran_perf.json", None, None)
    adapter = get_adapter("cairo")
    wanted = {"npbench_crc16_crc16__crc16", "npbench_polybench_gemm_gemm__kernel",
              "pythran_bubble_sort__bubble_sort", "pythran_log_likelihood__log_likelihood"}
    available = {c.case for c in cases if adapter.missing_program(c) is None}
    assert wanted <= available


def test_execute_selects_function_fixture_and_checks_shape_commitment(tmp_path, monkeypatch):
    (tmp_path / "cairo").mkdir()
    (tmp_path / "cairo/f.cairo").write_text("// output: i64\n")
    f = fixture([{"name": "x", "kind": "array", "dtype": "int", "shape": [2]}], {"x": [4, 5]})
    (tmp_path / "fixture.f.json").write_text(json.dumps(f))
    args = argparse.Namespace(case_dir=tmp_path, function="f", fixture=None, input="A", mode="execute",
                              work_dir=tmp_path / "work")
    seen = []
    def execute(report, case, case_dir, args_path, schema, keep_trace=False):
        seen.append((case, json.loads(args_path.read_text())))
        report["input_shapes"] = [3]
        return tmp_path / "execution"
    monkeypatch.setattr(cairo, "execute", execute)
    with pytest.raises(cairo.Failure, match="committed input shapes"):
        cairo.run(args, {"timings_s": {}})
    assert seen == [(tmp_path.name + "__f", ["0x1", "0x3", "0x4", "0x5", "0x2"])]


def test_mutated_declaration_must_match_reference(tmp_path):
    (tmp_path / "cairo").mkdir()
    (tmp_path / "cairo/f.cairo").write_text("// outputs: mutated B\n// output: tensor<1,i64>\n")
    case = check_outputs.Case("prog/f", tmp_path, "f", ["A"])
    with pytest.raises(ValueError, match="mutated outputs differ"):
        check_outputs.run_system("cairo", case, tmp_path / "fixture.json", "A", 1)


def test_setup_builds_only_the_selected_entry(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cairo, "scarb", lambda *a: calls.append(a))
    args = argparse.Namespace(case_dir=tmp_path, function="f", work_dir=tmp_path / "work",
                              input="A", mode="setup")
    cairo.run(args, {"timings_s": {}})
    assert calls[0][1:] == ("build", "build", "--target-names", tmp_path.name + "__f")


def test_verify_rejects_wrong_fixture_shapes_before_invoking_verifier(tmp_path, monkeypatch):
    (tmp_path / "cairo").mkdir()
    (tmp_path / "cairo/f.cairo").write_text("// output: i64\n")
    f = fixture([{"name": "x", "kind": "array", "dtype": "int", "shape": [2]}], {"x": [4, 5]})
    (tmp_path / "fixture.f.json").write_text(json.dumps(f))
    work = tmp_path / "work"
    work.mkdir()
    # Synthetic data for testing the decoder, not a generated proof.
    (work / "proof-stwo-A.json").write_text("{}")
    monkeypatch.setattr(cairo, "report_proof", lambda *a: None)
    monkeypatch.setattr(cairo, "proof_output", lambda proof: [1, 6, 0, 3, 1, 3, 9])
    monkeypatch.setattr(cairo, "scarb", lambda *a: pytest.fail("must reject mismatched shapes first"))
    args = argparse.Namespace(case_dir=tmp_path, function="f", fixture=None, input="A", mode="verify",
                              work_dir=work, tamper=False)
    with pytest.raises(cairo.Failure, match="committed input shapes"):
        cairo.run(args, {"timings_s": {}})


def test_a_result_tuple_with_arrays_reports_one_shape_per_output():
    schema, _ = cairo.parse_type("(tensor<1,q32>,i64)")
    words = [0, 2, 2**32, 2**31, 7]
    report = {}
    cairo.decode_output(report, ("shaped", schema), [len(words)] + [w % cairo.STARK_PRIME for w in words])
    assert report["output_shapes"] == [[2], []] and report["publics"] == [1., .5, 7]
    schema, _ = cairo.parse_type("(c32,q32)")
    report = {}
    cairo.decode_output(report, ("shaped", schema), [4, 0, 0, 0, 0])
    assert "output_shapes" not in report and "output_shape" not in report


def test_a_stacked_array_matches_a_list_of_rows_only_when_it_stacks_them():
    from benchmarking.harness.outputs import _same_shapes
    assert _same_shapes([[3, 2]], [[2], [2], [2]])
    assert not _same_shapes([[2, 3]], [[2], [2], [2]])
    assert not _same_shapes([[3, 2]], [[2], [2], [3]])
    assert not _same_shapes([[6]], [[2], [2], [2]])
