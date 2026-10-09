from pathlib import Path

from benchmarking.baselines.hw_check import noir_driver as driver


def test_stats_never_keygens_or_mutates_existing_setup(tmp_path, monkeypatch):
    case = tmp_path / 'case'
    case.mkdir()
    (case / 'main.f.nr').write_text('// @outputs: int[1]\nfn main(x: Field) -> pub Field { x }')
    work = tmp_path / 'work'
    (work / 'pkg').mkdir(parents=True)
    (work / 'pkg/Nargo.toml').write_text('existing setup')
    (work / 'keys').mkdir()
    (work / 'keys/vk').write_text('existing key')
    calls = []
    monkeypatch.setattr(driver, 'compile_package', lambda pkg: calls.append(('compile', pkg)))
    def run(cmd, pkg):
        calls.append((cmd, pkg))
        assert cmd[1] == 'gates'
        assert '--scheme' in cmd and 'ultra_honk' in cmd
        return 'backend log\n{"functions": [{"circuit_size": 123, "acir_opcodes": 45}]}'
    monkeypatch.setattr(driver, '_run', run)
    result = driver.stats(case, 'f', work, driver.DEFAULT_TARGET, {})
    assert result == {'gates': 123, 'acir_opcodes': 45}
    assert len(calls) == 2
    assert calls[0][1] == work / 'stats/pkg'
    assert (work / 'pkg/Nargo.toml').read_text() == 'existing setup'
    assert (work / 'keys/vk').read_text() == 'existing key'
