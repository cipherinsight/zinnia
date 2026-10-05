"""Compiled output extraction must never execute participant Python source."""
import builtins
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .. import worker
from ..execution import Runner


class WorkerTests(unittest.TestCase):
    def run_source(self, source, x=None):
        with tempfile.TemporaryDirectory() as runtime:
            root = Path(runtime)
            (root/'submission.py').write_text(source)
            (root/'request.json').write_text(json.dumps({'arguments':['x'], 'cases':[{'x':x or [2,3]}]}))
            original_exec = builtins.exec
            def forbid_participant_exec(code, *args, **kwargs):
                if getattr(code, 'co_filename', '') == 'submission.py':
                    raise AssertionError('Participant source reached the Python runtime')
                return original_exec(code, *args, **kwargs)
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch('builtins.exec', side_effect=forbid_participant_exec):
                    worker.main()
            finally:
                os.chdir(previous)
            return json.loads((root/'result.json').read_text())

    def test_native_participant_execution_is_never_used(self):
        result = self.run_source('def kernel(x):\n    return x + 7\n')
        self.assertEqual(result['status'], 'executed', result)
        self.assertEqual(result['outputs'], [{'shape':[2], 'values':[9,10]}])

    def test_constants_mixed_cells_shapes_and_early_returns(self):
        programs = [
            ('return np.array([[7, 0], [x[0], -3]])', [2,2], [7,0,2,-3]),
            ('return np.zeros((2, 3), dtype=int)', [2,3], [0]*6),
            ('return np.array([x[0] * 1.0, 4.0])', [2], [2,4]),
            ('if x[0] > 0:\n        return x + 1\n    return x - 1', [2], [3,4]),
        ]
        for body, shape, values in programs:
            with self.subTest(body=body):
                result = self.run_source('def kernel(x):\n    '+body+'\n')
                self.assertEqual(result['status'], 'executed', result)
                self.assertEqual(result['outputs'], [{'shape':shape,'values':values}])

    def test_fractional_return_is_rejected(self):
        result = self.run_source('def kernel(x):\n    return np.array([1.5, 2.0])\n')
        self.assertEqual(result['status'], 'compile_error', result)

    def test_native_rust_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as runtime:
            with self.assertRaisesRegex(ValueError, 'native execution is disabled'):
                Runner(runtime, rust_mode='native')
