"""Verify the exact manual examples in both participant execution engines."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..execution import Runner


class ManualTests(unittest.TestCase):
    def test_examples_match_documented_outputs(self):
        entries = json.loads((Path(__file__).parents[1] / 'static/manual.json').read_text())
        with tempfile.TemporaryDirectory() as runtime:
            runner = Runner(runtime)
            try:
                for entry in entries:
                    for condition in ('zinnia', 'rust'):
                        with self.subTest(topic=entry['id'], condition=condition):
                            snippet = entry['code'][condition]
                            if condition == 'zinnia':
                                source = 'def kernel(x, y):\n' + '\n'.join('    '+line for line in snippet.splitlines()) + '\n    return out\n'
                            else:
                                source = 'fn kernel(x: &Array1<i64>, y: &Array1<i64>) -> Array1<i64> {\n' + snippet + '\nout\n}'
                            with patch('studies.user_study.site.content.answer', return_value=entry['expected']):
                                result = runner.execute(condition, 'practice', source, [{'x':[0], 'y':[0]}])
                            self.assertEqual(result['status'], 'passed', result)
            finally:
                runner.pool.shutdown()
