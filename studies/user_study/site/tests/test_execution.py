import tempfile
import unittest
from pathlib import Path

from ..content import ORDER, cases
from ..execution import Runner
from .reference import PYTHON, RUST


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.runner = Runner(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.runner.pool.shutdown()
        cls.temp.cleanup()

    def test_all_tasks_both_real_engines(self):
        for language, programs in [('zinnia', PYTHON), ('rust', RUST)]:
            for task in ['practice', *ORDER]:
                with self.subTest(language=language, task=task):
                    inputs = cases(task) + (cases(task, hidden=True) if task != 'practice' else [])
                    result = self.runner.execute(language, task, programs[task], inputs)
                    self.assertEqual(result['status'], 'passed', result)
                    self.assertEqual(len(result['cases']), len(inputs))
                    self.assertEqual(result['engine'], 'risc0-zkvm' if language == 'rust' else 'zinnia-ir-mock')

    def test_wrong_answers_and_diagnostics(self):
        for language, source in [('zinnia', 'def kernel(x, y):\n    return x\n'), ('rust', 'fn kernel(x: &Array1<i64>, y: &Array1<i64>) -> Array1<i64> { x.clone() }')]:
            with self.subTest(language=language):
                result = self.runner.execute(language, 'practice', source, cases('practice'))
                self.assertEqual(result['status'], 'failed', result)
                self.assertFalse(result['cases'][0]['passed'])
                self.assertNotEqual(result['cases'][0]['actual'], result['cases'][0]['expected'])
                broken = self.runner.execute(language, 'practice', 'this is not valid source!', cases('practice'))
                self.assertEqual(broken['status'], 'compile_error', broken)
                self.assertTrue(broken['stderr'])
        panic = self.runner.execute('rust', 'practice', 'fn kernel(x: &Array1<i64>, y: &Array1<i64>) -> Array1<i64> { panic!("debug me") }', cases('practice'))
        self.assertEqual(panic['status'], 'runtime_error', panic)
        printed = self.runner.execute('rust', 'practice', RUST['practice'].replace('x + y', 'println!("hello debug"); x + y'), cases('practice'))
        self.assertEqual(printed['status'], 'passed', printed)
        self.assertIn('hello debug', printed['stdout'] + printed['stderr'])

    def test_zinnia_source_module_helpers(self):
        source = '''import numpy as np
def add(a, b):
    return a + b
def kernel(x, y):
    def apply(row):
        return add(row, y)
    return apply(x)
'''
        result = self.runner.execute('zinnia', 'practice', source, cases('practice'))
        self.assertEqual(result['status'], 'passed', result)

    def test_sandbox_files_network_timeout_and_output_limits(self):
        with tempfile.TemporaryDirectory(dir=self.runner.root) as directory:
            code = '''import os, socket
assert not os.path.exists('/home')
assert not os.path.exists('/opt/zinnia/studies')
assert not os.environ.get('STUDY_ADMIN_KEY')
s = socket.socket()
s.settimeout(.2)
try:
    s.connect(('1.1.1.1', 53))
    raise AssertionError('Network should be isolated')
except OSError:
    pass
print('isolated')
'''
            result = self.runner.sandbox(Path(directory), ['/usr/bin/python3.14', '-c', code])
            self.assertEqual(result[0], 0, result)
            self.assertIn('isolated', result[1])
            result = self.runner.sandbox(Path(directory), ['/usr/bin/python3.14', '-c', 'print("x"*100000)'])
            self.assertLessEqual(len(result[1]), 16000)
            old = self.runner.timeout
            self.runner.timeout = .5
            try:
                with self.assertRaises(TimeoutError):
                    self.runner.sandbox(Path(directory), ['/usr/bin/python3.14', '-c', 'while True: pass'])
            finally:
                self.runner.timeout = old
