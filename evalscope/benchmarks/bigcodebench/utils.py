"""Build an explicit runner for the test classes supplied by BigCodeBench."""

import ast

EMPTY_SUITE_ERROR = 'EVALSCOPE_BIGCODEBENCH_EMPTY_TEST_SUITE'


def build_test_program(solution: str, test: str, marker: str) -> str:
    """Run the dataset's unittest classes and mark only a completed, successful suite."""
    classes = [node.name for node in ast.parse(test).body if isinstance(node, ast.ClassDef)]
    if not classes:
        raise ValueError('BigCodeBench test data contains no test classes.')

    # Load only dataset classes; generated code may define unrelated test classes.
    runner = f"""
import unittest as _evalscope_unittest
_evalscope_suite = _evalscope_unittest.TestSuite()
for _evalscope_name in {classes!r}:
    _evalscope_class = globals()[_evalscope_name]
    if issubclass(_evalscope_class, _evalscope_unittest.TestCase):
        _evalscope_suite.addTests(_evalscope_unittest.defaultTestLoader.loadTestsFromTestCase(_evalscope_class))
if _evalscope_suite.countTestCases() == 0:
    raise RuntimeError({EMPTY_SUITE_ERROR!r})
_evalscope_result = _evalscope_unittest.TextTestRunner(verbosity=1).run(_evalscope_suite)
if not _evalscope_result.wasSuccessful():
    raise AssertionError('BigCodeBench tests failed')
print('\\n' + {marker!r})
"""
    return solution + '\n' + test + '\n' + runner
