"""Dependency-free test runner for roman_features/tests/.

The shared venv has no pytest and is deliberately left untouched. This runner
discovers every `test_*` callable in the test modules, runs it, and reports
pass/fail with tracebacks. Usage (from repo root):

    .venv/bin/python roman_features/tests/run_tests.py [module_name ...]

Exit code 0 iff every test passes.
"""
import importlib
import sys
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # repo src


def main():
    modules = sys.argv[1:] or ["test_feature_builder", "test_isolation"]
    n_pass = n_fail = 0
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            print(f"[ERROR] importing {mod_name}")
            traceback.print_exc()
            n_fail += 1
            continue
        tests = [(n, f) for n, f in sorted(vars(mod).items())
                 if n.startswith("test_") and callable(f)]
        print(f"=== {mod_name} ({len(tests)} tests) ===")
        for name, fn in tests:
            try:
                fn()
                print(f"  PASS {name}")
                n_pass += 1
            except Exception:
                print(f"  FAIL {name}")
                traceback.print_exc()
                n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
