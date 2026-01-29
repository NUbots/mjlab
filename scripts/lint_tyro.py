"""Lint script to ensure all tyro.cli() calls include mjlab.TYRO_FLAGS."""

import ast
import sys
from pathlib import Path


def check_file(filepath: Path) -> list[tuple[int, bool]]:
  """Check a file for tyro.cli() calls.

  Returns a list of (line_number, has_tyro_flags) tuples.
  """
  try:
    source = filepath.read_text()
    tree = ast.parse(source, filename=str(filepath))
  except SyntaxError:
    return []

  results = []

  for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
      continue

    # Check if this is a tyro.cli() call
    func = node.func
    is_tyro_cli = False

    if isinstance(func, ast.Attribute) and func.attr == "cli":
      # tyro.cli(...)
      if isinstance(func.value, ast.Name) and func.value.id == "tyro":
        is_tyro_cli = True
    elif isinstance(func, ast.Name) and func.id == "cli":
      # from tyro import cli; cli(...)
      is_tyro_cli = True

    if not is_tyro_cli:
      continue

    # Check if config= keyword argument contains TYRO_FLAGS
    has_tyro_flags = False
    for keyword in node.keywords:
      if keyword.arg == "config":
        # Check if the value contains TYRO_FLAGS
        config_source = ast.unparse(keyword.value)
        if "TYRO_FLAGS" in config_source:
          has_tyro_flags = True
          break

    results.append((node.lineno, has_tyro_flags))

  return results


def main() -> int:
  """Run the linter on all Python files in src/ and scripts/."""
  root = Path(__file__).parent.parent
  search_dirs = [root / "src", root / "scripts"]

  all_results: list[tuple[Path, int, bool]] = []

  for search_dir in search_dirs:
    if not search_dir.exists():
      continue
    for filepath in search_dir.rglob("*.py"):
      # Skip this lint script itself
      if filepath.name == "lint_tyro.py":
        continue
      results = check_file(filepath)
      for line, has_flags in results:
        all_results.append((filepath, line, has_flags))

  if not all_results:
    print("No tyro.cli() calls found.")
    return 0

  print("tyro.cli() calls:")
  has_violations = False
  for filepath, line, has_flags in sorted(all_results):
    rel_path = filepath.relative_to(root)
    status = "✓" if has_flags else "✗"
    print(f"  {status} {rel_path}:{line}")
    if not has_flags:
      has_violations = True

  print()
  if has_violations:
    print("Some tyro.cli() calls are missing config=mjlab.TYRO_FLAGS")
    return 1

  print("All tyro.cli() calls include TYRO_FLAGS.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
