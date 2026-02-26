"""
╔══════════════════════════════════════════════════════════╗
║  SETUP VERIFICATION SCRIPT                               ║
║  Run this to verify everything is installed correctly    ║
║                                                          ║
║  Usage: python scripts/verify_setup.py                   ║
╚══════════════════════════════════════════════════════════╝
"""

import sys
import os
import shutil

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def check_pass(name):
    print(f"  [PASS] {name}")
    return True

def check_fail(name, error=""):
    print(f"  [FAIL] {name}")
    if error:
        print(f"    Error: {error}")
    return False

def main():
    print("\n" + "="*60)
    print("  AIRFOIL GENERATOR — SETUP VERIFICATION")
    print("="*60)

    all_passed = True
    results = {}

    # ─── CHECK 1: Python Version ───
    print_header("CHECK 1: Python Version")
    v = sys.version_info
    if v.major == 3 and v.minor in [9, 10, 11]:
        check_pass(f"Python {v.major}.{v.minor}.{v.micro}")
        results['python'] = True
    else:
        check_fail(f"Python {v.major}.{v.minor}.{v.micro}",
                   "Need Python 3.9, 3.10, or 3.11")
        results['python'] = False
        all_passed = False

    # ─── CHECK 2: Virtual Environment ───
    print_header("CHECK 2: Virtual Environment")
    if hasattr(sys, 'prefix') and sys.prefix != sys.base_prefix:
        check_pass(f"Virtual env active: {sys.prefix}")
        results['venv'] = True
    else:
        check_fail("Virtual environment NOT active",
                   "Run: source venv/bin/activate (Linux/Mac) or "
                   "venv\\Scripts\\activate (Windows)")
        results['venv'] = False
        all_passed = False

    # ─── CHECK 3: Core Packages ───
    print_header("CHECK 3: Core Packages")

    packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'h5py': 'h5py',
        'yaml': 'pyyaml',
        'tqdm': 'tqdm',
        'joblib': 'joblib',
    }

    for import_name, pip_name in packages.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, '__version__', 'unknown')
            check_pass(f"{pip_name} ({version})")
            results[pip_name] = True
        except ImportError:
            check_fail(f"{pip_name}", f"pip install {pip_name}")
            results[pip_name] = False
            all_passed = False

    # ─── CHECK 4: PyTorch ───
    print_header("CHECK 4: PyTorch")
    try:
        import torch
        check_pass(f"PyTorch {torch.__version__}")
        results['torch'] = True

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            check_pass(f"CUDA available: {gpu_name}")
            check_pass(f"CUDA version: {torch.version.cuda}")
            results['cuda'] = True
        else:
            check_pass("CUDA not available (will use CPU — this is OK)")
            results['cuda'] = False

        # Test basic operation
        x = torch.randn(10, 10)
        y = torch.mm(x, x.t())
        check_pass("PyTorch basic operations work")

    except ImportError:
        check_fail("PyTorch NOT installed",
                   "See installation instructions in README")
        results['torch'] = False
        all_passed = False

    # ─── CHECK 5: Visualization ───
    print_header("CHECK 5: Visualization")
    viz_packages = {
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
    }
    for import_name, pip_name in viz_packages.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, '__version__', 'unknown')
            check_pass(f"{pip_name} ({version})")
            results[pip_name] = True
        except ImportError:
            check_fail(f"{pip_name}", f"pip install {pip_name}")
            results[pip_name] = False
            all_passed = False

    # ─── CHECK 6: XFOIL ───
    print_header("CHECK 6: XFOIL")
    
    # Check if xfoil executable is in PATH
    xfoil_path = shutil.which("xfoil")
    if xfoil_path:
        check_pass(f"found 'xfoil' executable: {xfoil_path}")
        results['xfoil'] = True
    else:
        check_fail("'xfoil' command NOT found in PATH",
                   "Add XFOIL directory to system PATH")
        results['xfoil'] = False
        print("    NOTE: XFOIL is required for data generation")

    # ─── CHECK 7: Directory Structure ───
    print_header("CHECK 7: Directory Structure")
    required_dirs = [
        'data/raw',
        'data/processed',
        'data/generated_designs',
        'src/data',
        'src/geometry',
        'src/models',
        'src/training',
        'src/evaluation',
        'src/utils',
        'notebooks',
        'scripts',
        'tests',
        'checkpoints',
        'results/figures',
        'results/metrics',
        'logs',
        'docs',
    ]

    for d in required_dirs:
        if os.path.isdir(d):
            check_pass(f"Directory: {d}")
        else:
            check_fail(f"Directory: {d}", f"mkdir -p {d}")
            all_passed = False

    # ─── CHECK 8: Config File ───
    print_header("CHECK 8: Configuration")
    if os.path.isfile('config.yaml'):
        try:
            import yaml
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            check_pass("config.yaml exists and is valid YAML")
            check_pass(f"Project: {config.get('project', {}).get('name', 'unknown')}")
            results['config'] = True
        except Exception as e:
            check_fail("config.yaml has errors", str(e))
            results['config'] = False
            all_passed = False
    else:
        check_fail("config.yaml not found", "Create config.yaml in project root")
        results['config'] = False
        all_passed = False

    # ─── CHECK 9: Test Imports ───
    print_header("CHECK 9: Project Imports")
    try:
        sys.path.insert(0, os.getcwd())
        import src
        check_pass("src package importable")
        results['src'] = True
    except ImportError as e:
        check_fail("src package", str(e))
        results['src'] = False
        all_passed = False

    # ─── CHECK 10: Disk Space ───
    print_header("CHECK 10: Disk Space")
    try:
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        if free_gb > 10:
            check_pass(f"Free disk space: {free_gb:.1f} GB")
        elif free_gb > 5:
            check_pass(f"Free disk space: {free_gb:.1f} GB (minimum, may need more)")
        else:
            check_fail(f"Free disk space: {free_gb:.1f} GB",
                       "Need at least 10 GB free")
            all_passed = False
    except Exception:
        check_pass("Could not check disk space (non-critical)")

    # ─── FINAL REPORT ───
    print_header("FINAL REPORT")

    total_checks = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total_checks - passed

    print(f"\n  Total checks:  {total_checks}")
    print(f"  Passed:        {passed}")
    print(f"  Failed:        {failed}")

    if all_passed:
        print(f"\n  ========================================")
        print(f"  ||  ALL CHECKS PASSED! [PASS]           ||")
        print(f"  ||  Ready to start building.            ||")
        print(f"  ========================================")
    else:
        print(f"\n  ========================================")
        print(f"  ||  SOME CHECKS FAILED [FAIL]           ||")
        print(f"  ||  Fix the issues above before         ||")
        print(f"  ||  proceeding to code.                 ||")
        print(f"  ========================================")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)