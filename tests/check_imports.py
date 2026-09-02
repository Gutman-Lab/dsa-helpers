#!/usr/bin/env python3
"""Smoke-test that dsa_helpers modules import and lazy names resolve.

Importing a module is not enough after switching to lazy-loader: `lazy.attach`
and `lazy.load` only fail when the name is first used. Sharing one Python
process is also not enough: if `ml.callbacks` already loaded torch, a later
`lazy.load("torch")` can look fine even though a fresh process RecursionErrors.

Each module is therefore imported in its own subprocess.

Usage, against the copy of the package currently on sys.path:

    python tests/check_imports.py

To test a built wheel the way a PyPI user will see it:

    python -m venv /tmp/dsa-helpers-smoke
    source /tmp/dsa-helpers-smoke/bin/activate
    pip install 'dist/dsa_helpers-*.whl[all]'
    python tests/check_imports.py
"""

from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import subprocess
import sys
import traceback

# Not in core pyproject dependencies; missing is SKIP, not FAIL.
OPTIONAL_MODULES = {
    "cv2",
    "torch",
    "transformers",
    "torchvision",
    "ultralytics",
    "albumentations",
    "datasets",
    "large_image",
    "large_image_source_openslide",
}

LAZY_TYPE_NAMES = {"_LazyModule", "DelayedImportErrorModule"}


def _is_lazy_proxy(obj: object) -> bool:
    return type(obj).__name__ in LAZY_TYPE_NAMES


def _lazy_name(obj: object) -> str:
    # Do not use getattr: _LazyModule.__getattribute__ executes the import.
    return object.__getattribute__(obj, "__name__")


def _iter_submodules(package_name: str):
    package = importlib.import_module(package_name)
    yield package_name
    for info in pkgutil.walk_packages(package.__path__, prefix=package_name + "."):
        yield info.name


def _optional_import_error(exc: BaseException) -> bool:
    missing = getattr(exc, "name", None) or ""
    root = missing.split(".", 1)[0] if missing else ""
    if root in OPTIONAL_MODULES:
        return True
    msg = str(exc)
    return any(m in msg for m in OPTIONAL_MODULES)


def _check_one_module(mod_name: str) -> list[dict]:
    rows = []
    try:
        module = importlib.import_module(mod_name)
        rows.append({"status": "ok", "msg": mod_name})
    except ImportError as exc:
        status = "skip" if _optional_import_error(exc) else "fail"
        rows.append({"status": status, "msg": f"{mod_name}: ImportError: {exc}"})
        return rows
    except Exception as exc:
        rows.append(
            {
                "status": "fail",
                "msg": f"{mod_name}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            }
        )
        return rows

    if "__getattr__" in vars(module):
        for name in module.__dir__():
            if name.startswith("_"):
                continue
            label = f"{mod_name}.{name}"
            try:
                getattr(module, name)
                rows.append({"status": "ok", "msg": label})
            except Exception as exc:
                rows.append(
                    {
                        "status": "fail",
                        "msg": f"{label}: {type(exc).__name__}: {exc}",
                    }
                )

    for attr, obj in list(vars(module).items()):
        if not _is_lazy_proxy(obj):
            continue
        target = _lazy_name(obj)
        label = f"{mod_name}.{attr} -> {target}"
        try:
            getattr(obj, "__doc__")
            rows.append({"status": "ok", "msg": label})
        except RecursionError:
            rows.append(
                {
                    "status": "fail",
                    "msg": (
                        f"{label}: RecursionError (incompatible with importlib "
                        "LazyLoader; use a function-local import instead of lazy.load)"
                    ),
                }
            )
        except ValueError as exc:
            if "substituted in sys.modules" in str(exc):
                rows.append(
                    {
                        "status": "fail",
                        "msg": (
                            f"{label}: {exc} (package replaces itself during import; "
                            "use a function-local import instead of lazy.load)"
                        ),
                    }
                )
            else:
                rows.append({"status": "fail", "msg": f"{label}: ValueError: {exc}"})
        except ImportError as exc:
            root = target.split(".", 1)[0]
            status = (
                "skip"
                if root in OPTIONAL_MODULES or target in OPTIONAL_MODULES
                else "fail"
            )
            rows.append({"status": status, "msg": f"{label}: ImportError: {exc}"})
        except Exception as exc:
            rows.append(
                {"status": "fail", "msg": f"{label}: {type(exc).__name__}: {exc}"}
            )

    return rows


def _worker(mod_name: str) -> int:
    json.dump(_check_one_module(mod_name), sys.stdout)
    return 0


def _worker_torch_after_package() -> int:
    """Import ML modules that used to stub torch-stack packages, then import torch.

    A leftover lazy.load("torchvision") (etc.) in sys.modules will crash
    during torch's own import via inspect.hasattr.
    """
    for name in (
        "dsa_helpers.ml.segformer_semantic_segmentation.transforms",
        "dsa_helpers.ml.segformer_semantic_segmentation.train",
        "dsa_helpers.ml.segformer_semantic_segmentation.inference",
        "dsa_helpers.ml.yolo.inference",
        "dsa_helpers.ml.yolo.train",
        "dsa_helpers.ml.metrics",
        "dsa_helpers.ml.evaluate",
    ):
        importlib.import_module(name)
    import torch

    json.dump(
        [
            {
                "status": "ok",
                "msg": f"import torch after package modules ({torch.__version__})",
            }
        ],
        sys.stdout,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker",
        metavar="MODULE",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.worker == "__torch_after__":
        return _worker_torch_after_package()
    if args.worker:
        return _worker(args.worker)

    try:
        import dsa_helpers
    except Exception:
        print("FAIL  import dsa_helpers")
        traceback.print_exc()
        return 1

    print(f"testing {dsa_helpers.__file__} ({dsa_helpers.__version__})", flush=True)

    names = list(_iter_submodules("dsa_helpers"))
    ok, skip, fail = [], [], []

    for name in names:
        proc = subprocess.run(
            [sys.executable, __file__, "--worker", name],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode != 0:
            fail.append(
                f"{name}: worker exited {proc.returncode}\n{proc.stderr or proc.stdout}"
            )
            print(f"  fail  {name}", flush=True)
            continue
        try:
            # Libraries like ultralytics print warnings to stdout; JSON is the last array.
            payload = proc.stdout[proc.stdout.rfind("[") :]
            rows = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            fail.append(f"{name}: bad worker output\n{proc.stdout}\n{proc.stderr}")
            print(f"  fail  {name}", flush=True)
            continue
        for row in rows:
            (ok if row["status"] == "ok" else skip if row["status"] == "skip" else fail).append(
                row["msg"]
            )
            if row["status"] != "ok":
                print(f"  {row['status']:4}  {row['msg']}", flush=True)

    torch_proc = subprocess.run(
        [sys.executable, __file__, "--worker", "__torch_after__"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if torch_proc.returncode != 0:
        fail.append(
            f"import torch after package modules: worker exited "
            f"{torch_proc.returncode}\n{torch_proc.stderr or torch_proc.stdout}"
        )
        print("  fail  import torch after package modules", flush=True)
    else:
        try:
            payload = torch_proc.stdout[torch_proc.stdout.rfind("[") :]
            rows = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            fail.append(
                "import torch after package modules: bad worker output\n"
                f"{torch_proc.stdout}\n{torch_proc.stderr}"
            )
            rows = []
        for row in rows:
            (
                ok
                if row["status"] == "ok"
                else skip
                if row["status"] == "skip"
                else fail
            ).append(row["msg"])
            if row["status"] != "ok":
                print(f"  {row['status']:4}  {row['msg']}", flush=True)

    print(f"\nOK     {len(ok)}")
    print(f"SKIP   {len(skip)}")
    print(f"FAIL   {len(fail)}")
    for msg in fail:
        print(f"  fail  {msg}")

    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
