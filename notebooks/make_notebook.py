"""Builds notebooks/gpu_stages.ipynb from gpu_stages.py (the source of truth) plus a Colab bootstrap cell.

    python notebooks/make_notebook.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

BOOTSTRAP = '''# Run first. Clones the repo (footage and tracking mesh are in it) and puts it on the latest main.
import os, subprocess, sys
from pathlib import Path

REPO = "https://github.com/adeshboudhnicedigitals/fpv-mocap.git"  # private repo: needs a token
BASE = Path("/content") if Path("/content").exists() else Path("/kaggle/working")  # Colab or Kaggle
ROOT = BASE / "fpv-mocap"
if not ROOT.exists():
    token = os.environ.get("GITHUB_TOKEN", "")
    try:
        from google.colab import userdata  # Colab: key icon > Secrets > GITHUB_TOKEN
        token = userdata.get("GITHUB_TOKEN")
    except ImportError:
        try:
            from kaggle_secrets import UserSecretsClient  # Kaggle: Add-ons > Secrets > GITHUB_TOKEN
            token = UserSecretsClient().get_secret("GITHUB_TOKEN")
        except ImportError:
            pass
    url = REPO.replace("https://", f"https://{token}@") if token else REPO
    subprocess.run(["git", "clone", "-b", "main", url, str(ROOT)], check=True, cwd=BASE)
subprocess.run(["git", "fetch", "origin"], check=True, cwd=ROOT)
subprocess.run(["git", "checkout", "main"], check=True, cwd=ROOT)  # an older clone may sit on another branch
subprocess.run(["git", "pull", "--ff-only", "origin", "main"], check=True, cwd=ROOT)
'''

LOAD = '''# Extract the frames from the committed clip (skipped when rgb/ already has them).
import sys
from pathlib import Path

BASE = Path("/content") if Path("/content").exists() else Path("/kaggle/working")
ROOT = BASE / "fpv-mocap"
CLIP = ROOT / "clips" / "m4_pullout"
sys.path.insert(0, str(ROOT))
from mocap.frames import extract
if not any((CLIP / "rgb").glob("*.png")):
    print(extract(ROOT / "m4_pullout_single_shot.mp4", CLIP / "rgb", 60.0), "frames")
'''


def cells(py: str):
    parts = re.split(r"^# %%(.*)$", py, flags=re.M)
    head = parts[0].strip()
    out = [("markdown", head)] if head else []
    for title, body in zip(parts[1::2], parts[2::2]):
        out.append(("code", f"# {title.strip()}\n{body.strip()}" if title.strip() else body.strip()))
    return out


def cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def main():
    py = (HERE / "gpu_stages.py").read_text(encoding="utf-8")
    head = "\n".join(line.lstrip("# ").rstrip() for line in py.split("# %%")[0].strip().splitlines())
    items = [("markdown", "# GPU stages for m4_pullout\n\nUse a GPU runtime (Colab T4, or Kaggle with internet on), then run the cells in order.\n\n" + head),
             ("code", "# bootstrap\n" + BOOTSTRAP)] + cells(py)[1 if py.lstrip().startswith("#") and not py.lstrip().startswith("# %%") else 0:]
    nb = {
        "cells": [cell(k, t) for k, t in items],
        "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (HERE / "gpu_stages.ipynb").write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(len(nb["cells"]), "cells")


if __name__ == "__main__":
    main()
