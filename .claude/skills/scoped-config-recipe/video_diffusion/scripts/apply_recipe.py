# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compress an optimum-intel LTX-2.5 export with a multi-component recipe.json and save it for WWB.

    python -P apply_recipe.py <export_dir> recipe.json <out_dir>

Only the components named in the recipe are compressed and written; every other entry of the export is
symlinked, so an output costs only the compressed weights on disk.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import openvino as ov

sys.path.insert(0, str(Path(__file__).parent))

from recipe_io import component_kwargs  # noqa: E402

import nncf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("recipe", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()

    recipe = json.loads(args.recipe.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=False)
    core = ov.Core()
    for entry in sorted(args.export_dir.iterdir()):
        target = args.out_dir / entry.name
        if entry.name not in recipe or recipe[entry.name] is None:
            target.symlink_to(entry.resolve())
            continue
        target.mkdir()
        for f in entry.iterdir():
            if not f.name.startswith("openvino_model."):
                (target / f.name).symlink_to(f.resolve())
        t0 = time.time()
        model = core.read_model(entry / "openvino_model.xml")
        model = nncf.compress_weights(model, **component_kwargs(recipe[entry.name]))
        ov.save_model(model, target / "openvino_model.xml", compress_to_fp16=False)
        size = (target / "openvino_model.bin").stat().st_size / 2**30
        print(f"{entry.name}: {size:.2f} GiB on disk, {time.time() - t0:.0f}s", flush=True)
        del model
    (args.out_dir / "recipe.json").write_text(json.dumps(recipe, indent=1))


if __name__ == "__main__":
    main()
