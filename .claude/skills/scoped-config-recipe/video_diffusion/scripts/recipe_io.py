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
recipe.json for a multi-component OpenVINO pipeline: {"<component>": <nncf.compress_weights kwargs>, ...}.
A component missing from the recipe (or null) is not compressed; keys starting with "_" are notes.

Per component: mode, group_size, ratio, scoped_configs, ignored_scope, all_layers, backup_mode,
advanced_parameters (nncf.Scope fields as JSON: names, patterns, types, subgraphs).

    from recipe_io import compress_pipeline
    compress_pipeline(pipe, "recipe.json")   # pipe = OVLTX2Pipeline.from_pretrained(..., compile=False)
"""

import json


def component_kwargs(r):
    """One component's recipe dict -> nncf.compress_weights() kwargs with NNCF objects."""
    import nncf

    def scope_fields(d):
        d = dict(d)
        if "subgraphs" in d:
            d["subgraphs"] = [nncf.Subgraph(**sg) for sg in d["subgraphs"]]
        return d

    kwargs = {k: v for k, v in r.items() if not k.startswith("_")}  # "_comment" etc. are notes
    kwargs["mode"] = nncf.CompressWeightsMode(r["mode"])
    if "backup_mode" in r:
        kwargs["backup_mode"] = nncf.BackupMode(r["backup_mode"])
    kwargs["scoped_configs"] = [
        nncf.ScopedConfig(**{**scope_fields(sc), "mode": nncf.CompressWeightsMode(sc["mode"])})
        for sc in r.get("scoped_configs", [])
    ]
    kwargs["ignored_scope"] = nncf.IgnoredScope(**scope_fields(r.get("ignored_scope", {})))
    if "advanced_parameters" in r:
        adv = dict(r["advanced_parameters"])
        if "group_size_fallback_mode" in adv:
            adv["group_size_fallback_mode"] = nncf.GroupSizeFallbackMode(adv["group_size_fallback_mode"])
        kwargs["advanced_parameters"] = nncf.AdvancedCompressionParameters(**adv)
    return kwargs


def compress_pipeline(pipe, recipe_path):
    import nncf

    with open(recipe_path) as f:
        recipe = json.load(f)
    for comp, r in recipe.items():
        if r is None or comp.startswith("_"):
            continue
        part = pipe.components[comp]
        part.model = nncf.compress_weights(part.model, **component_kwargs(r))
        part.request = None
    return pipe
