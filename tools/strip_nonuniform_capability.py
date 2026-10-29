#!/usr/bin/env python3
"""Remove the conservative non-uniform-indexing capabilities glslc emits for descriptor heaps.

Muyo's descriptor-heap pipeline is the graph's only binding path, and a GLSL shader indexes a
`layout(descriptor_heap)` array with an index from push data. glslang marks that index dynamically
indexed and emits `ShaderNonUniform` plus `StorageBufferArrayNonUniformIndexing`, whose Vulkan 1.2
feature `shaderStorageBufferArrayNonUniformIndexing` the device does not enable (and a consumer
cannot enable: `VkRenderDevice::CreateDevice` owns the feature chain). Muyo's own heap shaders are
Slang and reach resources through a function-local `.Handle()`, which needs neither capability.

The index here is uniform across the dispatch - it is one word of push data, the same for every
invocation - so the non-uniform semantics the capability promises are not used. Removing the two
capability lines leaves the `ResourceHeapEXT` access intact; `spirv-val --target-env vulkan1.4`
accepts the result and the shader validates against the features the device actually enables. This
is a workaround for the upstream gap recorded in docs/Findings.md, not a property of the shader.

Usage: strip_nonuniform_capability.py --dis <spirv-dis> --as <spirv-as> <file.spv>
The file is rewritten in place only when a capability was present.
"""

import argparse
import pathlib
import subprocess
import sys
import tempfile

REMOVED_CAPABILITIES = (
    "OpCapability StorageBufferArrayNonUniformIndexing",
    "OpCapability ShaderNonUniform",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dis", required=True, help="path to spirv-dis")
    parser.add_argument("--as", dest="assembler", required=True, help="path to spirv-as")
    parser.add_argument("spirv", type=pathlib.Path, help="SPIR-V file to rewrite in place")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tempDir:
        text = pathlib.Path(tempDir) / "module.spvasm"
        subprocess.run([args.dis, str(args.spirv), "-o", str(text)], check=True)
        lines = text.read_text(encoding="utf-8").splitlines()
        if any("NonUniform" in line and line.lstrip().startswith(("OpDecorate", "OpMemberDecorate"))
               for line in lines):
            print(f"{args.spirv}: declares non-uniform indexing; refusing to strip the capability",
                  file=sys.stderr)
            return 1
        kept = [line for line in lines if not any(capability in line for capability in REMOVED_CAPABILITIES)]
        if len(kept) == len(lines):
            return 0
        text.write_text("\n".join(kept) + "\n", encoding="utf-8")
        subprocess.run([args.assembler, "--target-env", "vulkan1.4", str(text), "-o", str(args.spirv)], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
