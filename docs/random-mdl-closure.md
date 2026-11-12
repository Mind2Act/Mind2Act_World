# Random material MDL dependencies

The ten Wood_Grain_Raw/Varnish materials import
`..::Templates::OmniUber_Automotive::*`. The selected45 asset subset omitted
that package. A present top-level MDL file therefore passed the old existence
gate while NVIDIA's compiler rejected its imports and the renderer showed a
red fallback surface.

`manifests/materials/random-automotive-v1` retains the original NVIDIA
2023_2_1 Automotive/Imperfect `OmniUber_Automotive.mdl`, `OmniGlass.mdl` and
package license. Source URL, archive ETag, member names and SHA256 are recorded
in `source.json`. The files were retrieved from the
[official Automotive pack](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html)
without editing material source or parameters.

The full delivered2250 old15 plus1800 new15 finalized layouts use39 table/ground
materials. The selected45 subset held34. `corpus-materials.json` pins the five
remaining original materials and their texture files, retrieved from the
existing authorized installation. No layout, slot, seed, geometry, rigid state,
physics material, camera, or scoring condition changes.

## Build a new cold asset patch

```sh
python tools/materials/restore_random_automotive.py \
  --source-release /path/to/current/old15-release-v1 \
  --corpus-material-pool /path/to/original-material-pool \
  --output /path/to/new-patch
```

The pool contains `Assets/Material/...` with the exact hashes recorded in
`corpus-materials.json`. The generator writes only a new `files/` patch and a
generation receipt. It adds the templates, source license, five missing corpus
materials, module pins and the material package search root. It rebinds each
random spec's variation-manifest SHA, then regenerates asset-manifest and
RELEASE hashes. Existing material pins and all layout bytes remain unchanged.

Apply this patch only to a new cold generation. Payloads may be hard-linked;
replace metadata using a separate temporary inode and `os.replace`, rather
than overwriting a live/source inode. The coordinator records the final shared
code commit. The generator deliberately refuses an unbound clean-only release;
its material pool can be shared with independently prepared new15 random
manifests after exact material SHA/name checks.

## Validate every material use

```sh
PYTHONPATH=src python tools/materials/audit_random_mdl.py \
  --inventory /path/to/frozen-4050-layout-inputs.json \
  --compiler /path/to/nvidia/mdlc \
  --report /path/to/new-report.json
```

The inventory fixes denominators, source layout paths/hashes and release
manifest pins. Draft layouts are excluded by the input producer. Every layout
links to the exact table/ground checks it consumes; identical source material
files are compiled once per source group. Missing catalog entries fail rather
than reducing the denominator. Texture bytes are checked against their pins.

NVIDIA MDLC1.10 build381500.6583 reproduced C120 before the dependency repair.
The repaired corpus passed78 checks (39 materials ×2 source groups),4050
layouts and zero failures. MDLC uses `--no-strict --resolve-resources`, allowing
the implicit compatible-type casts supported by the
[MDL SDK configuration](https://github.com/NVIDIA/MDL-SDK/blob/2024.1.4/include/mi/neuraylib/imdl_configuration.h).
The CLI's stricter subset rejects scalar-to-vector arguments present in
NVIDIA's original exports. No source rewrite, ignored import, or disabled
resource check is used to obtain a pass. Compiler arguments, return codes and
diagnostics are retained in the report, including the earlier strict-subset
failure for review.

## Runtime gate

The composed-stage gate now follows imports and resources of used MDL
shaders, including USDZ members. It then loads each used module through the
active Omniverse compiler, using its tracked `createMdlModule` interface.
Missing dependencies or compiler failure invalidate the episode before native
physics. Unused/unbound MDL remains a warning. CPU fixtures explicitly report
compilation as `not_run`; static existence never stands in for compilation.
Per-world successful module checks survive visibility-only updates such as
CP04's digits. Asset/property resync changes and reset clear that cache, so
changing a shader asset revalidates it before the next physics step.

This corpus check covers the appearance material pool. It does not certify
all distractor geometry, all native episodes, image appearance, or evaluation
scores. Native PRO validation of the eleven reported old15 random cells and
CP05hard text remains a separate, coordinator-scheduled step on the next unified
generation. GPU server/4090 synchronization requires code/assets/file SHA only and
does not start native smoke or model runtimes there.

## SDK-first symbolic module resolution

Kit can resolve a symbolic `gltf/pbr.mdl` in an imported distractor USDZ to its installation even when the release already includes the exact module. The random gate formerly accepted only the release path or the same physical inode; copies on another host consequently failed.

The runtime now maps a dependency only when the source layer authors a symbolic MDL token, a declared search root contains its already SHA-pinned release target, and the SDK-resolved file has identical bytes. Absolute authored paths do not gain this mapping. Missing, changed, or unpinned files still fail. Transitive MDL imports and literal texture/resource references are audited against the declared closure as well.

After importing the checked distractor, the random authoring layer overrides symbolic MDL asset attributes with relative references to those exact release files. The MDL compiler's declared release search roots precede its installation roots for random episodes; readback verifies this order. Clean episodes do not request the ordering change or execute the random authoring code.

No release data needs rewriting for READY generation `1887ec2422eb8e922fa83992afdc0aa312c6f4e4`: its task manifests already contain the 231 pinned MDL/library resource entries. The data publication owner must retain those files and search directories in the new generation, then reseal only through the normal owner workflow. No broad SDK-directory exemption, name exemption, or same-hash exemption for arbitrary absolute authored references is introduced.

CPU evidence is generated with `tools/materials/audit_random_mdl_closure.py`; it inventories all random manifests, audits each distinct distractor payload, replays the SDK-first dependency result, and checks that composed shader asset opinions resolve inside the release. The replay is explicit instrumentation of the dependency extractor, not a native Kit/compiler acceptance claim. It starts neither Kit nor a GPU. Clean checks cover unchanged fixed50 specs and the clean variation no-op; they do not claim a native task build.

Evidence on PRO: `/path/to/resource`. `authored-references.json` and `authored-payload-aliases.json` additionally cover every USD/MDL payload in both release inventories, including clean assets and archive members. The inventory retains an upstream unused `iray/environment.mdl` default `std_env.hdr` reference absent from the installed SDK too; no authored USD or MDL imports this environment module. It is recorded as an unused library limitation rather than replaced with invented image bytes.

The source SDK distribution metadata identifies Isaac Sim **5.1.0.0**; its `VERSION` additionally identifies the exact build. The original `gltf/pbr.mdl` and `OmniPBR.mdl` headers permit source/binary redistribution with retained copyright, conditions and disclaimer and prohibit endorsement using NVIDIA's name. No SDK bytes are modified or newly redistributed by this code change; keep the existing per-file license notices and Automotive package license with release files.
