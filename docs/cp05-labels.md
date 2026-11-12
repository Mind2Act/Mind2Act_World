# CP05 text material repair

The PRO easy/hard recordings rendered the nameplate and VOLUME/MAX/MIN as cyan
rectangles. Their PNG bytes existed, but the authored paths ended in a bare SHA.
RTX logged `createimage: Invalid image format extension`. Medium used identical
bytes with `.png` names and rendered the text.

`tools/materials/rebuild_cp05_labels.py` regenerates the two affected material
layers through typed Sdf attributes. Each layer changes only four relative
texture paths to the accepted medium PNGs. The generator checks exact source
layer identity, equality of all non-asset opinions, and byte equality of each
old/new image. Source assets, physics, H5, source specs, layouts and scores stay
unchanged. The versioned overlay is `assets/overlays/cp05-labels-v2`; its manifest
and generation receipt are recorded under `manifests/materials/cp05-labels-v2/`.
It must be assembled into a new immutable release, never copied over a live root.

The CPU material-closure audit now follows the actually visible bound label
shaders. It rejects missing/extensionless/undecodable images and single-color
label placeholders. Constant normal maps and color-only geometry lettering are
allowed; unused and invisible material images do not fail this additional check.
This is a CPU release audit, not a change to the runtime existence-only gate.
It requires the existing test extra (`usd-core` and Pillow).

Real-stage CPU comparison: old easy/hard each fail on four label images; old
medium passes. The private candidate passes separately on easy, medium and hard.
The three native PRO first frames were then captured separately on idle B7 and
compared with their own training H5 head0. All four text regions were legible in
each tier, with zero cyan-block pixels in the checked regions. The receipt is
`manifests/materials/cp05-labels-v2/native-first-frames.json`. These are deliberate
zero-action, zero-score diagnostics on frozen code 8f6e2abb; old-runtime cleanup
timeouts remain recorded. They qualify the first-frame material repair only,
not complete task replay or the next generation's 90-cell visual review.

For the 90-cell PRO visual review, every task/tier needs its own first and middle
native frame, the bound training H5 frames at the same control rows, the actual
code/asset/input identities, and separate checks of text/labels, screens, table
and objects. CPU success or one tier's image never promotes another cell. The
planning index starts all 90 cells as `NOT_RENDERED` at
`runs/material-portability/render-review-90/index.json`; native captures and review
receipts must fill each cell before a complete visual qualification is claimed.

Evidence remains on PRO under `runs/material-portability/cp05-labels/`, with small
comparison copies in `mind2act-eval-plan/reviews/cp04-render/cp05-labels/`.
