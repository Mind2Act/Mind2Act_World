# CP04 recipe-to-READY presentation

At logical tick 420, bind the authored READY material to the original
`RecipeScreen`, matching the collection implementation's
`RecipeDisplay.hide()` and selected `mind2act_cooking_assets.hide_recipe()`.
Keep `RecipeScreenHidden` invisible; it retains the READY material in the
portable asset closure rather than becoming a replacement draw surface.

Previously the evaluator hid the original mesh and showed the coplanar
template. This was a presentation difference from the collection mechanism.
The patch changes only material binding and visibility. Logical timing,
thermal state, physics, actions and scoring are unchanged.

The source comparison checked READY's 1200 x 720 RGB PNG, sRGB texture,
`UsdPreviewSurface` diffuse zero / texture-driven emission / roughness .72,
and absent explicit mipmap/filter overrides. These match the source.
READY PNG SHA256:
`c21e4e786b77ad109e21fc344df9854f918b1c57df5800f475b09edfcea873a4`.
PR95 already aligns FXAA (op 2, limited operations false), two zero-delta
warmup steps of eight subframes and two capture subframes with the pinned
NativeRGBRecorder implementation. None of those settings changes here.
The projected training H5 does not independently retain historical renderer
settings; JPEG training images and H264 replay also differ in encoding.

Validation covers both source/evaluation clocks and all three tiers using
real composed USD. The original implementation fails six original-mesh
assertions; the replacement passes. Linux USD 24.5 also checks each real
station's relative READY texture, PNG decoder, unchanged source layer,
unchanged points and session-layer properties limited to binding/visibility.

This establishes source-mechanism parity. The inspected historical PRO
video did not consistently reproduce an additional moire pattern; CPU tests
do not establish that the reported visual symptom is eliminated. A new
native frame matched to the training control tick is recorded separately.
