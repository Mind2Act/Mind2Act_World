# Task catalogue

Every task has Easy, Medium and Hard layouts: 20 public training and 10 validation per level.

| Task | Instruction excerpt | Definition |
| --- | --- | --- |
| CP01 | At the start, watch the reference meal trays pass on the conveyor and remember the three food types on each | [scene](../task/CP01/scene.py) / [score](../task/CP01/scorer.py) |
| CP02 | Remember each macaron and its place on the still cake shown for three seconds | [scene](../task/CP02/scene.py) / [score](../task/CP02/scorer.py) |
| CP03 | Load <N> upright plates into the moving empty slots without disturbing other plates | [scene](../task/CP03/scene.py) / [score](../task/CP03/scorer.py) |
| CP04 | Pour each required jar into the pot in the remembered recipe order, returning jars upright and caps to the tray | [scene](../task/CP04/scene.py) / [score](../task/CP04/scorer.py) |
| CP05 | Pick up the central mallet, press the triangle-marked button and watch the demonstration while holding it, then reproduce the eight key presses in order | [scene](../task/CP05/scene.py) / [score](../task/CP05/scorer.py) |
| CP06 | Move all fruit and vegetables from the left tray onto the left scale pan | [scene](../task/CP06/scene.py) / [score](../task/CP06/scorer.py) |
| CP07 | See task definition | [scene](../task/CP07/scene.py) / [score](../task/CP07/scorer.py) |
| CP08 | Remember the briefly shown front view, then build the structure from memory, centred in the marked area and aligned with the table reference line | [scene](../task/CP08/scene.py) / [score](../task/CP08/scorer.py) |
| CP09 | Remember the briefly shown original drawing on the whiteboard | [scene](../task/CP09/scene.py) / [score](../task/CP09/scorer.py) |
| CP10 | Predict each ball from the launcher's settled aim | [scene](../task/CP10/scene.py) / [score](../task/CP10/scorer.py) |
| DR1 | Pick up all <N> packages on the conveyor belt and place them in the central collection tray. | [scene](../task/DR1/scene.py) / [score](../task/DR1/scorer.py) |
| DR2 | Press the currently illuminated button within <N> seconds and release within <N> second | [scene](../task/DR2/scene.py) / [score](../task/DR2/scorer.py) |
| DR3 | Grasp the eraser, time the moving gate, and pass through its opening to place the eraser stably in the green tray behind it | [scene](../task/DR3/scene.py) / [score](../task/DR3/scorer.py) |
| DR4 | Carry the tray with two balls by its handle to the delivery platform, keeping each ball inside its marked circle | [scene](../task/DR4/scene.py) / [score](../task/DR4/scorer.py) |
| E1 | Observe and remember each batch of items | [scene](../task/E1/scene.py) / [score](../task/E1/scorer.py) |
| E2 | Observe the visible entity-value associations, remember them, decode the physical expression, and enter its exact result using only the closed gripper fingertips on the nine keys. | [scene](../task/E2/scene.py) / [score](../task/E2/scorer.py) |
| F1 | Probe the two unmarked buttons and remember how each moves the slider | [scene](../task/F1/scene.py) / [score](../task/F1/scorer.py) |
| F2 | Place one colored block in each slot, then press and release the confirmation button on the right | [scene](../task/F2/scene.py) / [score](../task/F2/scorer.py) |
| M1 | Remember the target location and station route | [scene](../task/M1/scene.py) / [score](../task/M1/scorer.py) |
| M2 | See task definition | [scene](../task/M2/scene.py) / [score](../task/M2/scorer.py) |
| P1 | Remember the initial orders and place all requested ingredients into the indicated containers. | [scene](../task/P1/scene.py) / [score](../task/P1/scorer.py) |
| P2 | Remove the lid, remember the dim sum order as you unstack the baskets from top to bottom onto the mats, then restack them in the same order from bottom to top on the left steamer pot and replace the lid. | [scene](../task/P2/scene.py) / [score](../task/P2/scorer.py) |
| PR1 | Adjust the resistance between the left terminal and wiper to {TARGET} ohm, then release the slider and keep that reading for 0.5 seconds. | [scene](../task/PR1/scene.py) / [score](../task/PR1/scorer.py) |
| PR2 | Adjust the output voltage to {TARGET} V, then release the knob and keep that reading for 1 second. | [scene](../task/PR2/scene.py) / [score](../task/PR2/scorer.py) |
| PR3 | Grasp the marker and trace the complete visible path, then lift. | [scene](../task/PR3/scene.py) / [score](../task/PR3/scorer.py) |
| PR4 | Fold the square cloth along its centerline into the displayed rectangle, release it fully and leave it flat and stable. | [scene](../task/PR4/scene.py) / [score](../task/PR4/scorer.py) |
| PR5 | Pick up square stamp A and stamp once, fully inside the blue box, with its arrow aligned to the target arrow | [scene](../task/PR5/scene.py) / [score](../task/PR5/scorer.py) |
| PR6 | Stand the dominos anywhere inside the continuous curve band, in order along the curve | [scene](../task/PR6/scene.py) / [score](../task/PR6/scorer.py) |
| S1 | Remember the four colored slots, observe the carousel rotation, then place each colored block into its original matching slot after the carousel stops. | [scene](../task/S1/scene.py) / [score](../task/S1/scorer.py) |
| S2 | Remember the briefly lit forbidden cells | [scene](../task/S2/scene.py) / [score](../task/S2/scorer.py) |

## Scoring notes

These notes describe the implemented scorer. They do not replace the fixed model instructions or add new scoring gates. Scores use `mind2act-final-scoring-20261105`.

| Task | Implemented rule or boundary |
| --- | --- |
| M1 | The first N supported placements form the submitted route; placement N ends the round. An exact route succeeds. Let C be its longest common subsequence with the expected route, W = submitted placements − C, and m = N − C + W. For m = 1 / 2, Easy earns 15 / 7.5, Medium 11 / 5.5, and Hard 7.5 / 3.75; other incomplete routes earn 0. Uncover/extraction diagnostics do not gate these scores. |
| F1 | At the goal, an unreleased last key earns cap/4, including when extra presses were made. “Goal reached with extra presses” earns cap/2 only after that key has returned; full success also requires the shortest route and protocol prerequisites. |
| P1 | Replacing a requested item with a wrong type counts both one missing and one wrong item: 7.5 / 5.5 / 3.75 for Easy / Medium / Hard. Missing items are counted against the fixed total N even before all additional orders have appeared. |
| P2 | The intermediate tier requires ceil(N/2) ≤ p < N, where p is the consecutive correct prefix from the bottom. Distractor baskets must be absent from the final stack; they may be moved aside earlier. Visiting each mat is not a scoring gate. |
| CP04 | Recipe order is credited by longest common subsequence. Jar uprightness is not scored; the other ingredient and thermal gates remain in force. |
| CP05 | Mallet contact, last-key release and final withdrawal are not scoring gates. Hard checks MAX before Play and at the terminal sample, not continuously between them. |
| PR1 / PR2 | Scoring reads the final T0 resistance/voltage. Release and holding are not scoring gates; the released-hold window controls scene termination only. PR1 tolerances are ±0.2 / ±0.15 / ±0.12 Ω. PR2 tolerances are ±1 / ±0.5 / ±0.5 V; Hard tightens the partial-credit zero distance, not the voltage tolerance. |
| PR5 | Reject missing, repeated, dragged or out-of-box imprints first; then grade heading, then stamp return. |
| DR3 | Ordered physical aperture passage and terminal delivery are scored. Gripper withdrawal is not a scoring gate. |
| DR4 | Ball touching is not a scoring gate. Easy / Medium / Hard have 2 / 3 / 4 balls and 0 / 1 / 2 columns respectively. Retention, terminal delivery and obstacle-contact rules still apply. |
| E1 | The tray accepts one submission per round; a wrong submission still settles that round and cannot be replaced with another answer. |
| E2 | Remember the objects and their values, calculate the expression, and enter its result. Fingertip contact is not a scoring gate; the fixed short instruction does not specify a key-return requirement. |
| S1 | Early placement is recorded but does not itself force a zero. Easy uses two rotations in the same direction. Original training demonstrations use a 2-second cue; formal evaluation uses a 1-second cue. |

PR3 short-stroke partial credit and the remaining source rules are unchanged. A stricter phrase in an instruction does not silently introduce an additional scoring condition.

CP01 assigns reference orders to trays by the correspondence that maximizes credited foods minus wrong or extra foods. Ties use the first order-index permutation. Tray support, separation and placement rules are unchanged.
