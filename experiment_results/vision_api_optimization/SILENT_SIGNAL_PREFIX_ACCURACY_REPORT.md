# Silent Signal prefix accuracy experiment (read-only)

## Scope

This experiment did not modify `api/vision_api.py`, `detector_front.py`, `tracking/`, models, or videos. It was started against the working-tree baseline whose pre-existing modifications were `api/vision_api.py` and `hand_motion_desktop/src/detector_front.py`. Both `VISION_SILENT_SIGNAL_SAME_HAND_ONLY` and `VISION_SILENT_SIGNAL_OPERATOR_ONLY` were forced to `false`.

The GPU had third-party processes that could not be identified due to Windows permissions, so no latency or GPU-performance claim is made here. This is an accuracy and state-machine-equivalence experiment only.

## Method

`silent_signal_prefix_accuracy.py` imports the current detector's classifier, constants, HandLandmarker construction, and `RepeatedFistToOpenTracker`. For each source frame it runs the unchanged rule and stops exactly when `tracker.update()` first returns `True`. It then independently runs the existing full-video detector and compares its `detected_frame` to the prefix `confirmedFrame`.

The API's default coordinate setting is `SILENT_SIGNAL_COORDINATE_MODE=legacy`; the separate current front-camera setting is `front`. Both must be reported because the requested frame-114 evidence is specific to front mode.

## Results

| Clip | Human label / note | API default legacy | Front current rule | Prefix equals full first event |
| --- | --- | --- | --- | --- |
| 175204 | user-provided positive | no event | event frame 114 | yes, both modes |
| 175021 | user-provided positive/ambiguous | event frame 345 | event frame 333 | yes, both modes |
| 174439 | human-reviewed negative | false positive frame 31 | false positive frame 35 | yes, both modes |
| 174454 | human-reviewed negative | false positive frame 100 | false positive frame 174 | yes, both modes |
| 174341 | no-signal comparator | no event | no event | yes, both modes |

Full raw rows, FPS, frame counts, hand-present counts, and SHA-256 source identities are in [summary.csv](silent_signal_prefix_accuracy/summary.csv). Each of the 10 mode/clip checks has `prefixEqualsFullFirstEvent=true`.

## Accuracy interpretation

No threshold, classifier, or ownership policy was changed, so this is not an accuracy improvement. It proves only that an early path can preserve the existing first-event decision if it uses the exact same mode and state machine.

For the small labelled set, front mode sees both supplied positives but also both human-reviewed negatives; legacy sees 175021 but misses 175204 and also flags both negatives. Neither mode is safe to describe as an accuracy improvement. Early return would make these known false positives arrive earlier:

- `174439`: frame 31 legacy / frame 35 front.
- `174454`: frame 100 legacy / frame 174 front.

## Integration implication for the parallel implementation

The request's `175204` frame 114 can only be asserted when the implementation runs the front-coordinate detector (`SILENT_SIGNAL_COORDINATE_MODE=front`) with both experimental filter flags off. The present API default (`legacy`) does not detect that clip in this controlled replay. An early-return feature must therefore preserve and log the coordinate mode; it must not silently substitute front mode, since that changes baseline results.

The state machine confirms at the same frame at which the legacy full detector records `detectedFrame`: a 5-frame fist sequence followed by an open palm within its mode-specific transition gap (15 frames legacy, 30 front; one repeat required). There is no need to inspect future frames to establish that existing rule's first event.
