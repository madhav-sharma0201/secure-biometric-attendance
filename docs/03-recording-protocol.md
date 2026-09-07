# Self-Collected Test Set — Recording Protocol

This set is the project's only measurement of generalisation to unseen people, unseen
cameras and unseen conditions. It is used **exclusively as a held-out test set**: never
for training, never for threshold selection.

Target: ~15 live clips, ~15 spoof clips. Around 20 minutes.

## Rules that matter

**Every clip: 3-5 seconds, 1080p or 720p, 30fps, face fills roughly a third of frame.**
Do not shoot 4K — the files are huge and get downscaled to 112x112 anyway.

**Hold the camera reasonably still.** The temporal model reads motion. Camera shake is
motion that has nothing to do with liveness, and too much of it adds noise to the one
signal we are testing.

**Vary the bezel.** Record about half the spoof clips with the screen filling the frame
so no phone or laptop edge is visible, and half naturally with edges showing. If every
spoof shows a bezel, the model can score well by detecting rectangles rather than
detecting attacks — it becomes a bezel detector. The bezel-free clips are the honest
part of the test.

**Spoofs must replay YOUR live clips.** Same face, same clothes, same scene. If the
live and spoof clips show different content, any measured difference could come from
the content rather than from liveness.

## 1. live/ — about 15 clips

Record yourself, front camera, one clip per condition:

| # | Condition |
|---|---|
| 1-2 | Bright room light, facing camera |
| 3-4 | Dim room, lights mostly off |
| 5-6 | Backlit — window or lamp behind you |
| 7-8 | Head turned left, then right (~30 degrees) |
| 9-10 | Looking slightly up, then down |
| 11-12 | Close (arm's length), then far (2-3 m) |
| 13-14 | Outdoors or a different room |
| 15 | Wearing glasses, if you have them |

Blink and breathe normally. Do not hold unnaturally still — micro-motion is exactly
the signal the model should be using.

## 2. phone/ — about 6 clips

Play your `live/` clips full-screen on one phone, film that screen with another phone
(or your laptop webcam).
- Maximum screen brightness
- Fill the frame with the screen for half the clips
- Try one at a slight angle to catch screen glare

## 3. laptop/ — about 5 clips

Same, playing on your laptop screen. Laptop panels are larger and have different
refresh and reflection characteristics than phones, which is why this is a separate
category rather than lumped in with phone replay.

## 4. print/ — about 4 clips

Print 3-4 photos of your face on plain A4 (colour if possible, greyscale is fine —
note which in the README).
- Hold up and film, filling the frame with the paper
- Do one flat, one slightly curved toward the camera
- Do one under bright light to catch paper sheen

## Getting the files onto your Mac

AirDrop is simplest. Then sort into folders:

```
data/raw/self/
    live/     your genuine clips
    phone/    filmed off a phone screen
    laptop/   filmed off a laptop screen
    print/    filmed off printed photos
```

The folder name becomes the attack label in the per-attack results table, so put each
clip in the right place. Filenames do not matter.

## iPhone HEVC note

iPhones record HEVC `.mov` by default, which OpenCV often cannot decode. If the check
script reports files it cannot open, convert them:

```
brew install ffmpeg
cd data/raw/self
for f in */*.mov; do ffmpeg -i "$f" -c:v libx264 -crf 20 -an "${f%.mov}.mp4" && rm "$f"; done
```

To avoid it entirely: iPhone Settings > Camera > Formats > **Most Compatible**.

## Verify before relying on it

```
python scripts/check_selfcollected.py data/raw/self
```

It fails loudly on unreadable files, clips that are too short, missing categories, or
an empty directory. Fix anything it reports before uploading.

## Consent and privacy

All recordings are of the project author, who is the sole subject and consents to their
use for this project. No third party is recorded. Raw video is gitignored and never
committed. In the deployed system only embeddings are stored, never raw images
(see Phase 25).

If you later add other people: get written consent, record what they consented to, and
delete on request.
