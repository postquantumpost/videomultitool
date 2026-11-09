# videomultitool
A program to perform various video evaluation, editing, and filtering.

## Setup

Use the project-local Python virtual environment (Debian/Ubuntu/Mint):

```
sudo apt install python3-venv ffmpeg
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Activate this environment in each new shell before running the program or
tests. Install dependencies through `python3 -m pip` in the same environment.
The requirements keep NumPy below 2 and headless OpenCV below 4.12: FastANPR's
PaddleOCR dependency installs OpenCV 4.6 wheels compiled against NumPy 1.x,
while newer headless OpenCV wheels require NumPy 2.

If an existing environment reports `numpy.core.multiarray failed to import`
or a NumPy ABI version mismatch, repair these dependencies together:

```
source .venv/bin/activate
python3 -m pip install --force-reinstall 'numpy>=1.26,<2' 'opencv-python-headless<4.12'
python3 -m pip install 'numpy>=1.26,<2' scipy pandas tifffile contourpy
python3 -m pip check
python3 -c 'import cv2, numpy; print(cv2.__version__, numpy.__version__)'
```

The second command resolves scientific dependencies that may also have been
upgraded to releases requiring NumPy 2. Keep the NumPy constraint when adding
packages later; otherwise a separate install can upgrade it again.

Model weights are downloaded on first use (into `~/.deepface` and `~/.hyperlpr3`).

## Usage

```
./videomultitool.py [--debug] [--debug2] [--nobluredge] [--fill]
                    [--blur | --blankless] [--alacarte LIST]
                    [--input FILE] [--output FILE]
```

| Flag | Effect |
|---|---|
| `--debug` | debug output; with `--blur`, draws the boxes and timestamp/frame number |
| `--debug2` | enables `procfun_debug2` (random test boxes) |
| `--nobluredge` | with `--blur`, do not extend the blur beyond each box |
| `--fill` | with `--blur`, fill each box with its average colour instead of blurring |
| `--blur` | blur mode (below) |
| `--blankless` | border-cropping mode (below) |
| `--alacarte LIST` | blur using only the listed processing functions; an empty string copies the file unchanged |

A file is only processed when `--input`, `--output` and one of the modes
`--blur`, `--blankless` or `--alacarte` are given. `--blankless` cannot be
combined with either of the other modes. In processing modes the audio is
copied over unchanged and progress is printed every 10%.

### `--alacarte LIST`

Select processing functions with a comma-separated list: `retinaface`,
`deepface`, `nudenet`, `hyperlpr`, `fastanpr`, and `debug2`.
All six are disabled first,
then only the listed functions are enabled, regardless of `--debug2`.
Functions still run in the fixed order shown below, not
the list order. Surrounding whitespace and duplicate names are accepted;
unknown names and empty entries within a nonempty list are errors.

```
./videomultitool.py --alacarte "deepface,retinaface" --input in.mp4 --output out.mp4
./videomultitool.py --alacarte "fastanpr,hyperlpr" --input in.mp4 --output out.mp4
./videomultitool.py --alacarte "" --input in.mp4 --output out.mp4
```

`--blur` is optional with `--alacarte`. An empty list copies the entire file
byte-for-byte, with no decoding, re-encoding, detector loading, or debug
overlays, even if other blur/debug options are supplied. Without `--alacarte`,
all detectors, including RetinaFace, run with `--blur`; `debug2` remains opt-in.

### `--blur`

Every video frame is decoded, processed, and re-encoded (H.264).
Each frame is passed through these processing functions, in order, and every
box they return is blurred:

| Function | Detects | Library |
|---|---|---|
| `procfun_retinaface` | faces | RetinaFace (frame scaled to 960 px max; ~5 s/frame on CPU) |
| `procfun_deepface` | faces | deepface, YuNet backend |
| `procfun_nudenet` | faces and exposed breasts/genitalia/buttocks/anus | NudeNet |
| `procfun_hyperlpr` | license plates | HyperLPR3 (trained on Chinese plates) |
| `procfun_fastanpr` | license plates | FastANPR's bundled detector, CPU, without OCR |
| `procfun_debug2` | 10 random boxes, only with `--debug2` | — |

Each list of detected boxes also applies to frames up to `half_window_seconds`
before and after its source frame (inclusive), using video timestamps rather
than a fixed frame count. This global setting defaults to `0.15` seconds at
the top of `filter_in_to_out.py`; set it to `0` for frame-local processing.
Output is delayed until the future half-window has been analyzed. Buffered
frames at the end of the video are still written, and output timestamps and
audio timing are unchanged. All applicable lists are combined before rendering,
so overlapping boxes across frames are blended without repeatedly blurring or
filling the same pixel.

Blur strength scales with box size: boxes up to 100 px (largest side) use the
base blur, larger boxes proportionally more (200 px gets twice the blur). By
default the blur also extends 10% of the box width beyond the left and right
edges and 10% of the box height beyond the top and bottom, fading from full
strength at the box edge to none at the outer edge (`--nobluredge` turns this
off). With `--fill`, each box is filled with the average colour of its
contents, and that colour fades out over the same edge margin, blended with
the blur. Overlapping boxes are blended so every pixel is changed only once.
These constants are at the top of `filter_in_to_out.py`.

Thresholds and options are constants at the top of each `procfun_*.py` file.
A detector that fails to load prints a warning and is skipped.

FastANPR uses detection only, retaining boxes even without recognized text.
Its model weights ship with the package.
FastANPR and its Ultralytics dependency use AGPL licensing; review their
license terms before redistribution.

NudeNet includes both `FACE_FEMALE` and `FACE_MALE`. Its local postprocessing
uses `NUDENET_NMS_OVERLAP = 0.9` instead of the library's 0.45, retaining more
overlapping boxes for blurring while still suppressing near-duplicates.
The existing NMS score cutoff of 0.25 is unchanged (`NUDENET_NMS_SCORE`).

### `--blankless`

Detects letterbox/pillarbox (or windowbox) borders and writes only the real
video region. About 40 frames spread over the file are sampled; a side is
cropped only when the bar is almost entirely static (black, or filler such as a
gradient or UI, with changing text like subtitles allowed) and a strong
persistent edge marks the boundary. Dark or static parts of a full-frame video
are therefore kept. A 2 px margin is trimmed inside each cropped edge to drop
blended pixels. Thresholds are constants at the top of `blankless_in_to_out.py`;
`--debug` prints the decision for each side.

### Output quality

Video is encoded with x264 in constant-quality mode
(CRF 18, visually lossless; adjust `X264_CRF` in `video_util.py`), using
the input's chroma format (4:2:0/4:2:2/4:4:4) when possible. Pixels not altered
by the blur renderers, and the cropped region in `--blankless`, are taken directly from the decoded input, avoiding YUV/RGB
conversion loss.
