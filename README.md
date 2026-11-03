# videomultitool
A program to perform various video evaluation, editing, and filtering.

## Setup

Uses the system Python 3 with distribution packages (Debian/Ubuntu/Mint):

```
sudo apt install python3-av python3-opencv python3-numpy ffmpeg
```

The detectors (RetinaFace, deepface, NudeNet, HyperLPR) are only on PyPI.
Install them for the current user, keeping numpy at the apt version so the
apt OpenCV/PyAV modules keep working:

```
echo "numpy==$(python3 -c 'import numpy; print(numpy.__version__)')" > /tmp/np.txt
pip install --user --break-system-packages -c /tmp/np.txt \
    retina-face deepface tf-keras werkzeug nudenet hyperlpr3
```

Model weights are downloaded on first use (into `~/.deepface` and `~/.hyperlpr3`).

## Usage

```
./videomultitool.py [--debug] [--debug2] [--slowok] [--blur | --blankless]
                    [--input FILE] [--output FILE]
```

| Flag | Effect |
|---|---|
| `--debug` | debug output; with `--blur`, draws the boxes and timestamp/frame number |
| `--debug2` | enables `procfun_debug2` (random test boxes) |
| `--slowok` | allows slow processing functions (currently `procfun_retinaface`) |
| `--blur` | blur mode (below) |
| `--blankless` | border-cropping mode (below) |

A file is only processed when `--input`, `--output` and one of the modes
`--blur` or `--blankless` are given; the two modes cannot be combined. In both
modes the audio is copied over unchanged and progress is printed every 10%.

### `--blur`

Every video frame is decoded, processed, and re-encoded (H.264).
Each frame is passed through these processing functions, in order, and every
box they return is blurred:

| Function | Detects | Library |
|---|---|---|
| `procfun_retinaface` | faces | RetinaFace, only with `--slowok` (frame scaled to 960 px max; ~5 s/frame on CPU) |
| `procfun_deepface` | faces | deepface, YuNet backend |
| `procfun_nudenet` | exposed breasts/genitalia/buttocks/anus | NudeNet |
| `procfun_hyperlpr` | license plates | HyperLPR3 (trained on Chinese plates) |
| `procfun_debug2` | 10 random boxes, only with `--debug2` | — |

Thresholds and options are constants at the top of each `procfun_*.py` file.
A detector that fails to load prints a warning and is skipped.

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
