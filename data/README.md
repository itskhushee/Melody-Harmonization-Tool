# Data

## `raw/demo/`
A small set of generated MIDI melodies used for development and testing.
**Committed to the repo** — small enough to ship.

Regenerate with:
```bash
python scripts/generate_demo_midi.py
```

### `raw/demo/test/` (3 files, melody only)
For sanity-checking the MIDI parser and inspecting harmonizer output.

| File | Key | Description |
|---|---|---|
| `c_major_scale.mid` | C major | Scale up and down, quarter notes |
| `c_minor_scale.mid` | C natural minor | Scale up and down, quarter notes |
| `c_major_arpeggio.mid` | C major | Triadic arpeggio pattern |

### `raw/demo/train/` (5 files + annotations)
Early training data: melody + per-measure chord labels.

| File | Key | Notes |
|---|---|---|
| `twinkle_c_major.mid` | C major | Twinkle Twinkle Little Star |
| `mary_lamb_c_major.mid` | C major | Mary Had a Little Lamb |
| `ode_joy_c_major.mid` | C major | Ode to Joy theme |
| `cm_lament.mid` | C minor | Original C minor lament melody |
| `c_major_pop.mid` | C major | Melody over I-V-vi-IV ("axis") progression |

Chord annotations are in `train/chords.json` with this schema:

```json
{
  "<filename>.mid": {
    "key": "C major",
    "time_signature": "4/4",
    "chords_per_measure": ["C", "F", "F", "C", ...]
  }
}
```

All files are 4/4, 100 BPM, single melody track. The training melodies are
8 measures each. Chord labels are per-measure (one chord lasts the whole
measure), which keeps the segmentation simple to start with — we can revisit
this once the basic pipeline works.

## `raw/` (other corpora)
Source MIDI files and chord-annotation corpora as downloaded. Large files
should be gitignored. When you add a corpus, edit this README to record:
- Where it came from (URL, citation)
- Its license
- The version / date downloaded
- Any access restrictions

### Corpora under consideration
- _fill in as we pick them_

## `processed/`
The cleaned, parsed form that the trainer (`harmonizer.train`) consumes.
Files here are produced by the preprocessing pipeline and should be
reproducible from `raw/` via documented commands.
