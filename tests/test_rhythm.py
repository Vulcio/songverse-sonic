"""The baseline can run without API, job storage, or full music analysis."""
import subprocess
import sys
from pathlib import Path


def test_rhythm_is_independent_of_service_and_optional_models(tmp_path):
    # A fresh interpreter prevents imports from other tests hiding dependencies.
    app_dir = Path(__file__).resolve().parent.parent / "app"
    synth_dir = Path(__file__).resolve().parent
    code = """
import sys
sys.path[:0] = sys.argv[1:3]
import rhythm
assert not {'main', 'analysis', 'fastapi', 'boto3', 'torch', 'allin1'} & sys.modules.keys()
import numpy as np
import soundfile as sf
import synth
from pathlib import Path
root = Path(sys.argv[3])
synth.make_song(root, bpm=100, intro_bars=2, bars=12)
stems = {p.stem: sf.read(p, dtype='float32')[0] for p in root.glob('*.wav')}
mix = np.sum(list(stems.values()), axis=0)
result = rhythm.analyze_rhythm(mix, stems['drums'] + stems['bass'], synth.SR,
                               stems['drums'], stems['bass'] + stems['other'])
assert abs(result['tempo']['bpm'] - 100) < 4
assert result['first_beat'] == result['downbeats'][0]
assert set(result['downbeats']) <= set(result['beats'])
assert result['first_beat'] < 2.4  # Includes the drumless intro, not only drum entry.
assert 'key' not in result and 'sections' not in result
assert not {'main', 'analysis', 'fastapi', 'boto3', 'torch', 'allin1'} & sys.modules.keys()
"""
    subprocess.run(
        [sys.executable, "-c", code, str(app_dir), str(synth_dir), str(tmp_path)],
        check=True, capture_output=True, text=True, timeout=120,
    )
