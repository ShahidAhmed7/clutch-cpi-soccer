"""Run the pipeline.

    python run_all.py           # everything from the raw data (Stage 3 uses a GPU if present, about 25 minutes on a Colab T4)
    python run_all.py --quick   # from the stored per-action data in data/: ratings, tests, figures, numbers
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FULL = [['s1_actions.py'], ['s2_features.py'], ['s3_action_values.py', '--model', 'catboost'], ['s4_stakes.py'], ['s5_values.py']]
ANALYSIS = [['s6_ratings.py'], ['s7_validation.py'], ['s8_pressure.py'], ['s9_coaches.py'], ['s11_stories.py'], ['s13_figures.py'], ['s14_claims.py']]


def main():
    (ROOT / 'work').mkdir(exist_ok=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    if '--quick' in sys.argv:
        for f in (ROOT / 'data').glob('*'):
            if f.is_dir():
                shutil.copytree(f, ROOT / 'work' / f.name, dirs_exist_ok=True)
            else:
                shutil.copy(f, ROOT / 'work' / f.name)
        stages = ANALYSIS
    else:
        stages = FULL + ANALYSIS
    for stage in stages:
        print('---', ' '.join(stage), flush=True)
        subprocess.run([sys.executable, str(ROOT / 'pipeline' / stage[0])] + stage[1:], check=True)
        if stage[0] == 's2_features.py':            # data contracts: a broken table stops the run here
            subprocess.run([sys.executable, '-m', 'pytest', str(ROOT / 'tests'), '-q'], check=True, cwd=ROOT)
    print('done. Results in results/, paper sources in paper/.')


if __name__ == '__main__':
    main()
