"""CPI for any league, season or set of matches, from StatsBomb, Wyscout (v2 or v3), any kloppy provider, or your own tables.

The fitted models are included in models/. To refit them (needs the work/ files from python run_all.py, about 30 minutes):
    python run_league.py train

Get data (Hudl StatsBomb open data needs no account):
    python run_league.py fetch --competition "1. Bundesliga" --season 2023/2024
    python run_league.py list --data data_in/statsbomb                       # what a folder contains

Run:
    python run_league.py run --data data_in/statsbomb --competition "1. Bundesliga" --season 2023/2024
    python run_league.py run --data WyScout --provider wyscout-v2 --competition Spain
    python run_league.py run --data my_v3_matches/ --provider wyscout-v3 --name "Serie A 2020/21"
    python run_league.py run --data opta_manifest.csv --provider kloppy:opta --name "Eredivisie 2022/23"
    python run_league.py run --data my_tables/ --provider tables             # your own CSV/parquet tables (see cpi/league/bundle.py)

Options: --models auto|pretrained|retrain, --min-minutes N, --boot N (interval resamples), --out FOLDER, --workers N,
         --save-bundle (keep the converted tables next to the report, to rerun quickly with --provider tables),
         --keep-carries / --no-harmonize / --no-recalibrate (pretrained models: skip matching the 2017/18 Wyscout conventions,
         or the recalibration of predicted chances on this data).
Results go to runs/<name>/: report.md, player_ratings.csv, action_values.parquet, figures/ and more.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PROVIDERS = ['auto', 'statsbomb', 'wyscout-v2', 'wyscout-v3', 'tables', 'kloppy:<loader>']


def detect(path):
    """Guess the provider from the files in a folder (or a manifest CSV)."""
    p = Path(path)
    if p.is_file() and p.suffix == '.csv':
        raise ValueError('A CSV manifest needs --provider kloppy:<loader>, e.g. kloppy:opta')
    if (p / 'actions.parquet').exists() or (p / 'actions.csv').exists():
        return 'tables'
    for cand in (p, p / 'data'):
        if (cand / 'events').is_dir() and (cand / 'lineups').is_dir():
            return 'statsbomb'
    if list(p.glob('events_*.json')) or (p / 'events.zip').exists():
        return 'wyscout-v2'
    files = sorted(p.glob('*.json')) if p.is_dir() else [p]
    for f in files[:3]:
        head = f.read_text(encoding='utf-8', errors='ignore')[:20000]
        if '"primary"' in head and '"matchTimestamp"' in head:
            return 'wyscout-v3'
        if '"eventId"' in head and '"subEventId"' in head:
            return 'wyscout-v2'
    raise ValueError(f'Could not recognise the data in {path}. Choose --provider from {PROVIDERS}.')


def load_bundle(args, log):
    provider = args.provider if args.provider != 'auto' else detect(args.data)
    log(f'provider: {provider}')
    if provider == 'statsbomb':
        from cpi.league.providers import statsbomb
        return statsbomb.load(args.data, args.competition, args.season, args.name, args.workers, log)
    if provider == 'wyscout-v2':
        from cpi.league.providers import wyscout_v2
        return wyscout_v2.load(args.data, args.competition, args.name, args.workers, log)
    if provider == 'wyscout-v3':
        from cpi.league.providers import wyscout_v3
        return wyscout_v3.load(args.data, args.name, args.workers, log)
    if provider == 'tables':
        from cpi.league import bundle
        return bundle.load_folder(args.data, args.name)
    if provider.startswith('kloppy:'):
        from cpi.league.providers import kloppy_generic
        return kloppy_generic.load_manifest(args.data, provider.split(':', 1)[1], args.name, args.workers, log)
    raise ValueError(f'unknown provider {provider!r}; choose from {PROVIDERS}')


def slug(name):
    return re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_') or 'run'


def cmd_run(args):
    from cpi.league import engine, report
    t0 = time.time()
    log = lambda s: print(s, flush=True)  # noqa: E731
    b = load_bundle(args, log)
    out = Path(args.out) if args.out else ROOT / 'runs' / slug(b.name)
    if args.save_bundle:
        b.save(out / 'bundle')
    res = engine.run(b, models=args.models, min_minutes=args.min_minutes, n_boot=args.boot, workers=args.workers,
                     harmonize_actions=not args.no_harmonize, keep_carries=args.keep_carries, recalibrate_models=not args.no_recalibrate, log=log)
    info = report.write(res, out, data_path=args.data, log=log)
    ck = info['checks']
    print(f"\n{b.name}: {ck['matches']} matches, {ck['actions']:,} actions, models {info['mode']}, "
          f"scoring AUC {info['model_accuracy']['scores']['auc'] or float('nan'):.3f}, {len(res['ratings'])} rated players, {time.time() - t0:.0f}s")
    print(f'open {out / "report.md"}')


def cmd_list(args):
    provider = args.provider if args.provider != 'auto' else detect(args.data)
    if provider == 'statsbomb':
        from cpi.league.providers import statsbomb
        s = statsbomb.list_seasons(args.data)
        print(s.to_string(index=False) if len(s) else 'flat StatsBomb folder (no competitions.json)')
    elif provider == 'wyscout-v2':
        from cpi.league.providers import wyscout_v2
        print('competitions:', wyscout_v2.list_competitions(args.data) or 'per-match files')
    else:
        print(f'{provider}: every file in the folder is one match')


def cmd_fetch(args):
    from cpi.league.providers import statsbomb
    root, cid, sid = statsbomb.fetch_open_data(args.competition, args.season, args.out)
    print(f'next:  python run_league.py run --data {root} --competition {cid} --season {sid}')


def cmd_train(args):
    from cpi.league import models
    meta = models.train_all(log=lambda s: print(s, flush=True))
    print(json.dumps({k: meta[k] for k in ('trained_utc', 'paper_held_out_auc', 'best_iteration_scores', 'best_iteration_concedes')}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='compute CPI for a dataset and write a report')
    r.add_argument('--data', required=True, help='data folder, file, or kloppy manifest CSV')
    r.add_argument('--provider', default='auto', help=f'one of {PROVIDERS}')
    r.add_argument('--competition', help='StatsBomb competition name or id; Wyscout v2 file suffix (e.g. England)')
    r.add_argument('--season', help='StatsBomb season name (e.g. 2023/2024) or id')
    r.add_argument('--name', help='name for the report and output folder')
    r.add_argument('--models', default='auto', choices=['auto', 'pretrained', 'retrain'])
    r.add_argument('--min-minutes', type=float, default=None, help='minutes to be rated (default 900, scaled down for short datasets)')
    r.add_argument('--boot', type=int, default=1000, help='match resamples for the intervals')
    r.add_argument('--out', help='output folder (default runs/<name>)')
    r.add_argument('--workers', type=int, default=6)
    r.add_argument('--save-bundle', action='store_true', help='also save the converted tables (rerun with --provider tables)')
    r.add_argument('--keep-carries', action='store_true', help='pretrained models: keep StatsBomb carries instead of Wyscout-style dribbles')
    r.add_argument('--no-harmonize', action='store_true', help='pretrained models: use the provider actions exactly as converted')
    r.add_argument('--no-recalibrate', action='store_true', help='pretrained models: skip the cross-fitted recalibration on this data')
    r.set_defaults(func=cmd_run)
    li = sub.add_parser('list', help='list competitions and seasons in a data folder')
    li.add_argument('--data', required=True)
    li.add_argument('--provider', default='auto')
    li.set_defaults(func=cmd_list)
    f = sub.add_parser('fetch', help='download one season of Hudl StatsBomb open data')
    f.add_argument('--competition', required=True)
    f.add_argument('--season', required=True)
    f.add_argument('--out', default=str(ROOT / 'data_in' / 'statsbomb'))
    f.set_defaults(func=cmd_fetch)
    t = sub.add_parser('train', help='refit the saved models (needs work/ from python run_all.py)')
    t.set_defaults(func=cmd_train)
    args = ap.parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as e:          # wrong season, folder or format: say what is available, no traceback
        sys.exit(f'\nerror: {e}')


if __name__ == '__main__':
    main()
