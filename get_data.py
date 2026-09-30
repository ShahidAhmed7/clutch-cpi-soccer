"""Fetch the public Wyscout soccer-logs (Pappalardo et al., 2019; CC BY 4.0) into WyScout/ and unpack them.

    python get_data.py              # download anything missing, then unpack events/ and matches/
    python get_data.py --no-extract # download only

Source: figshare collection 4415000, https://doi.org/10.6084/m9.figshare.c.4415000
"""
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'WyScout'
WANTED = {'events.zip', 'matches.zip', 'players.json', 'teams.json', 'competitions.json'}
API = 'https://api.figshare.com/v2'


def get_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'clutch-cpi-soccer'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def download():
    ROOT.mkdir(exist_ok=True)
    for art in get_json(f'{API}/collections/4415000/articles?page_size=100'):
        for f in get_json(f"{API}/articles/{art['id']}")['files']:
            dest = ROOT / f['name']
            if f['name'] not in WANTED or (dest.exists() and dest.stat().st_size == f['size']):
                continue
            print(f"downloading {f['name']} ({f['size'] / 1e6:.1f} MB)")
            urllib.request.urlretrieve(f['download_url'], dest)


def extract():
    for name in ('events', 'matches'):
        out = ROOT / name
        if out.exists() and any(out.iterdir()):
            continue
        with zipfile.ZipFile(ROOT / f'{name}.zip') as z:
            z.extractall(out)
        print(f'unpacked {name}.zip -> {out}')


if __name__ == '__main__':
    download()
    if '--no-extract' not in sys.argv:
        extract()
    print('WyScout data ready in', ROOT)
