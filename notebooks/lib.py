from pathlib import Path


ws = Path('workspaces')
if ws.exists():
    wd =ws /os.getenv('RepositoryName')
else:
    wd = Path(__file__).parent.parent

vd = wd/'video' # video_directory

assert vd.exists()
