import shutil

import typer
from rich import print

from railgen.constants import INPUT_DATA_DIR, RAIL_NETWORK_DIR


def clean():
    """Delete all downloaded input data and generated rail network files."""
    dirs = [d for d in (INPUT_DATA_DIR, RAIL_NETWORK_DIR) if d.exists()]
    if not dirs:
        print("Nothing to clean.")
        return
    print("[bold red]WARNING[/bold red] This will delete the following files:\n")
    for d in dirs:
        print(f"[blue]{d}/[/blue]: {', '.join(sorted(p.name for p in d.iterdir()))}")
    typer.confirm("\nRemove files?", default=False, abort=True)
    for d in dirs:
        shutil.rmtree(d)
    print(f"Deleted {', '.join(map(str, dirs))}")
