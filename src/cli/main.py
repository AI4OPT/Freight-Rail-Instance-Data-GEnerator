import typer

from cli.clean import clean
from cli.config import config_schema
from cli.network import build_network, list_yards
from cli.scenario import generate_scenarios
from cli.schedule import generate_schedule

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Generate synthetic train schedules and scenarios.",
    context_settings={"help_option_names": ["-h", "--help"]},
)


app.command("build-network")(build_network)
app.command("schedule")(generate_schedule)
app.command("scenario")(generate_scenarios)
app.command("yards")(list_yards)
app.command("clean")(clean)
app.command("config-schema")(config_schema)


if __name__ == "__main__":
    app()
