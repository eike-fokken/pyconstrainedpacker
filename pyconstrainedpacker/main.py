import click


@click.command()
# @click.option("--count", default=1, help="Number of greetings.")
# @click.option("--name", prompt="Your name", help="The person to greet.")
def optimize_distribution() -> None:
    """Optimize food distribution on constrained supply."""
    pass


if __name__ == "__main__":
    optimize_distribution()
