"""Console script for cvutils."""

import click


@click.command()
def main():
    """Main entrypoint."""
    click.echo("cvutils")
    click.echo("=" * len("cvutils"))
    click.echo("Computer vision auxiliary rutines")


if __name__ == "__main__":
    main()  # pragma: no cover
