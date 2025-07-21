import click

@click.group()
def cli():
    """A command-line interface for the e8leech library."""
    pass

@cli.group()
def crypto():
    """Commands for cryptographic operations."""
    pass

@crypto.command()
@click.option('--algo', default='KYBER-E8', help='The encryption algorithm to use.')
def encrypt(algo):
    """Encrypts a message using the specified algorithm."""
    click.echo(f"Encrypting with {algo}...")

@cli.group()
def ai():
    """Commands for AI operations."""
    pass

@ai.command()
@click.option('--model', default='E8GNN', help='The AI model to use.')
def train(model):
    """Trains an AI model."""
    click.echo(f"Training {model}...")

@cli.group()
def visualize():
    """Commands for visualization."""
    pass

@visualize.command()
@click.option('--dim', default=24, help='The dimension of the space to visualize.')
@click.option('--projection', default='hologram', help='The projection to use.')
def show(dim, projection):
    """Visualizes a space."""
    click.echo(f"Visualizing {dim}D space with {projection} projection...")

if __name__ == '__main__':
    cli()
