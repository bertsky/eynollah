import click

@click.command(context_settings=dict(
    help_option_names=['-h', '--help'],
    show_default=True))
@click.option(
    "--input",
    "-i",
    help="PAGE-XML input filename",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--dir_in",
    "-di",
    help="directory of PAGE-XML input files (instead of --input)",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--out",
    "-o",
    help="directory for output images",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--overwrite",
    "-O",
    help="overwrite (instead of skipping) if output xml exists",
    is_flag=True,
)
@click.pass_context
def readingorder_cli(ctx, input, dir_in, out, overwrite):
    """
    Generate ReadingOrder from ML model
    """
    from ..reorder import Reorder
    assert bool(input) != bool(dir_in), "Either -i (single input) or -di (directory) must be provided, but not both."
    orderer = Reorder(model_zoo=ctx.obj.model_zoo,
                      device=ctx.obj.device)
    orderer.run(overwrite=overwrite,
                xml_filename=input,
                dir_in=dir_in,
                dir_out=out,
    )

