import pathlib
from typing import Any

import click
import numpy as np
import pandas as pd
import yaml


@click.command()
@click.argument(
    "INPUT_FILE",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path),
)
@click.argument(
    "OUTPUT_FILE",
    type=click.Path(dir_okay=False, writable=True, path_type=pathlib.Path),
)
@click.option(
    "--params-file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path),
    default="params.yaml",
)
def main(
    input_file: pathlib.Path, output_file: pathlib.Path, params_file: pathlib.Path
):
    params: dict[str, Any] = yaml.safe_load(params_file.read_text())["load"]
    output_file.parent.mkdir(exist_ok=True, parents=True)

    df_raw = pd.read_csv(
        input_file,
        chunksize=None,
        iterator=False,
        **params,
    )

    # Do basic data wrangling things?

    np.savez_compressed(output_file, data=df_raw.values)


if __name__ == "__main__":
    main()
