from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any

import click
import numpy as np
import scipy.signal
import yaml

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
    params: dict[str, Any] = yaml.safe_load(params_file.read_text())["condition"]
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with np.load(input_file) as data:
        data_raw: NDArray[np.float_] = data["data"]

    window_length: int = params["window_length"]
    assert data_raw.shape[0] > window_length

    data = scipy.signal.savgol_filter(
        data_raw,
        axis=0,
        polyorder=3,
        window_length=window_length,
    )

    np.savez_compressed(output_file, data=data)


if __name__ == "__main__":
    main()
