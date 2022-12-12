from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any

import click
import numpy as np
import sklearn.linear_model
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
    params: dict[str, Any] = yaml.safe_load(params_file.read_text())["analyze"]
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with np.load(input_file) as data:
        data: NDArray[np.float_] = data["data"]

    reg = sklearn.linear_model.LinearRegression(**params)
    X, y_true = np.split(data, [-1], axis=1)
    reg.fit(X, y_true)
    y_pred = reg.predict(X)

    np.savez_compressed(
        output_file,
        coef=reg.coef_,
        intercept=reg.intercept_,
        y_pred=y_pred,
        y_true=y_true,
    )


if __name__ == "__main__":
    main()
