from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Any, Optional

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import yaml

if TYPE_CHECKING:
    from numpy.typing import NDArray


@click.command()
@click.argument(
    "MODEL_DIR",
    type=click.Path(
        exists=True, file_okay=False, readable=True, path_type=pathlib.Path
    ),
)
@click.argument(
    "METRICS_FILE",
    type=click.Path(dir_okay=False, writable=True, path_type=pathlib.Path),
)
@click.option(
    "--plot",
    "plot_file",
    type=click.Path(dir_okay=False, writable=True, path_type=pathlib.Path),
)
@click.option(
    "--params-file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path),
    default="params.yaml",
)
@click.option(
    "--sessions-file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path),
    default="../sessions.yaml",
)
def main(
    model_dir: pathlib.Path,
    metrics_file: pathlib.Path,
    params_file: pathlib.Path,
    sessions_file: pathlib.Path,
    plot_file: Optional[pathlib.Path],
):
    params: dict[str, Any] = yaml.safe_load(params_file.read_text())["aggregate"]
    sessions: dict[str, dict[str, str]] = yaml.safe_load(sessions_file.read_text())[
        "sessions"
    ]
    metrics_file.parent.mkdir(exist_ok=True, parents=True)

    data_coef, data_intercept = map(
        np.array,
        zip(
            *(
                [
                    (
                        data_session := np.load(
                            model_dir
                            / session_info["participant_id"]
                            / f"{session_info['session_number']}.npz"
                        )
                    )["coef"],
                    data_session["intercept"],
                ]
                for session_info in sessions.values()
            )
        ),
    )
    del data_session

    data_coef = data_coef.squeeze(axis=1)
    assert data_coef.ndim == 2
    assert data_coef.shape[0] == len(sessions)
    assert data_intercept.shape == (len(sessions), 1)

    df_params = pd.DataFrame(
        np.hstack([data_coef, data_intercept]),
        columns=[
            *(f"coef_{idx_coef}" for idx_coef in range(data_coef.shape[1])),
            "intercept",
        ],
    )
    metrics = (
        df_params.apply(
            lambda x: pd.Series(
                scipy.stats.ranksums(x, y=0, **params["ranksums"]),
                index=["stat", "p_value"],
            ),
            axis=0,
        )
        .round(params["precision"])
        .to_dict()
    )
    metrics_file.write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
    )

    if not plot_file:
        return

    plot_file.parent.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True, **params["plot"])
    df_params.boxplot(ax=ax)
    ax.grid(False, axis="x")
    ax.set_title("Distribution of model parameters")
    fig.savefig(str(plot_file))


if __name__ == "__main__":
    main()
