"""Fit and plot language-model scaling laws from validation losses."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Callable

if "MPLCONFIGDIR" not in os.environ:
    config_root = Path(
        os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
    )
    if not os.access(config_root, os.W_OK):
        os.environ["MPLCONFIGDIR"] = str(
            Path(tempfile.gettempdir()) / f"pavullmo-matplotlib-{os.getuid()}"
        )

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.stats import qmc


REQUIRED_COLUMNS = {
    "tokenizer",
    "parameters",
    "train_tokens",
    "validation_loss",
}
HUBER_DELTA = 0.05
COEFFICIENT_BOUNDS = (1e-6, 1e7)
EXPONENT_BOUNDS = (0.01, 2.0)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset" / "ds"


@dataclass(frozen=True)
class ScalingData:
    tokenizer: str
    parameters: np.ndarray
    train_tokens: np.ndarray
    validation_loss: np.ndarray
    parameter_reference: float
    token_reference: float

    @property
    def normalized_parameters(self) -> np.ndarray:
        return self.parameters / self.parameter_reference

    @property
    def normalized_tokens(self) -> np.ndarray:
        return self.train_tokens / self.token_reference


@dataclass(frozen=True)
class LawFit:
    name: str
    result: OptimizeResult
    normalized_coefficients: dict[str, float]
    raw_coefficients: dict[str, float]
    mape: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit Skaling and Chinchilla laws for one tokenizer and plot the "
            "data- and model-scaling curves."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="CSV containing tokenizer, parameters, train_tokens, validation_loss",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Value from the CSV tokenizer column, for example 8k",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional output image path; when omitted, open an interactive "
            "Matplotlib window"
        ),
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        help="Optional path for fitted coefficients and diagnostics",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=256,
        help="Number of Sobol-initialized L-BFGS-B fits per law (default: 256)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sobol initialization seed (default: 42)",
    )
    parser.add_argument(
        "--max-floor",
        type=float,
        default=3.0,
        help="Upper bound for irreducible loss E (default: 3.0)",
    )
    parser.add_argument(
        "--extrapolation-factor",
        type=float,
        default=2.0,
        help=(
            "Extend plots this far beyond the largest observed N and D; "
            "1 disables extrapolation (default: 2.0)"
        ),
    )
    parser.add_argument(
        "--plot-bpb",
        action="store_true",
        help=(
            "Plot validation bits per source byte instead of raw validation "
            "loss; fitting remains in nats per token"
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=(
            "Dataset artifact root used by --plot-bpb "
            f"(default: {DEFAULT_DATASET_DIR})"
        ),
    )
    return parser.parse_args()


def geometric_mean(values: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(values))))


def validation_bpb_scale(
    dataset_dir: Path, tokenizer: str
) -> tuple[float, int, int]:
    dataset_root = dataset_dir.expanduser().resolve()
    validation_dir = (dataset_root / f"validation_{tokenizer}").resolve()
    if validation_dir.parent != dataset_root:
        raise ValueError(f"invalid tokenizer dataset name: {tokenizer!r}")

    metadata_path = validation_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"validation metadata for tokenizer {tokenizer!r} not found: "
            f"{metadata_path}"
        )
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    token_count = metadata.get("token_count")
    source_byte_count = metadata.get("source_text_byte_count")
    if (
        not isinstance(token_count, int)
        or isinstance(token_count, bool)
        or token_count <= 0
    ):
        raise ValueError(f"invalid token_count in {metadata_path}")
    if (
        not isinstance(source_byte_count, int)
        or isinstance(source_byte_count, bool)
        or source_byte_count <= 0
    ):
        raise ValueError(f"invalid source_text_byte_count in {metadata_path}")

    scale = token_count / (source_byte_count * math.log(2.0))
    return scale, token_count, source_byte_count


def read_scaling_data(csv_path: Path, tokenizer: str) -> ScalingData:
    if not csv_path.is_file():
        raise FileNotFoundError(f"scaling-loss CSV does not exist: {csv_path}")

    rows: list[tuple[float, float, float]] = []
    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(sorted(missing))}"
            )

        for line_number, row in enumerate(reader, start=2):
            if row["tokenizer"].strip().casefold() != tokenizer.casefold():
                continue
            try:
                parameters = float(row["parameters"])
                train_tokens = float(row["train_tokens"])
                validation_loss = float(row["validation_loss"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid numeric value in {csv_path} line {line_number}"
                ) from error

            values = (parameters, train_tokens, validation_loss)
            if not all(math.isfinite(value) and value > 0.0 for value in values):
                raise ValueError(
                    f"all numeric values must be finite and positive in "
                    f"{csv_path} line {line_number}"
                )
            rows.append(values)

    if not rows:
        raise ValueError(f"no rows found for tokenizer {tokenizer!r} in {csv_path}")

    observations = np.asarray(rows, dtype=np.float64)
    parameters = observations[:, 0]
    train_tokens = observations[:, 1]
    validation_loss = observations[:, 2]
    unique_points = np.unique(observations[:, :2], axis=0)

    if len(unique_points) < 7:
        raise ValueError(
            "Skaling has six free parameters; provide at least seven distinct "
            f"(parameters, train_tokens) points, found {len(unique_points)}"
        )
    if len(np.unique(parameters)) < 2 or len(np.unique(train_tokens)) < 2:
        raise ValueError("the selected tokenizer needs variation in both N and D")

    return ScalingData(
        tokenizer=tokenizer,
        parameters=parameters,
        train_tokens=train_tokens,
        validation_loss=validation_loss,
        parameter_reference=geometric_mean(parameters),
        token_reference=geometric_mean(train_tokens),
    )


def predict(
    theta: np.ndarray,
    normalized_parameters: np.ndarray,
    normalized_tokens: np.ndarray,
    fixed_k: float | None,
) -> np.ndarray:
    log_a, log_b, alpha, beta = theta[:4]
    if fixed_k is None:
        k = theta[4]
        floor = theta[5]
    else:
        k = fixed_k
        floor = theta[4]

    size_term = np.exp(log_a) * normalized_parameters ** (-alpha)
    data_term = np.exp(log_b) * normalized_tokens ** (-beta)
    return (size_term + data_term) ** k + floor


def make_objective(
    data: ScalingData,
    fixed_k: float | None,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    normalized_parameters = data.normalized_parameters
    normalized_tokens = data.normalized_tokens
    target = data.validation_loss
    log_parameters = np.log(normalized_parameters)
    log_tokens = np.log(normalized_tokens)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        log_a, log_b, alpha, beta = theta[:4]
        if fixed_k is None:
            k = theta[4]
            floor = theta[5]
        else:
            k = fixed_k
            floor = theta[4]

        size_term = np.exp(log_a - alpha * log_parameters)
        data_term = np.exp(log_b - beta * log_tokens)
        inner = size_term + data_term
        reducible = inner**k
        prediction = reducible + floor
        residual = np.log(prediction) - np.log(target)

        absolute_residual = np.abs(residual)
        huber = np.where(
            absolute_residual <= HUBER_DELTA,
            0.5 * residual**2,
            HUBER_DELTA * (absolute_residual - 0.5 * HUBER_DELTA),
        )
        huber_gradient = np.where(
            absolute_residual <= HUBER_DELTA,
            residual,
            HUBER_DELTA * np.sign(residual),
        )

        common = huber_gradient / prediction
        inner_power_gradient = k * inner ** (k - 1.0)
        prediction_gradients = [
            inner_power_gradient * size_term,
            inner_power_gradient * data_term,
            -inner_power_gradient * size_term * log_parameters,
            -inner_power_gradient * data_term * log_tokens,
        ]
        if fixed_k is None:
            prediction_gradients.append(reducible * np.log(inner))
        prediction_gradients.append(np.ones_like(prediction))

        gradient = np.asarray(
            [np.mean(common * value) for value in prediction_gradients],
            dtype=np.float64,
        )
        return float(np.mean(huber)), gradient

    return objective


def sobol_starts(
    bounds: list[tuple[float, float]], restarts: int, seed: int
) -> np.ndarray:
    sampler = qmc.Sobol(d=len(bounds), scramble=True, seed=seed)
    exponent = math.ceil(math.log2(restarts))
    unit_starts = sampler.random_base2(exponent)[:restarts]
    lower = np.asarray([bound[0] for bound in bounds])
    upper = np.asarray([bound[1] for bound in bounds])
    return qmc.scale(unit_starts, lower, upper)


def fit_law(
    data: ScalingData,
    *,
    fixed_k: float | None,
    restarts: int,
    seed: int,
    max_floor: float,
) -> LawFit:
    floor_upper = min(max_floor, float(np.min(data.validation_loss)) - 1e-9)
    if floor_upper <= 0.0:
        raise ValueError("--max-floor and validation losses must permit E > 0")

    log_coefficient_bounds = tuple(np.log(COEFFICIENT_BOUNDS))
    bounds = [
        log_coefficient_bounds,
        log_coefficient_bounds,
        EXPONENT_BOUNDS,
        EXPONENT_BOUNDS,
    ]
    if fixed_k is None:
        bounds.append(EXPONENT_BOUNDS)
    bounds.append((0.0, floor_upper))

    objective = make_objective(data, fixed_k)
    best: OptimizeResult | None = None
    for start in sobol_starts(bounds, restarts, seed):
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 5_000, "ftol": 1e-14, "gtol": 1e-10},
        )
        if math.isfinite(float(result.fun)) and (
            best is None or result.fun < best.fun
        ):
            best = result

    if best is None:
        law_name = "Skaling" if fixed_k is None else "Chinchilla"
        raise RuntimeError(f"all optimization attempts failed for {law_name}")

    theta = np.asarray(best.x)
    log_a, log_b, alpha, beta = theta[:4]
    if fixed_k is None:
        k = float(theta[4])
        floor = float(theta[5])
        law_name = "Skaling"
    else:
        k = fixed_k
        floor = float(theta[4])
        law_name = "Chinchilla"

    a = float(np.exp(log_a))
    b = float(np.exp(log_b))
    normalized_coefficients = {
        "a": a,
        "b": b,
        "alpha": float(alpha),
        "beta": float(beta),
        "k": float(k),
        "E": floor,
    }
    raw_coefficients = {
        "A": a * data.parameter_reference ** float(alpha),
        "B": b * data.token_reference ** float(beta),
        "alpha": float(alpha),
        "beta": float(beta),
        "k": float(k),
        "E": floor,
    }
    fitted = predict(
        theta,
        data.normalized_parameters,
        data.normalized_tokens,
        fixed_k,
    )
    mape = float(np.mean(np.abs(fitted - data.validation_loss) / data.validation_loss))

    return LawFit(
        name=law_name,
        result=best,
        normalized_coefficients=normalized_coefficients,
        raw_coefficients=raw_coefficients,
        mape=100.0 * mape,
    )


def human_count(value: float, _position: float | None = None) -> str:
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if value >= scale:
            return f"{value / scale:g}{suffix}"
    return f"{value:g}"


def plot_scaling_curves(
    data: ScalingData,
    fits: list[LawFit],
    destination: Path | None,
    extrapolation_factor: float,
    validation_scale: float,
    validation_metric: str,
) -> None:
    minimum_parameters = float(np.min(data.parameters))
    minimum_tokens = float(np.min(data.train_tokens))
    data_band = data.parameters == minimum_parameters
    model_band = data.train_tokens == minimum_tokens
    if np.count_nonzero(data_band) < 2 or np.count_nonzero(model_band) < 2:
        raise ValueError(
            "plotting requires at least two D-band points at minimum N and two "
            "N-band points at minimum D"
        )

    token_grid = np.geomspace(
        float(np.min(data.train_tokens)),
        float(np.max(data.train_tokens)) * extrapolation_factor,
        500,
    )
    parameter_grid = np.geomspace(
        float(np.min(data.parameters)),
        float(np.max(data.parameters)) * extrapolation_factor,
        500,
    )

    surface_parameters = np.geomspace(
        float(np.min(data.parameters)),
        float(np.max(data.parameters)) * extrapolation_factor,
        250,
    )
    surface_tokens = np.geomspace(
        float(np.min(data.train_tokens)),
        float(np.max(data.train_tokens)) * extrapolation_factor,
        250,
    )
    parameter_mesh, token_mesh = np.meshgrid(
        surface_parameters, surface_tokens
    )

    fit_by_name = {fit.name: fit for fit in fits}
    surfaces: dict[str, np.ndarray] = {}
    for law_name in ("Skaling", "Chinchilla"):
        fit = fit_by_name[law_name]
        fixed_k = 1.0 if law_name == "Chinchilla" else None
        surfaces[law_name] = predict(
            np.asarray(fit.result.x),
            parameter_mesh / data.parameter_reference,
            token_mesh / data.token_reference,
            fixed_k,
        )

    unique_parameters = np.unique(data.parameters)
    unique_tokens = np.unique(data.train_tokens)
    discrete_parameters, discrete_tokens = np.meshgrid(
        unique_parameters, unique_tokens
    )
    discrete_pairs = np.column_stack(
        (discrete_parameters.ravel(), discrete_tokens.ravel())
    )
    observed_pairs = set(zip(data.parameters, data.train_tokens, strict=True))
    unobserved_pairs = np.asarray(
        [
            pair
            for pair in discrete_pairs
            if (pair[0], pair[1]) not in observed_pairs
        ],
        dtype=np.float64,
    )

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    curve_axes = axes[0]
    surface_axes = axes[1]
    colors = {"Skaling": "#0068b5", "Chinchilla": "#d1495b"}
    linestyles = {"Skaling": "-", "Chinchilla": "--"}

    curve_axes[0].scatter(
        data.train_tokens[data_band],
        data.validation_loss[data_band] * validation_scale,
        color="black",
        marker="o",
        s=38,
        label="Observed",
        zorder=3,
    )
    curve_axes[1].scatter(
        data.parameters[model_band],
        data.validation_loss[model_band] * validation_scale,
        color="black",
        marker="o",
        s=38,
        label="Observed",
        zorder=3,
    )

    for fit in fits:
        fixed_k = 1.0 if fit.name == "Chinchilla" else None
        data_curve = predict(
            np.asarray(fit.result.x),
            np.full_like(token_grid, minimum_parameters / data.parameter_reference),
            token_grid / data.token_reference,
            fixed_k,
        ) * validation_scale
        model_curve = predict(
            np.asarray(fit.result.x),
            parameter_grid / data.parameter_reference,
            np.full_like(parameter_grid, minimum_tokens / data.token_reference),
            fixed_k,
        ) * validation_scale
        label = f"{fit.name} (MAPE {fit.mape:.2f}%)"
        curve_axes[0].plot(
            token_grid,
            data_curve,
            color=colors[fit.name],
            linestyle=linestyles[fit.name],
            linewidth=2,
            label=label,
        )
        curve_axes[1].plot(
            parameter_grid,
            model_curve,
            color=colors[fit.name],
            linestyle=linestyles[fit.name],
            linewidth=2,
            label=label,
        )

    curve_axes[0].set_title(
        f"Data scaling at N={human_count(minimum_parameters)}"
    )
    curve_axes[0].set_xlabel("Training tokens D")
    curve_axes[0].set_ylabel(validation_metric)
    curve_axes[1].set_title(
        f"Model scaling at D={human_count(minimum_tokens)} tokens"
    )
    curve_axes[1].set_xlabel("Non-embedding parameters N")
    curve_axes[1].set_ylabel(validation_metric)

    for axis in curve_axes:
        axis.set_xscale("log")
        axis.xaxis.set_major_formatter(FuncFormatter(human_count))
        axis.grid(True, which="both", alpha=0.25)

    if extrapolation_factor > 1.0:
        curve_axes[0].axvspan(
            float(np.max(data.train_tokens)),
            float(np.max(data.train_tokens)) * extrapolation_factor,
            color="black",
            alpha=0.06,
            label="Extrapolation",
        )
        curve_axes[1].axvspan(
            float(np.max(data.parameters)),
            float(np.max(data.parameters)) * extrapolation_factor,
            color="black",
            alpha=0.06,
            label="Extrapolation",
        )

    for axis in curve_axes:
        axis.legend()

    surface_minimum = min(
        float(np.min(data.validation_loss)) * validation_scale,
        *(
            float(np.min(surface)) * validation_scale
            for surface in surfaces.values()
        ),
    )
    surface_maximum = max(
        float(np.max(data.validation_loss)) * validation_scale,
        *(
            float(np.max(surface)) * validation_scale
            for surface in surfaces.values()
        ),
    )
    contour_levels = np.linspace(surface_minimum, surface_maximum, 24)
    contour_plot = None
    for axis, law_name in zip(
        surface_axes, ("Skaling", "Chinchilla"), strict=True
    ):
        contour_plot = axis.contourf(
            parameter_mesh,
            token_mesh,
            surfaces[law_name] * validation_scale,
            levels=contour_levels,
            cmap="viridis_r",
            extend="both",
        )
        axis.scatter(
            data.parameters,
            data.train_tokens,
            c=data.validation_loss * validation_scale,
            cmap="viridis_r",
            vmin=surface_minimum,
            vmax=surface_maximum,
            edgecolors="black",
            linewidths=0.8,
            s=48,
            label="Observed runs",
            zorder=3,
            clip_on=False,
        )
        if unobserved_pairs.size:
            axis.scatter(
                unobserved_pairs[:, 0],
                unobserved_pairs[:, 1],
                color="white",
                edgecolors="black",
                linewidths=0.7,
                marker="X",
                s=38,
                label="Unobserved grid points",
                zorder=2,
                clip_on=False,
            )
        if extrapolation_factor > 1.0:
            observed_minimum_parameters = float(np.min(data.parameters))
            observed_maximum_parameters = float(np.max(data.parameters))
            observed_minimum_tokens = float(np.min(data.train_tokens))
            observed_maximum_tokens = float(np.max(data.train_tokens))
            axis.plot(
                [
                    observed_minimum_parameters,
                    observed_maximum_parameters,
                    observed_maximum_parameters,
                    observed_minimum_parameters,
                    observed_minimum_parameters,
                ],
                [
                    observed_minimum_tokens,
                    observed_minimum_tokens,
                    observed_maximum_tokens,
                    observed_maximum_tokens,
                    observed_minimum_tokens,
                ],
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="Observed N-D range",
                zorder=4,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.xaxis.set_major_formatter(FuncFormatter(human_count))
        axis.yaxis.set_major_formatter(FuncFormatter(human_count))
        axis.set_title(f"{law_name} predicted {validation_metric.lower()} surface")
        axis.set_xlabel("Non-embedding parameters N")
        axis.set_ylabel("Training tokens D")
        axis.legend(loc="upper right")

    assert contour_plot is not None
    figure.colorbar(
        contour_plot,
        ax=list(surface_axes),
        label=f"Predicted {validation_metric.lower()}",
        shrink=0.9,
    )

    figure.suptitle(
        f"Scaling-law fits for tokenizer {data.tokenizer} "
        f"({len(data.validation_loss)} observations, {validation_metric})"
    )
    if destination is None:
        plt.show()
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180)
    plt.close(figure)


def fit_summary(data: ScalingData, fits: list[LawFit]) -> dict[str, object]:
    return {
        "tokenizer": data.tokenizer,
        "observations": len(data.validation_loss),
        "unique_parameters": len(np.unique(data.parameters)),
        "unique_train_tokens": len(np.unique(data.train_tokens)),
        "reference_scales": {
            "parameters": data.parameter_reference,
            "train_tokens": data.token_reference,
        },
        "fits": {
            fit.name.lower(): {
                "objective": float(fit.result.fun),
                "optimizer_success": bool(fit.result.success),
                "optimizer_message": str(fit.result.message),
                "mape_percent": fit.mape,
                "normalized_coefficients": fit.normalized_coefficients,
                "raw_coefficients": fit.raw_coefficients,
            }
            for fit in fits
        },
    }


def main() -> None:
    args = parse_args()
    if args.restarts <= 0:
        raise ValueError("--restarts must be greater than zero")
    if not math.isfinite(args.max_floor) or args.max_floor <= 0.0:
        raise ValueError("--max-floor must be finite and greater than zero")
    if (
        not math.isfinite(args.extrapolation_factor)
        or args.extrapolation_factor < 1.0
    ):
        raise ValueError("--extrapolation-factor must be finite and at least 1")

    data = read_scaling_data(args.csv, args.tokenizer)
    validation_scale = 1.0
    validation_metric = "Validation loss"
    validation_metadata: dict[str, object] | None = None
    if args.plot_bpb:
        validation_scale, validation_tokens, validation_bytes = (
            validation_bpb_scale(args.dataset_dir, args.tokenizer)
        )
        validation_metric = "Validation BPB"
        validation_metadata = {
            "token_count": validation_tokens,
            "source_text_byte_count": validation_bytes,
            "bpb_per_nat_per_token": validation_scale,
        }
    fits = [
        fit_law(
            data,
            fixed_k=None,
            restarts=args.restarts,
            seed=args.seed,
            max_floor=args.max_floor,
        ),
        fit_law(
            data,
            fixed_k=1.0,
            restarts=args.restarts,
            seed=args.seed + 1,
            max_floor=args.max_floor,
        ),
    ]

    summary = fit_summary(data, fits)
    summary["plot"] = {
        "extrapolation_factor": args.extrapolation_factor,
        "metric": "validation_bpb" if args.plot_bpb else "validation_loss",
    }
    if validation_metadata is not None:
        summary["plot"]["validation_metadata"] = validation_metadata
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))

    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote fit results to {args.results_json.resolve()}")

    plot_scaling_curves(
        data,
        fits,
        args.output,
        args.extrapolation_factor,
        validation_scale,
        validation_metric,
    )
    if args.output is not None:
        print(f"Wrote scaling plot to {args.output.resolve()}")


if __name__ == "__main__":
    main()
