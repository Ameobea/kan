import sqlite3
import itertools
import time
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tinygrad import Tensor
import pymde


def create_database(db_path: str):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slice_count INTEGER NOT NULL,
                use_embedding BOOLEAN NOT NULL,
                slice_ix_channels_per_dim INTEGER NOT NULL,
                coord_channels_per_dim INTEGER NOT NULL,
                batch_size INTEGER NOT NULL,
                learning_rate REAL NOT NULL,
                epochs INTEGER NOT NULL,
                model_repr TEXT NOT NULL,
                eval_loss REAL NOT NULL,
                training_time REAL NOT NULL,
                param_count INTEGER NOT NULL,
                run_end TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()


def insert_run(
    db_path: str,
    model: Any,
    params: Dict[str, Any],
    eval_loss: float,
    training_time: float,
):
    param_count = model.param_count()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO runs (
                slice_count, use_embedding, slice_ix_channels_per_dim, coord_channels_per_dim,
                batch_size, learning_rate, epochs, eval_loss, training_time, model_repr, param_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                params["slice_count"],
                params["use_embedding"],
                params["slice_ix_channels_per_dim"],
                params["coord_channels_per_dim"],
                params["batch_size"],
                params["learning_rate"],
                params["epochs"],
                eval_loss,
                training_time,
                repr(model),
                param_count,
            ),
        )
        conn.commit()


def get_run(
    db_path: str, model: Any, params: Dict[str, Any]
) -> Optional[Tuple[float, float]]:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT eval_loss, training_time
            FROM runs
            WHERE
                slice_count = ? AND
                use_embedding = ? AND
                slice_ix_channels_per_dim = ? AND
                coord_channels_per_dim = ? AND
                batch_size = ? AND
                learning_rate = ? AND
                epochs = ? AND
                model_repr = ?
            """,
            (
                params["slice_count"],
                params["use_embedding"],
                params["slice_ix_channels_per_dim"],
                params["coord_channels_per_dim"],
                params["batch_size"],
                params["learning_rate"],
                params["epochs"],
                repr(model),
            ),
        )
        return cursor.fetchone()


def generate_param_grid(param_bounds: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_bounds.keys())
    values = list(param_bounds.values())

    param_grid = list(itertools.product(*values))
    return [{keys[i]: value for i, value in enumerate(params)} for params in param_grid]


def grid_search(
    build_model: Callable,
    train_model: Callable,
    eval_model_full: Callable,
    load_training_data: Callable,
    db_path: str,
    static_params: Dict[str, Any],
    dynamic_param_bounds: Dict[str, Tuple[Any, Any, int]],
):
    create_database(db_path)

    param_grid = generate_param_grid(dynamic_param_bounds)
    print(param_grid)
    total_runs = len(param_grid)
    print(f"Starting grid search with {total_runs} runs...")

    for i, dynamic_params in enumerate(param_grid):
        np.random.seed(0)
        Tensor.manual_seed(0)
        pymde.seed(0)

        params = {**static_params, **dynamic_params}

        training_data = load_training_data(slice_count=params["slice_count"])
        model, embedding = build_model(
            training_data,
            use_embedding=params["use_embedding"],
            slice_ix_channels_per_dim=params["slice_ix_channels_per_dim"],
            coord_channels_per_dim=params["coord_channels_per_dim"],
            FirstLayer=params["FirstLayer"],
            Layer=params["Layer"],
            hidden_layer_defs=params["hidden_layer_defs"],
            base_layer_params=params["base_layer_params"],
        )

        print(f"\nRun {i + 1}/{total_runs}:")
        print(dynamic_params)
        print(f"Param count: {model.param_count()}")

        # check if run already exists for these parameters
        run = get_run(db_path, model, params)
        if run is not None:
            print("Found existing run for these parameters")
            continue

        start_time = time.time()
        train_model(
            training_data,
            model,
            embedding,
            slice_ix_channels_per_dim=params["slice_ix_channels_per_dim"],
            coord_channels_per_dim=params["coord_channels_per_dim"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            epochs=params["epochs"],
            quiet=True,
        )
        training_time = time.time() - start_time

        eval_loss = eval_model_full(
            model,
            embedding,
            training_data,
            params["slice_count"],
            params["slice_ix_channels_per_dim"],
            params["coord_channels_per_dim"],
        )

        insert_run(db_path, model, params, eval_loss, training_time)

        print(f"Eval Loss: {eval_loss}")
        print(f"Training Time: {training_time:.2f} seconds\n")
