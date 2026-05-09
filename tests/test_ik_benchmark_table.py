"""
Бенчмарк IK: таблица MSE, средние P/R/препятствие, время обучения, время решения,
плюс P/R после op_solve (уточнение LM).

Запуск (печать таблицы в stdout):
    MPLBACKEND=Agg pytest tests/test_ik_benchmark_table.py -s

Быстрый режим (меньше эпох/поколений):
    BENCHMARK_QUICK=1 MPLBACKEND=Agg pytest tests/test_ik_benchmark_table.py -s

Переменные окружения (опционально):
    BENCHMARK_GEN_POP, BENCHMARK_GEN_GEN, BENCHMARK_DDPG_EPISODES, BENCHMARK_DDPG_STEPS,
    BENCHMARK_ML_SAMPLES (число случайных конфигураций суставов для X,y ML),
    BENCHMARK_ML_TEST (отдельная выборка только для столбца MSE)
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robots.robot import Robot
from tests.ik_benchmark_utils import (
    benchmark_env_int,
    default_fixed_angle_rows,
    fixed_targets_from_angles,
    obstacle_exp_penalty_sum,
    orientation_error_fk,
    position_error_fk,
    standard_angle_limits,
    standard_dh_parameters,
    standard_obstacles,
)


@dataclass
class BenchmarkRow:
    algorithm: str
    mse: str
    p_error_mean: float
    r_error_mean: float
    obstacle_error_mean: float
    time_to_learn: str
    time_to_execute_mean: float
    p_error_refined_mean: float
    r_error_refined_mean: float


def _fmt(x: float, nd: int = 4) -> str:
    if np.isnan(x):
        return "nan"
    return f"{x:.{nd}f}"


def _fmt_refined(x: float) -> str:
    """Экспоненциальная запись (1.23e-05 ≡ 1.23·10⁻⁵) для малых ошибок после op_solve."""
    if not np.isfinite(x) or np.isnan(x):
        return "nan"
    return f"{x:.2e}"


def _fmt_time(s: float) -> str:
    if not np.isfinite(s):
        return "nan"
    return f"{s:.6f}s"


def _print_table(rows: list[BenchmarkRow]) -> None:
    headers = [
        "Algorithm",
        "MSE",
        "P_error_mean",
        "R_error_mean",
        "Obstacle_error_mean",
        "TimeToLearn",
        "TimeToExecute(mean)",
        "P_error_refined_mean",
        "R_error_refined_mean",
    ]
    cols = [headers]
    for r in rows:
        cols.append([
            r.algorithm,
            r.mse,
            _fmt(r.p_error_mean),
            _fmt(r.r_error_mean),
            _fmt(r.obstacle_error_mean),
            r.time_to_learn,
            _fmt_time(r.time_to_execute_mean),
            _fmt_refined(r.p_error_refined_mean),
            _fmt_refined(r.r_error_refined_mean),
        ])
    widths = [max(len(row[i]) for row in cols) for i in range(len(headers))]
    sep = " | "
    line = sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print("\n" + line)
    print("-+-".join("-" * w for w in widths))
    for row in cols[1:]:
        print(sep.join(row[i].ljust(widths[i]) for i in range(len(headers))))


def _quick() -> bool:
    return os.environ.get("BENCHMARK_QUICK", "").lower() in ("1", "true", "yes")


@pytest.fixture(scope="module")
def robot_and_targets():
    robot = Robot(standard_dh_parameters())
    bounds = standard_angle_limits()
    angle_rows = default_fixed_angle_rows()
    targets = fixed_targets_from_angles(robot, angle_rows)
    obstacles = standard_obstacles()
    return robot, bounds, targets, obstacles, angle_rows


def test_benchmark_table(robot_and_targets):
    robot, bounds, targets, obstacles, _angle_rows = robot_and_targets
    rows: list[BenchmarkRow] = []

    try:
        import torch  # noqa: F401
        torch_ok = True
    except ImportError:
        torch_ok = False

    quick = _quick()
    gen_pop = benchmark_env_int("BENCHMARK_GEN_POP", 80 if quick else 200)
    gen_gen = benchmark_env_int("BENCHMARK_GEN_GEN", 15 if quick else 40)
    ddpg_ep = benchmark_env_int("BENCHMARK_DDPG_EPISODES", 25 if quick else 80)
    ddpg_steps = benchmark_env_int("BENCHMARK_DDPG_STEPS", 60 if quick else 120)
    # ML: ml_n случайных конфигураций → X_tr,y_tr; в train() у RF/XGB/NN ещё split 80/20.
    # MSE в таблице — на отдельных ml_test конфигурациях (не пересекаются с 5 фикс. целями FK).
    ml_n = benchmark_env_int("BENCHMARK_ML_SAMPLES", 2500 if quick else 12_000)
    ml_test = benchmark_env_int("BENCHMARK_ML_TEST", 500 if quick else 1500)

    # --- Genetic ---
    from control.IK.genetic import GeneticIK

    robot.set_inverse(
        GeneticIK,
        bounds=bounds,
        obstacles=obstacles,
        population_size=gen_pop,
        generations=gen_gen,
        save_generation_images=False,
    )

    p_list, r_list, o_list, t_list = [], [], [], []
    pr_list, rr_list = [], []
    for tgt in targets:
        t0 = time.perf_counter()
        ang, m = robot.solve(tgt)
        t_list.append(time.perf_counter() - t0)
        p_list.append(float(m["position_error"]))
        r_list.append(float(m["orientation_error"]))
        o_list.append(obstacle_exp_penalty_sum(robot, ang, obstacles))
        _, m_ref = robot.op_solve(ang, tgt, obstacles=obstacles)
        pr_list.append(float(m_ref["position_error"]))
        rr_list.append(float(m_ref["orientation_error"]))

    rows.append(BenchmarkRow(
        algorithm="GeneticIK",
        mse="—",
        p_error_mean=float(np.mean(p_list)),
        r_error_mean=float(np.mean(r_list)),
        obstacle_error_mean=float(np.mean(o_list)),
        time_to_learn="—",
        time_to_execute_mean=float(np.mean(t_list)),
        p_error_refined_mean=float(np.mean(pr_list)),
        r_error_refined_mean=float(np.mean(rr_list)),
    ))

    # --- DDPG: обучение на первой цели, инференс на всех ---
    if not torch_ok:
        rows.append(BenchmarkRow(
            algorithm="DDPGIK",
            mse="—",
            p_error_mean=float("nan"),
            r_error_mean=float("nan"),
            obstacle_error_mean=float("nan"),
            time_to_learn="torch missing",
            time_to_execute_mean=float("nan"),
            p_error_refined_mean=float("nan"),
            r_error_refined_mean=float("nan"),
        ))
    else:
        from control.IK.ddpg import DDPGIK

        ckpt = str(PROJECT_ROOT / "checkpoints" / "benchmark_ddpg.pt")
        robot.set_inverse(
            DDPGIK,
            bounds=bounds,
            obstacles=obstacles,
            episodes=ddpg_ep,
            max_steps=ddpg_steps,
            actor_lr=1e-4,
            critic_lr=5e-4,
            tau=0.005,
            action_scale=0.06,
            noise_sigma=0.15,
            exp_weight_alpha=4.0,
            save_episode_images=False,
            save_training_plot=False,
            weights_path=ckpt,
            save_weights_after_run=True,
            load_weights_if_exist=False,
        )
        _, m_train = robot.solve(targets[0])
        learn_time = float(m_train["total_time"])

        p2, r2, o2, t2 = [], [], [], []
        pr2, rr2 = [], []
        for tgt in targets:
            t0 = time.perf_counter()
            ang, m = robot.solve(
                tgt,
                episodes=0,
                load_weights_if_exist=True,
                save_training_plot=False,
            )
            t2.append(time.perf_counter() - t0)
            p2.append(position_error_fk(robot, ang, tgt))
            r2.append(orientation_error_fk(robot, ang, tgt))
            o2.append(obstacle_exp_penalty_sum(robot, ang, obstacles))
            _, m_ref = robot.op_solve(ang, tgt, obstacles=obstacles)
            pr2.append(float(m_ref["position_error"]))
            rr2.append(float(m_ref["orientation_error"]))

        rows.append(BenchmarkRow(
            algorithm="DDPGIK",
            mse="—",
            p_error_mean=float(np.mean(p2)),
            r_error_mean=float(np.mean(r2)),
            obstacle_error_mean=float(np.mean(o2)),
            time_to_learn=_fmt_time(learn_time),
            time_to_execute_mean=float(np.mean(t2)),
            p_error_refined_mean=float(np.mean(pr2)),
            r_error_refined_mean=float(np.mean(rr2)),
        ))

    # --- ML: RandomForest ---
    from control.IK.decision_trees import RandomForestIK
    from control.IK.ml_dataset import build_xy_from_robot, sample_random_joint_configs

    rng = np.random.default_rng(42)
    train_angles = sample_random_joint_configs(bounds, ml_n, rng=rng)
    test_angles = sample_random_joint_configs(bounds, ml_test, rng=np.random.default_rng(43))
    X_tr, y_tr = build_xy_from_robot(robot, train_angles, dtype_x=np.float64, dtype_y=np.float64)
    X_te, y_te = build_xy_from_robot(robot, test_angles, dtype_x=np.float64, dtype_y=np.float64)

    robot.set_inverse(RandomForestIK, angle_limits=bounds, n_estimators=60 if quick else 120,
                      max_depth=12, dataset_size=ml_n)
    t0 = time.perf_counter()
    robot.ik_solver.X, robot.ik_solver.y = X_tr, y_tr
    robot.ik_solver.trained = False
    robot.ik_solver.train(plot_learning_curve=False)
    learn_rf = time.perf_counter() - t0
    pred_te = robot.ik_solver.model.predict(X_te)
    mse_rf = float(mean_squared_error(y_te, pred_te))

    p3, r3, o3, t3 = [], [], [], []
    pr3, rr3 = [], []
    for tgt in targets:
        t0 = time.perf_counter()
        ang, m = robot.ik_solver.solve(tgt)
        t3.append(time.perf_counter() - t0)
        p3.append(float(m["position_error"]))
        r3.append(float(m["orientation_error"]))
        o3.append(obstacle_exp_penalty_sum(robot, ang, obstacles))
        _, m_ref = robot.op_solve(ang, tgt, obstacles=obstacles)
        pr3.append(float(m_ref["position_error"]))
        rr3.append(float(m_ref["orientation_error"]))

    rows.append(BenchmarkRow(
        algorithm="RandomForestIK",
        mse=_fmt(mse_rf, 6),
        p_error_mean=float(np.mean(p3)),
        r_error_mean=float(np.mean(r3)),
        obstacle_error_mean=float(np.mean(o3)),
        time_to_learn=_fmt_time(learn_rf),
        time_to_execute_mean=float(np.mean(t3)),
        p_error_refined_mean=float(np.mean(pr3)),
        r_error_refined_mean=float(np.mean(rr3)),
    ))

    # --- XGBoost ---
    try:
        import xgboost  # noqa: F401
    except ImportError:
        rows.append(BenchmarkRow(
            algorithm="XGBoostIK",
            mse="xgboost missing",
            p_error_mean=float("nan"),
            r_error_mean=float("nan"),
            obstacle_error_mean=float("nan"),
            time_to_learn="—",
            time_to_execute_mean=float("nan"),
            p_error_refined_mean=float("nan"),
            r_error_refined_mean=float("nan"),
        ))
    else:
        from control.IK.decision_trees import XGBoostIK

        robot.set_inverse(
            XGBoostIK,
            angle_limits=bounds,
            n_estimators=80 if quick else 200,
            max_depth=5,
            learning_rate=0.1,
            dataset_size=ml_n,
            model_path=None,
        )
        t0 = time.perf_counter()
        robot.ik_solver.X, robot.ik_solver.y = X_tr, y_tr
        robot.ik_solver.trained = False
        robot.ik_solver.train(plot_learning_curve=False)
        learn_xgb = time.perf_counter() - t0
        pred_te = robot.ik_solver.model.predict(X_te)
        mse_xgb = float(mean_squared_error(y_te, pred_te))

        p4, r4, o4, t4 = [], [], [], []
        pr4, rr4 = [], []
        for tgt in targets:
            t0 = time.perf_counter()
            ang, m = robot.ik_solver.solve(tgt)
            t4.append(time.perf_counter() - t0)
            p4.append(float(m["position_error"]))
            r4.append(float(m["orientation_error"]))
            o4.append(obstacle_exp_penalty_sum(robot, ang, obstacles))
            _, m_ref = robot.op_solve(ang, tgt, obstacles=obstacles)
            pr4.append(float(m_ref["position_error"]))
            rr4.append(float(m_ref["orientation_error"]))

        rows.append(BenchmarkRow(
            algorithm="XGBoostIK",
            mse=_fmt(mse_xgb, 6),
            p_error_mean=float(np.mean(p4)),
            r_error_mean=float(np.mean(r4)),
            obstacle_error_mean=float(np.mean(o4)),
            time_to_learn=_fmt_time(learn_xgb),
            time_to_execute_mean=float(np.mean(t4)),
            p_error_refined_mean=float(np.mean(pr4)),
            r_error_refined_mean=float(np.mean(rr4)),
        ))

    # --- Neural ---
    if not torch_ok:
        rows.append(BenchmarkRow(
            algorithm="ForwardNeuralIK",
            mse="torch missing",
            p_error_mean=float("nan"),
            r_error_mean=float("nan"),
            obstacle_error_mean=float("nan"),
            time_to_learn="torch missing",
            time_to_execute_mean=float("nan"),
            p_error_refined_mean=float("nan"),
            r_error_refined_mean=float("nan"),
        ))
    else:
        import torch
        from control.IK.nn import ForwardNeuralIK

        n_j = len(bounds)
        robot.set_inverse(
            ForwardNeuralIK,
            layers=[12, 64, 64, n_j],
            angle_limits=bounds,
            epochs=8 if quick else 25,
            batch_size=128,
            lr=1e-3,
            dataset_size=ml_n,
            model_path=None,
        )
        t0 = time.perf_counter()
        robot.ik_solver.X, robot.ik_solver.y = X_tr.astype(np.float32), y_tr.astype(np.float32)
        robot.ik_solver.trained = False
        robot.ik_solver.train()
        learn_nn = time.perf_counter() - t0

        device = next(robot.ik_solver.model.parameters()).device
        with torch.no_grad():
            pred_te_t = robot.ik_solver.model(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()
        mse_nn = float(mean_squared_error(y_te, pred_te_t))

        p5, r5, o5, t5 = [], [], [], []
        pr5, rr5 = [], []
        for tgt in targets:
            t0 = time.perf_counter()
            ang, m = robot.ik_solver.solve(tgt)
            t5.append(time.perf_counter() - t0)
            p5.append(float(m["position_error"]))
            r5.append(float(m["orientation_error"]))
            o5.append(obstacle_exp_penalty_sum(robot, ang, obstacles))
            _, m_ref = robot.op_solve(ang, tgt, obstacles=obstacles)
            pr5.append(float(m_ref["position_error"]))
            rr5.append(float(m_ref["orientation_error"]))

        rows.append(BenchmarkRow(
            algorithm="ForwardNeuralIK",
            mse=_fmt(mse_nn, 6),
            p_error_mean=float(np.mean(p5)),
            r_error_mean=float(np.mean(r5)),
            obstacle_error_mean=float(np.mean(o5)),
            time_to_learn=_fmt_time(learn_nn),
            time_to_execute_mean=float(np.mean(t5)),
            p_error_refined_mean=float(np.mean(pr5)),
            r_error_refined_mean=float(np.mean(rr5)),
        ))

    _print_table(rows)
    ml_fit_rows = int(ml_n * 0.8)
    print(
        "\nПояснения: MSE — только для ML, MSE по углам суставов на отложенной случайной выборке (не по тем же "
        "5 фиксированным целям).\n"
        f"ML обучение: сгенерировано {ml_n} случайных конфигураций суставов (X,y); в train() внутренний "
        f"train_test_split(test_size=0.2) → порядка {ml_fit_rows} строк на фактическое обучение модели.\n"
        f"Столбец MSE: отдельная выборка из {ml_test} конфигураций (BENCHMARK_ML_SAMPLES / BENCHMARK_ML_TEST).\n"
        "P_error / R_error / Obstacle_error_mean — среднее по одному и тому же набору фиксированных целей FK.\n"
        "Obstacle_error_mean — сумма exp(-d/sigma) по отрезкам звеньев и всем сферам (sigma=0.01).\n"
        "TimeToLearn: Genetic — «—»; DDPG — total_time первого solve (обучение); ML — время train() после "
        "подготовки X,y.\n"
        "TimeToExecute(mean): Genetic — среднее время solve на цель; DDPG — solve(episodes=0) с загрузкой весов; "
        "ML — среднее время solve (инференс).\n"
        "P_error_refined_mean / R_error_refined_mean — средние ошибки после уточнения "
        "``Robot.op_solve(..., obstacles=...)`` (Levenberg–Marquardt), в экспоненциальной записи (например 1.2e-07).\n"
        "Строка «torch missing» — в том же интерпретаторе, что запускает pytest, нет пакета torch; "
        "установите: ``pip install torch`` (или ``pip install -r requirements.txt`` из корня репозитория).\n"
    )

    for r in rows:
        if r.algorithm == "GeneticIK":
            assert r.time_to_learn == "—"
        if r.p_error_mean == r.p_error_mean:  # not nan
            if not np.isnan(r.p_error_mean):
                assert np.isfinite(r.p_error_mean)
