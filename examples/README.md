# Примеры

Обзор проекта: [README в корне репозитория](../README.md).

Сводная таблица бенчмарка (Genetic / DDPG / ML): `BENCHMARK_QUICK=1 MPLBACKEND=Agg pytest tests/test_ik_benchmark_table.py -s`

Запускай из **корня репозитория** (`rl-for-robots/`):


| Скрипт                          | Описание                                                |
| ------------------------------- | ------------------------------------------------------- |
| `python examples/genetic_ik.py` | Genetic IK                                              |
| `python examples/ddpg_ik.py`    | DDPG IK (затюненные гиперы + `ddpg_training_curve.png`) |
| `python examples/ml_rf_ik.py`   | Random Forest IK, датасет по сетке суставов             |
| `python examples/ml_xgb_ik.py`  | XGBoost IK, сетка + сохранение модели                   |
| `python examples/ml_nn_ik.py`   | MLP IK, сетка + случайные сэмплы                        |


Общая кинематика и цель: `[common.py](common.py)`.

Для ML метод `generate_dataset_grid(angle_limits, joint_value_grid={индекс: [значения]}, ...)` строит декартово произведение по указанным суставам; остальные углы задаются `default_angles` или серединами `angle_limits`.