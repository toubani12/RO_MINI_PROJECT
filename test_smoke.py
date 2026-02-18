

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASSED = []
FAILED = []


def _test(name: str):
    def decorator(fn):
        try:
            fn()
            PASSED.append(name)
            print(f"  ✓  {name}")
        except Exception as exc:
            FAILED.append((name, exc))
            print(f"  ✗  {name}")
            traceback.print_exc()
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@_test("Import src.tsp_core")
def _():
    from src.tsp_core import TSPInstance  # noqa: F401


@_test("TSPInstance — création (N=10, seed=0)")
def _():
    from src.tsp_core import TSPInstance
    inst = TSPInstance(10, seed=0)
    assert inst.num_cities == 10
    assert inst.coords.shape == (10, 2)
    assert inst.dist_matrix.shape == (10, 10)


@_test("TSPInstance — evaluate() retourne float positif")
def _():
    from src.tsp_core import TSPInstance
    import numpy as np
    inst = TSPInstance(10, seed=1)
    tour = np.arange(10)
    cost = inst.evaluate(tour)
    assert isinstance(cost, float)
    assert cost > 0


@_test("TSPInstance — delta_2opt() cohérence avec evaluate()")
def _():
    from src.tsp_core import TSPInstance
    import numpy as np
    inst = TSPInstance(15, seed=7)
    tour = np.random.permutation(15)
    i, j = 2, 8
    cost_before = inst.evaluate(tour)
    new_tour = inst.apply_2opt_move(tour, i, j)
    cost_after = inst.evaluate(new_tour)
    delta = inst.delta_2opt(tour, i, j)
    assert abs((cost_after - cost_before) - delta) < 1e-6, (
        f"Δf={delta:.6f}  diff réelle={cost_after - cost_before:.6f}"
    )


@_test("Import src.algorithms")
def _():
    from src.algorithms import HillClimbing, MultiStartHillClimbing, SimulatedAnnealing  # noqa


@_test("HillClimbing — solve() retourne SolverResult valide")
def _():
    from src.tsp_core import TSPInstance
    from src.algorithms import HillClimbing, SolverResult
    inst = TSPInstance(20, seed=42)
    hc = HillClimbing(inst)
    result = hc.solve(max_no_improv=10)
    assert isinstance(result, SolverResult)
    assert result.cost > 0
    assert len(result.tour) == 20
    print(f"       HC_First cost={result.cost:.2f}", end="")


@_test("MultiStartHillClimbing — solve() retourne SolverResult valide")
def _():
    from src.tsp_core import TSPInstance
    from src.algorithms import MultiStartHillClimbing, SolverResult
    inst = TSPInstance(20, seed=42)
    mshc = MultiStartHillClimbing(inst)
    result = mshc.solve(restarts=3, max_no_improv=10)
    assert isinstance(result, SolverResult)
    assert result.cost > 0
    print(f"       MSHC cost={result.cost:.2f}", end="")


@_test("SimulatedAnnealing — solve() retourne SolverResult valide")
def _():
    from src.tsp_core import TSPInstance
    from src.algorithms import SimulatedAnnealing, SolverResult
    inst = TSPInstance(20, seed=42)
    sa = SimulatedAnnealing(inst)
    result = sa.solve(T0=1000, max_iter=500)
    assert isinstance(result, SolverResult)
    assert result.cost > 0
    print(f"       SA cost={result.cost:.2f}", end="")


@_test("Import data.generator")
def _():
    from data.generator import generate_instance, generate_batch  # noqa


@_test("generate_batch() — tailles correctes")
def _():
    from data.generator import generate_batch
    instances = generate_batch([10, 20, 30])
    assert set(instances.keys()) == {10, 20, 30}
    assert all(inst.num_cities == n for n, inst in instances.items())


@_test("Import src.utils")
def _():
    from src.utils import plot_solution, plot_comparison_bar, export_results, summarize_results  # noqa


@_test("summarize_results() — structure du DataFrame")
def _():
    from src.utils import summarize_results
    raw = [
        {"Size": 20, "Algorithm": "HC_First", "cost": 450.0, "time": 0.05},
        {"Size": 20, "Algorithm": "HC_First", "cost": 460.0, "time": 0.06},
        {"Size": 20, "Algorithm": "HC_MultiStart", "cost": 420.0, "time": 2.1},
        {"Size": 20, "Algorithm": "HC_MultiStart", "cost": 415.0, "time": 2.2},
    ]
    df = summarize_results(raw)
    assert "Best Cost" in df.columns
    assert "Std Dev" in df.columns
    assert len(df) == 2


# ---------------------------------------------------------------------------
# Rapport final
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Smoke Tests — TSP Metaheuristics Project")
    print("=" * 50 + "\n")

    total = len(PASSED) + len(FAILED)
    print(f"\n{'='*50}")
    print(f"  Résultat : {len(PASSED)}/{total} tests passés")
    if FAILED:
        print(f"  Échecs   : {[name for name, _ in FAILED]}")
    print("=" * 50)
    sys.exit(0 if not FAILED else 1)
