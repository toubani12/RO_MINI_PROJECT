"""
main.py
-------
Script principal d'orchestration des expériences TSP.

Protocole expérimental :
  - Instances : N ∈ {20, 50, 80} villes générées dans [0, 100]²
  - Algorithmes : HC-FI, MS-HC, Recuit Simulé
  - Runs : N_RUNS = 30 exécutions indépendantes par (algorithme × instance)
  - Métriques : best cost, avg cost, std dev, avg CPU time

Artefacts produits :
  - assets/tsp_results.csv      — tableau de résultats agrégés
  - images/best_tour_<algo>_<N>.png — visualisation des meilleures tournées
  - images/results_bar_<N>.png  — diagramme de comparaison par taille

"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ajout du répertoire racine du projet au PYTHONPATH
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from data import generate_batch
from src import (
    HillClimbing,
    MultiStartHillClimbing,
    SimulatedAnnealing,
    export_results,
    plot_comparison_bar,
    plot_solution,
)
from src.utils import summarize_results

# ---------------------------------------------------------------------------
# Configuration de l'expérimentation
# ---------------------------------------------------------------------------
INSTANCE_SIZES: list[int] = [20, 50, 80]
N_RUNS: int = 30
BASE_SEED: int = 42

# Paramètres des solveurs
HC_PARAMS: dict = {"max_no_improv": 100}
MS_HC_PARAMS: dict = {"restarts": 10, "max_no_improv": 100}


def sa_params(n: int) -> dict:
    """Adapte les paramètres SA à la taille de l'instance."""
    return {"T0": 1_000.0, "alpha": 0.995, "T_min": 1e-3, "max_iter": n * 200}


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------

def run_experiments() -> list[dict]:
    """
    Exécute N_RUNS runs pour chaque combinaison (algorithme × instance).

    Retourne
    --------
    list[dict]
        Enregistrements bruts avec clés : Size, Algorithm, cost, time.
    """
    instances = generate_batch(INSTANCE_SIZES, base_seed=BASE_SEED)
    raw_records: list[dict] = []

    for n, instance in instances.items():
        print(f"\n{'='*55}")
        print(f"  Instance N = {n} | seed = {BASE_SEED + INSTANCE_SIZES.index(n)}")
        print(f"{'='*55}")

        solvers = {
            "HC_First": HillClimbing(instance),
            "HC_MultiStart": MultiStartHillClimbing(instance),
            "SimulatedAnnealing": SimulatedAnnealing(instance),
        }

        best_results: dict[str, object] = {}

        for algo_name, solver in solvers.items():
            costs: list[float] = []
            times: list[float] = []
            best_result = None

            for run in tqdm(range(N_RUNS), desc=f"  {algo_name:<20} (N={n})"):
                t0 = time.perf_counter()

                if algo_name == "HC_First":
                    result = solver.solve(**HC_PARAMS)
                elif algo_name == "HC_MultiStart":
                    result = solver.solve(**MS_HC_PARAMS)
                else:
                    result = solver.solve(**sa_params(n))

                elapsed = time.perf_counter() - t0

                costs.append(result.cost)
                times.append(elapsed)
                raw_records.append({
                    "Size": n,
                    "Algorithm": algo_name,
                    "cost": result.cost,
                    "time": elapsed,
                })

                if best_result is None or result.cost < best_result.cost:
                    best_result = result

            best_results[algo_name] = best_result

            print(
                f"    → best={min(costs):.2f}  avg={np.mean(costs):.2f}  "
                f"std={np.std(costs):.2f}  t̄={np.mean(times):.3f}s"
            )

        # Génération des figures de visualisation pour cette taille N
        _save_figures(instance, best_results, n)

    return raw_records


def _save_figures(instance, best_results: dict, n: int) -> None:
    """Sauvegarde les figures de tournée et de comparaison pour l'instance N."""
    for algo_name, result in best_results.items():
        if result is None:
            continue
        safe_name = algo_name.lower().replace(" ", "_")
        plot_solution(
            coords=instance.coords,
            tour=result.tour,
            cost=result.cost,
            title=f"Meilleure tournée — {algo_name} (N={n})",
            save_as=f"best_tour_{safe_name}_{n}.png",
            show=False,
        )


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("   Étude comparative de métaheuristiques — TSP")
    print("   ENSET Mohammedia | Master SDIA 2025-2026")
    print("=" * 55)

    raw = run_experiments()

    # Agrégation et affichage
    df_summary = summarize_results(raw)
    print("\n\n=== RÉSULTATS AGRÉGÉS (30 runs) ===")
    print(df_summary.to_string(index=False))

    # Export CSV
    out_path = export_results(df_summary)
    print(f"\n✓ Résultats exportés → {out_path}")

    # Diagrammes de comparaison par taille
    for n in INSTANCE_SIZES:
        plot_comparison_bar(
            df=df_summary,
            size=n,
            metric="Avg Cost",
            error_col="Std Dev",
            save_as=f"results_bar_{n}.png",
            show=False,
        )
    print(f"✓ Figures sauvegardées → images/")
