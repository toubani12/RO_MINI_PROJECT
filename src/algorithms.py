"""
algorithms.py
-------------
Implémentation des trois algorithmes métaheuristiques pour le TSP :

  1. HillClimbing       — Recherche locale Hill-Climbing (First Improvement)
  2. MultiStartHillClimbing — Multi-Start Hill-Climbing (diversification par
                              redémarrages aléatoires)
  3. SimulatedAnnealing — Recuit Simulé avec schéma de refroidissement géométrique

Tous les solveurs exploitent la classe TSPInstance (src/tsp_core.py) et les
mouvements 2-opt (inversion de segment) via delta_2opt() O(1).

Références
----------
[1] Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by
    Simulated Annealing. Science, 220(4598), 671–680.
[2] Lin, S. (1965). Computer solutions of the traveling salesman problem.
    Bell System Technical Journal, 44(10), 2245–2269.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .tsp_core import TSPInstance


# ---------------------------------------------------------------------------
# Dataclass résultat
# ---------------------------------------------------------------------------

@dataclass
class SolverResult:
    """Encapsule le résultat d'une exécution d'un solveur TSP."""
    tour: np.ndarray
    cost: float
    algorithm: str
    n_cities: int
    iterations: int = 0
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"SolverResult(algo={self.algorithm!r}, "
                f"N={self.n_cities}, cost={self.cost:.4f}, "
                f"iters={self.iterations})")


# ---------------------------------------------------------------------------
# Classe de base
# ---------------------------------------------------------------------------

class _BaseSolver:
    """Interface commune à tous les solveurs TSP."""

    name: str = "BaseSolver"

    def __init__(self, instance: TSPInstance) -> None:
        self.instance = instance
        self.n = instance.num_cities

    # ------------------------------------------------------------------
    # Utilitaires partagés
    # ------------------------------------------------------------------

    def _random_tour(self) -> np.ndarray:
        """Retourne une permutation aléatoire uniforme des N villes."""
        return self.instance.random_tour()

    def _evaluate(self, tour: np.ndarray) -> float:
        """Proxy vers TSPInstance.evaluate()."""
        return self.instance.evaluate(tour)

    def solve(self, **kwargs) -> SolverResult:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Hill-Climbing – Première Amélioration (HC-FI)
# ---------------------------------------------------------------------------

class HillClimbing(_BaseSolver):
    """
    Hill-Climbing avec stratégie de première amélioration (First Improvement).

    À chaque itération, l'algorithme parcourt les paires de villes en ordre
    aléatoire et accepte **immédiatement** le premier mouvement 2-opt dont
    le delta est négatif (Δf < 0), puis recommence.
    S'arrête quand aucun voisin améliorant n'est trouvé (optimum local 2-opt).

    Complexité par itération : O(N²) dans le pire cas.
    """

    name = "HC_First"

    def solve(
        self,
        max_no_improv: int = 100,
        initial_tour: np.ndarray | None = None,
    ) -> SolverResult:
        """
        Paramètres
        ----------
        max_no_improv : int
            Nombre maximal d'itérations consécutives sans amélioration.
        initial_tour : np.ndarray, optionnel
            Solution de départ imposée. Si None, tirée aléatoirement.

        Retourne
        --------
        SolverResult
        """
        tour = self._random_tour() if initial_tour is None else initial_tour.copy()
        cost = self._evaluate(tour)

        # Toutes les paires (i, j) avec i < j — voisinage 2-opt
        pairs = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        pairs_arr = list(pairs)

        no_improv = 0
        total_iters = 0

        while no_improv < max_no_improv:
            improved = False
            np.random.shuffle(pairs_arr)  # ordre aléatoire → first improvement

            for i, j in pairs_arr:
                delta = self.instance.delta_2opt(tour, i, j)
                if delta < -1e-10:          # mouvement améliorant
                    tour = self.instance.apply_2opt_move(tour, i, j)
                    cost += delta
                    improved = True
                    total_iters += 1
                    break                   # premier mouvement accepté

            no_improv = 0 if improved else no_improv + 1

        return SolverResult(
            tour=tour,
            cost=self._evaluate(tour),      # recalcul propre pour éviter dérive flottante
            algorithm=self.name,
            n_cities=self.n,
            iterations=total_iters,
        )


# ---------------------------------------------------------------------------
# 2. Multi-Start Hill-Climbing (MS-HC)
# ---------------------------------------------------------------------------

class MultiStartHillClimbing(_BaseSolver):
    """
    Multi-Start Hill-Climbing.

    Lance k exécutions indépendantes de HC-FI depuis des solutions initiales
    aléatoirement différentes, et retient la meilleure solution globale :

        f* = min_{r=1}^{k} f(HC-FI(π_r^(0))),   π_r^(0) ~ U(Permutations(N))

    Complexité totale : k × O(N² × I_max).
    """

    name = "HC_MultiStart"

    def __init__(self, instance: TSPInstance) -> None:
        super().__init__(instance)
        self._hc = HillClimbing(instance)

    def solve(
        self,
        restarts: int = 10,
        max_no_improv: int = 100,
    ) -> SolverResult:
        """
        Paramètres
        ----------
        restarts : int
            Nombre de redémarrages aléatoires (k).
        max_no_improv : int
            Transmis à chaque exécution HC-FI.

        Retourne
        --------
        SolverResult (meilleure solution parmi les k runs)
        """
        best: SolverResult | None = None
        total_iters = 0

        for _ in range(restarts):
            result = self._hc.solve(max_no_improv=max_no_improv)
            total_iters += result.iterations
            if best is None or result.cost < best.cost:
                best = result

        assert best is not None
        return SolverResult(
            tour=best.tour,
            cost=best.cost,
            algorithm=self.name,
            n_cities=self.n,
            iterations=total_iters,
            extra={"restarts": restarts},
        )


# ---------------------------------------------------------------------------
# 3. Recuit Simulé (Simulated Annealing – SA)
# ---------------------------------------------------------------------------

class SimulatedAnnealing(_BaseSolver):
    """
    Recuit Simulé avec schéma de refroidissement géométrique.

    Acceptation selon la règle de Metropolis–Hastings :

        P(accepter π') = 1              si Δf ≤ 0
                       = exp(-Δf / T)   sinon

    Schéma de température :
        T_{k+1} = α · T_k,   T_min ≤ T_k ≤ T_0

    Référence : Kirkpatrick et al. (1983), Science 220(4598), 671–680.
    """

    name = "SimulatedAnnealing"

    def solve(
        self,
        T0: float = 1_000.0,
        alpha: float = 0.995,
        T_min: float = 1e-3,
        max_iter: int = 10_000,
        initial_tour: np.ndarray | None = None,
    ) -> SolverResult:
        """
        Paramètres
        ----------
        T0 : float
            Température initiale (contrôle l'exploration initiale).
        alpha : float ∈ (0, 1)
            Facteur de refroidissement géométrique.
        T_min : float
            Température minimale (critère d'arrêt thermique).
        max_iter : int
            Nombre maximal de mouvements candidats.
        initial_tour : np.ndarray, optionnel
            Solution initiale imposée.

        Retourne
        --------
        SolverResult
        """
        tour = self._random_tour() if initial_tour is None else initial_tour.copy()
        cost = self._evaluate(tour)

        best_tour = tour.copy()
        best_cost = cost

        T = T0
        accepted = 0

        for k in range(max_iter):
            # Génération d'un voisin par mouvement 2-opt aléatoire
            i, j = sorted(np.random.choice(self.n, 2, replace=False))
            delta = self.instance.delta_2opt(tour, i, j)

            # Règle de Metropolis–Hastings
            if delta < 0 or np.random.rand() < math.exp(-delta / T):
                tour = self.instance.apply_2opt_move(tour, i, j)
                cost += delta
                accepted += 1

                if cost < best_cost:
                    best_cost = cost
                    best_tour = tour.copy()

            # Refroidissement géométrique
            T *= alpha
            if T < T_min:
                break

        return SolverResult(
            tour=best_tour,
            cost=self._evaluate(best_tour),  # recalcul propre
            algorithm=self.name,
            n_cities=self.n,
            iterations=k + 1,
            extra={
                "T0": T0,
                "alpha": alpha,
                "T_final": T,
                "accepted_moves": accepted,
            },
        )
