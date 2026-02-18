"""
data/generator.py
-----------------
Fonctions de génération d'instances TSP euclidiennes.

Les instances sont générées aléatoirement dans l'espace [0, 100]² selon
une distribution uniforme. Chaque instance est reproductible via une graine
(seed) NumPy.

Fonctions
---------
generate_instance(num_cities, seed) -> TSPInstance
    Crée et retourne une instance unique.

generate_batch(sizes, base_seed) -> dict[int, TSPInstance]
    Crée un dictionnaire {taille: instance} pour une liste de tailles.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Résolution du chemin pour import relatif depuis data/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tsp_core import TSPInstance


def generate_instance(num_cities: int, seed: int | None = None) -> TSPInstance:
    """
    Génère une instance TSP euclidienne à num_cities villes dans [0, 100]².

    Paramètres
    ----------
    num_cities : int
        Nombre de villes N.
    seed : int, optionnel
        Graine pour la reproductibilité.

    Retourne
    --------
    TSPInstance
    """
    if num_cities < 2:
        raise ValueError(f"num_cities doit être ≥ 2, reçu : {num_cities}")
    return TSPInstance(num_cities=num_cities, seed=seed)


def generate_batch(
    sizes: list[int],
    base_seed: int = 42,
) -> dict[int, TSPInstance]:
    """
    Génère un ensemble d'instances de tailles variées.

    Chaque instance reçoit une graine dérivée de base_seed pour garantir
    la reproductibilité : seed_i = base_seed + i.

    Paramètres
    ----------
    sizes : list[int]
        Liste des tailles N (ex : [20, 50, 80]).
    base_seed : int
        Graine de base (chaque instance utilise base_seed + index).

    Retourne
    --------
    dict[int, TSPInstance]
        Dictionnaire {N: instance}.

    Exemple
    -------
    >>> instances = generate_batch([20, 50, 80])
    >>> instances[50].num_cities
    50
    """
    return {
        n: generate_instance(n, seed=base_seed + i)
        for i, n in enumerate(sizes)
    }
