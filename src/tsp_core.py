"""
tsp_core.py
-----------
Définition vectorisée du problème TSP.

Fournit la classe TSPInstance qui encapsule :
  - la génération aléatoire des coordonnées de villes,
  - le pré-calcul de la matrice de distances N×N (distance euclidienne),
  - l'évaluation vectorisée du coût d'un tour via l'indexation avancée NumPy.

Référence vectorisation :
  Harris et al. (2020). Array programming with NumPy. Nature, 585, 357-362.
"""

import numpy as np


class TSPInstance:
    """
    Représente une instance du problème du Voyageur de Commerce (TSP).

    Attributs
    ---------
    num_cities : int
        Nombre de villes N.
    coords : np.ndarray, shape (N, 2)
        Coordonnées euclidiennes des villes, tirées dans [0, 100]².
    dist_matrix : np.ndarray, shape (N, N)
        Matrice des distances euclidiennes pré-calculée.
        dist_matrix[i, j] = ||coords[i] - coords[j]||_2.
    seed : int or None
        Graine utilisée pour la reproductibilité.
    """

    def __init__(self, num_cities: int, seed: int | None = None) -> None:
        """
        Paramètres
        ----------
        num_cities : int
            Nombre de villes à générer.
        seed : int, optionnel
            Graine NumPy pour la reproductibilité des instances.
        """
        if seed is not None:
            np.random.seed(seed)

        self.num_cities = num_cities
        self.seed = seed

        # Génération aléatoire uniforme dans [0, 100]²
        self.coords: np.ndarray = np.random.rand(num_cities, 2) * 100

        # Pré-calcul unique de la matrice de distances O(N²)
        self.dist_matrix: np.ndarray = self._build_distance_matrix()

    # ------------------------------------------------------------------
    # Méthodes privées
    # ------------------------------------------------------------------

    def _build_distance_matrix(self) -> np.ndarray:
        """
        Calcule la matrice des distances N×N en une seule passe vectorisée.

        Utilise le broadcasting NumPy :
            diff[i, j] = coords[i] - coords[j]   ∈ ℝ²
        Complexité : O(N²) en temps et en espace.

        Retourne
        --------
        np.ndarray, shape (N, N), dtype float64
        """
        # diff : (N, 1, 2) - (1, N, 2) → (N, N, 2)
        diff = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    # ------------------------------------------------------------------
    # Méthodes publiques
    # ------------------------------------------------------------------

    def evaluate(self, tour: np.ndarray | list) -> float:
        """
        Calcule la longueur totale d'un tour par indexation avancée NumPy.

        f(π) = Σ_{i=0}^{N-1} dist_matrix[π_i, π_{(i+1) mod N}]

        Paramètres
        ----------
        tour : array-like of int, longueur N
            Permutation des indices de villes.

        Retourne
        --------
        float
            Longueur totale du tour (somme des arêtes).
        """
        tour = np.asarray(tour, dtype=int)
        shifted = np.roll(tour, -1)
        return float(np.sum(self.dist_matrix[tour, shifted]))

    def random_tour(self) -> np.ndarray:
        """Génère et retourne une permutation aléatoire uniformes des N villes."""
        return np.random.permutation(self.num_cities)

    def apply_2opt_move(self, tour: np.ndarray, i: int, j: int) -> np.ndarray:
        """
        Applique un mouvement 2-opt sur le tour en inversant le segment [i+1, j].

        Le coût de la variation est :
            Δf = d(π_i, π_j) + d(π_{i+1}, π_{j+1})
               - d(π_i, π_{i+1}) - d(π_j, π_{j+1})

        Paramètres
        ----------
        tour : np.ndarray
            Tour courant.
        i, j : int
            Indices tels que i < j.

        Retourne
        --------
        np.ndarray
            Nouveau tour avec le segment inversé.
        """
        new_tour = tour.copy()
        new_tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
        return new_tour

    def delta_2opt(self, tour: np.ndarray, i: int, j: int) -> float:
        """
        Calcule la variation de coût Δf d'un mouvement 2-opt sans recalculer
        le tour complet — complexité O(1).

        Paramètres
        ----------
        tour : np.ndarray
        i, j : int  (0 ≤ i < j ≤ N-1)

        Retourne
        --------
        float : Δf = coût(nouveau) - coût(courant).
                Négatif si le mouvement est améliorant.
        """
        n = self.num_cities
        a, b = tour[i], tour[(i + 1) % n]
        c, d = tour[j], tour[(j + 1) % n]
        D = self.dist_matrix
        return float(D[a, c] + D[b, d] - D[a, b] - D[c, d])

    def __repr__(self) -> str:
        return f"TSPInstance(N={self.num_cities}, seed={self.seed})"
