"""
utils.py
--------
Fonctions utilitaires pour le projet TSP :

  - plot_solution()         : visualisation d'un tour sur la carte des villes.
  - plot_comparison_bar()   : diagramme en barres comparant les coûts moyens
                              avec barres d'erreur (±σ).
  - export_results()        : export du DataFrame résultats vers CSV.
  - summarize_results()     : calcul des statistiques agrégées (best, mean, std, time).

Toutes les figures sont sauvegardées dans le dossier ``images/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Dossier de sortie des figures (créé automatiquement si absent)
_IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_ASSETS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Visualisation : tour optimal
# ---------------------------------------------------------------------------

def plot_solution(
    coords: np.ndarray,
    tour: np.ndarray,
    cost: float,
    title: str = "Meilleure tournée TSP",
    save_as: str | None = None,
    show: bool = True,
) -> None:
    """
    Affiche la tournée TSP sur la carte euclidienne des villes.

    Paramètres
    ----------
    coords : np.ndarray, shape (N, 2)
        Coordonnées des villes.
    tour : np.ndarray, shape (N,)
        Permutation des indices de villes représentant la tournée.
    cost : float
        Coût (longueur) du tour à afficher dans le titre.
    title : str
        Titre de la figure.
    save_as : str, optionnel
        Nom de fichier (sans chemin) pour sauvegarder la figure dans ``images/``.
        Ex : ``"best_tour_sa_80.png"``.
    show : bool
        Si True, appelle plt.show().
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # Arêtes du tour (fermeture du cycle)
    closed_tour = np.append(tour, tour[0])
    tour_coords = coords[closed_tour]
    ax.plot(tour_coords[:, 0], tour_coords[:, 1],
            color="#E74C3C", linewidth=1.2, zorder=1)

    # Nœuds (villes)
    ax.scatter(coords[:, 0], coords[:, 1],
               s=50, color="#2C3E50", zorder=2)

    # Ville de départ mise en évidence
    ax.scatter(coords[tour[0], 0], coords[tour[0], 1],
               s=120, color="#F39C12", zorder=3, label="Départ")

    ax.set_title(f"{title}\nCoût = {cost:.2f}", fontsize=13, fontweight="bold")
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save_as:
        fig.savefig(_IMAGES_DIR / save_as, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Visualisation : comparaison des algorithmes
# ---------------------------------------------------------------------------

def plot_comparison_bar(
    df: pd.DataFrame,
    size: int,
    metric: str = "Avg Cost",
    error_col: str = "Std Dev",
    save_as: str | None = None,
    show: bool = True,
) -> None:
    """
    Diagramme en barres comparant les algorithmes pour une taille N donnée.

    Paramètres
    ----------
    df : pd.DataFrame
        Tableau de résultats avec colonnes : Size, Algorithm, Avg Cost, Std Dev, …
    size : int
        Taille d'instance à afficher (filtre sur la colonne Size).
    metric : str
        Colonne à afficher (hauteur des barres).
    error_col : str
        Colonne à utiliser comme barres d'erreur (±σ).
    save_as : str, optionnel
        Nom de fichier pour sauvegarder dans ``images/``.
    show : bool
        Si True, appelle plt.show().
    """
    subset = df[df["Size"] == size].copy()
    algorithms = subset["Algorithm"].tolist()
    values = subset[metric].tolist()
    errors = subset[error_col].tolist() if error_col in subset.columns else None

    colors = ["#3498DB", "#2ECC71", "#E74C3C"][: len(algorithms)]
    x = np.arange(len(algorithms))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, values, color=colors, width=0.5,
                  yerr=errors, capsize=6, error_kw={"linewidth": 1.5},
                  zorder=2)

    # Annotations des valeurs au-dessus des barres
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(values) * 0.01),
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=10)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(
        f"Comparaison des algorithmes — N = {size}\n"
        f"(30 runs indépendants, barres d'erreur = ±σ)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_ylim(0, max(v + (e or 0) for v, e in zip(values, errors or [0] * len(values))) * 1.15)
    fig.tight_layout()

    if save_as:
        fig.savefig(_IMAGES_DIR / save_as, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Statistiques agrégées
# ---------------------------------------------------------------------------

def summarize_results(raw_records: list[dict]) -> pd.DataFrame:
    """
    Calcule les statistiques agrégées (best, mean, std, avg_time) à partir
    d'une liste de dicts bruts produits par l'expérimentation.

    Chaque dict doit contenir : Size, Algorithm, cost, time.

    Retourne
    --------
    pd.DataFrame avec colonnes :
        Size, Algorithm, Best Cost, Avg Cost, Std Dev, Avg Time (s)
    """
    df_raw = pd.DataFrame(raw_records)
    agg = (
        df_raw.groupby(["Size", "Algorithm"], sort=False)
        .agg(
            **{
                "Best Cost": ("cost", "min"),
                "Avg Cost": ("cost", "mean"),
                "Std Dev": ("cost", "std"),
                "Avg Time (s)": ("time", "mean"),
            }
        )
        .reset_index()
    )
    # Arrondi à 2 décimales
    numeric_cols = ["Best Cost", "Avg Cost", "Std Dev", "Avg Time (s)"]
    agg[numeric_cols] = agg[numeric_cols].round(2)
    return agg


# ---------------------------------------------------------------------------
# Export CSV
# ---------------------------------------------------------------------------

def export_results(df: pd.DataFrame, filename: str = "tsp_results.csv") -> Path:
    """
    Exporte le DataFrame des résultats vers ``assets/<filename>``.

    Paramètres
    ----------
    df : pd.DataFrame
    filename : str

    Retourne
    --------
    Path : chemin absolu du fichier créé.
    """
    out_path = _ASSETS_DIR / filename
    df.to_csv(out_path, index=False)
    return out_path
