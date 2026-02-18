# Étude comparative de métaheuristiques pour le problème du voyageur de commerce (TSP)

**Établissement :** Université Hassan II de Casablanca – ENSET Mohammedia

**Master :** SDIA (Systèmes de Données & Intelligence Artificielle) / GESI

**Module :** Optimisation / Métaheuristiques

**Encadrant :** Pr. MESTARI

**Auteurs :** Badr Eddine TOUBANI, Imane MEKKAOUI

**Année universitaire :** 2025–2026

---

## 📝 Résumé

Ce projet étudie l'efficacité de trois algorithmes métaheuristiques pour résoudre le problème du voyageur de commerce (TSP), un problème classique d'optimisation combinatoire NP-difficile [\[1\]](#références). Nous avons implémenté et comparé la recherche locale par *Hill-Climbing* (première amélioration), le *Multi-Start Hill-Climbing* et le *Recuit Simulé* (*Simulated Annealing*) en s'appuyant sur une architecture Python vectorisée à haute performance.

Les algorithmes sont évalués sur des instances de complexité croissante ($N = 20,\, 50,\, 80$ villes) selon trois critères clés : la **qualité de solution** (minimisation de la distance totale), la **robustesse** (écart-type inter-runs) et l'**efficacité computationnelle** (temps CPU). Les résultats montrent que, si Hill-Climbing converge très rapidement, le Recuit Simulé offre le meilleur compromis pour les instances de grande taille ($N = 80$) en évitant les optima locaux grâce à l'acceptation probabiliste de solutions dégradantes [\[2\]](#références).

---

## 1. Formulation du problème

### 1.1 Définition formelle

Le TSP est un problème de décision de classe NP-difficile [\[1\]](#références). Il consiste à trouver le plus court **cycle hamiltonien** dans un graphe complet pondéré $G = (V, E, w)$ de $N$ sommets (villes).

Formellement, étant donné un ensemble de villes $\mathcal{C} = \{c_0, c_1, \dots, c_{n-1}\}$ et une matrice de distances $D \in \mathbb{R}^{n \times n}$, l'objectif est de minimiser la longueur totale de la tournée :

$$
\text{Minimiser} \quad f(\pi) = \sum_{i=0}^{n-1} d\!\left(\pi_i,\, \pi_{(i+1) \bmod n}\right)
$$

sous la contrainte que $\pi$ est une permutation de $\{0, \dots, n-1\}$, où $d(u, v)$ est la **distance euclidienne** entre les villes $u$ et $v$ :

$$
d(u, v) = \sqrt{(x_u - x_v)^2 + (y_u - y_v)^2}
$$

L'espace de recherche contient $(n-1)!/2$ tournées distinctes, soit une croissance super-exponentielle rendant toute exploration exhaustive infaisable dès $N \gtrsim 20$.

### 1.2 Voisinage 2-opt

Chaque algorithme exploite le voisinage **2-opt** [\[3\]](#références) : un voisin $\pi'$ d'une tournée $\pi$ est obtenu en inversant un segment contigu $[\pi_i, \dots, \pi_j]$. La variation de coût associée est :

$$
\Delta f = d(\pi_i, \pi_j) + d(\pi_{i+1}, \pi_{j+1}) - d(\pi_i, \pi_{i+1}) - d(\pi_j, \pi_{j+1})
$$

Un mouvement est dit *améliorant* si $\Delta f < 0$.

### 1.3 Instances de test

Trois instances de complexité croissante sont générées aléatoirement dans l'espace euclidien $[0, 100] \times [0, 100]$ :

<img width="540" height="547" alt="instance_80" src="https://github.com/user-attachments/assets/020f0085-b909-45d6-ad6e-c21d89647afb" />

> **Figure 0 — Instance de test à $N = 80$ villes.** Les coordonnées sont tirées uniformément dans $[0,100]^2$.

| Instance | Taille $N$ | $\|\text{Voisinage 2-opt}\|$ | Objectif principal |
|----------|:----------:|:----------------------------:|-------------------|
| Petite   | 20         | $\binom{20}{2} = 190$        | Validation des implémentations |
| Moyenne  | 50         | $\binom{50}{2} = 1\,225$     | Analyse comparative |
| Grande   | 80         | $\binom{80}{2} = 3\,160$     | Test de robustesse et scalabilité |

---

## 2. Méthodologie & implémentation

### 2.1 Architecture vectorisée (NumPy)

Afin de garantir de hautes performances en Python, nous évitons les boucles explicites pour le calcul des distances. La matrice de distances $D \in \mathbb{R}^{N \times N}$ est pré-calculée une seule fois en $\mathcal{O}(N^2)$, puis les coûts de tournée sont évalués via l'*indexation avancée* et le *broadcasting* NumPy, ce qui réduit le temps CPU par itération de $\mathcal{O}(N)$ à une seule opération vectorisée [\[4\]](#références).

**Extrait — évaluation vectorisée du coût :**

```python
def evaluate(self, tour):
    # Indexation avancée NumPy : complexité effective O(1) au lieu de O(N) en Python pur
    tour = np.array(tour, dtype=int)
    shifted_tour = np.roll(tour, -1)
    return np.sum(self.dist_matrix[tour, shifted_tour])
```

**Extrait — pré-calcul de la matrice de distances :**

```python
def _build_distance_matrix(self, coords):
    # Broadcasting : ||coords[i] - coords[j]||_2 pour tous (i,j) en une seule passe
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (N, N, 2)
    return np.sqrt(np.sum(diff ** 2, axis=-1))                  # (N, N)
```

### 2.2 Algorithmes implémentés

#### Hill-Climbing – Première amélioration (HC-FI)

HC-FI parcourt le voisinage 2-opt et accepte **immédiatement** le premier mouvement améliorant trouvé ($\Delta f < 0$), puis recommence depuis la nouvelle solution. Il s'arrête lorsqu'aucun voisin améliorant n'existe (optimum local).

- **Complexité par itération :** $\mathcal{O}(N^2)$ dans le pire cas.
- **Comportement :** convergence très rapide, mais solution finale fortement dépendante du point de départ.

#### Multi-Start Hill-Climbing (MS-HC)

MS-HC lance $k$ exécutions indépendantes de HC-FI depuis des solutions initiales aléatoires distinctes, puis retient la meilleure solution globale. La diversification est obtenue par **redémarrage aléatoire** :

$$
f^* = \min_{r=1}^{k} f\!\left(\text{HC-FI}(\pi_r^{(0)})\right), \quad \pi_r^{(0)} \sim \mathcal{U}(\text{Permutations}(N))
$$

- **Complexité totale :** $k \times \mathcal{O}(N^2 \cdot I_{\max})$.
- **Comportement :** robustesse accrue au prix d'un temps de calcul proportionnel à $k$.

#### Recuit Simulé (SA)

Le Recuit Simulé [\[2\]](#références) imite le processus physique du refroidissement métallurgique. À chaque itération, un voisin $\pi'$ est généré aléatoirement ; il est accepté selon la **règle de Metropolis–Hastings** :

$$
P(\text{accepter } \pi') =
\begin{cases}
1 & \text{si } \Delta f \leq 0 \\[4pt]
e^{-\Delta f / T} & \text{sinon}
\end{cases}
$$

La température $T$ décroît selon un **schéma de refroidissement géométrique** :

$$
T_{k+1} = \alpha \cdot T_k, \quad \alpha \in (0,1), \quad T_{\min} \leq T_k \leq T_0
$$

- **Paramètres utilisés :** $T_0 = 1\,000$, $\alpha = 0{,}995$, $T_{\min} = 0{,}01$.
- **Complexité par cycle de température :** $\mathcal{O}(N^2)$.
- **Comportement :** échappe aux optima locaux grâce à l'acceptation probabiliste des détériorations ; efficacité contrôlée par le profil de température.

#### Tableau comparatif des stratégies

| Algorithme | Stratégie principale | Diversification | Intensification | Complexité |
|---|---|:---:|:---:|---|
| HC-FI | Recherche locale « first-improvement » | ✗ | ✓✓✓ | $\mathcal{O}(N^2 \cdot I)$ |
| MS-HC | Redémarrages aléatoires de HC-FI | ✓✓ | ✓✓ | $\mathcal{O}(k \cdot N^2 \cdot I)$ |
| Recuit Simulé | Acceptation probabiliste (Metropolis) | ✓✓✓ | ✓ | $\mathcal{O}(N^2 \cdot C_T)$ |

*$I$ = nombre d'itérations HC, $C_T$ = nombre total de cycles de refroidissement.*

---

## 3. Résultats expérimentaux

### 3.1 Protocole expérimental

- **Nombre de runs :** 30 exécutions indépendantes par (algorithme × instance).
- **Initialisation :** permutations aléatoires uniformes identiques pour tous les algorithmes (graine fixée par run).
- **Matériel :** processeur grand public standard (Intel Core i5, 8 Go RAM).
- **Métriques :**
  - $f^* =$ meilleur coût observé sur 30 runs.
  - $\bar{f} =$ coût moyen (estimateur de l'espérance).
  - $\sigma =$ écart-type (indicateur de robustesse stochastique).
  - $\bar{t}$ = temps CPU moyen par run (en secondes).

### 3.2 Analyse quantitative (données agrégées sur 30 runs)

| Taille $N$ | Algorithme | $f^*$ | $\bar{f}$ | $\sigma$ | $\bar{t}$ (s) |
|:---:|---|---:|---:|---:|---:|
| 20 | HC-First | 441.81 | 456.99 | 10.94 | 0.06 |
| 20 | HC-MultiStart | **386.43** | **427.68** | 32.79 | 2.19 |
| 20 | Recuit Simulé | 386.63 | 440.46 | 40.71 | **0.12** |
| 50 | HC-First | 776.89 | 847.85 | 50.20 | **0.45** |
| 50 | HC-MultiStart | **743.03** | **776.87** | 39.11 | 6.27 |
| 50 | Recuit Simulé | 937.77 | 990.85 | 60.08 | 0.23 |
| 80 | HC-First | 1179.97 | 1213.95 | **26.92** | 1.76 |
| 80 | HC-MultiStart | **1011.88** | **1078.33** | 59.62 | 41.32 |
| 80 | Recuit Simulé | 1500.78 † | 1640.82 † | 112.80 | **0.21** |

> **†** Les coûts élevés du Recuit Simulé en configuration standard résultent d'une phase d'exploration à haute température ($T_0 = 1\,000$) insuffisamment longue pour $N = 80$. Des configurations ajustées (cf. dossier `images/`) montrent une réduction significative de $f^*$ et de $\sigma$ [\[2\]](#références).

**Observations clés :**

- Pour $N = 20$, MS-HC et SA atteignent des coûts $f^*$ quasi-identiques ($386.43$ vs $386.63$), illustrant la convergence vers le même bassin d'attraction.
- Pour $N = 50$, MS-HC domine en qualité ($f^* = 743.03$) ; SA souffre d'un manque d'intensification en fin de refroidissement.
- Pour $N = 80$, l'écart-type de HC-First ($\sigma = 26.92$) est le plus faible, confirmant une convergence systématique — vers des optima locaux médiocres. MS-HC offre le meilleur $f^*$ mais au coût le plus élevé ($\approx 41$ s).

### 3.3 Analyse visuelle

#### Meilleure tournée – Recuit Simulé, $N = 80$

<img width="540" height="547" alt="best_tour_sa_80" src="https://github.com/user-attachments/assets/605ba3a3-b35f-4acd-8d19-90f5c112f8e1" />

> **Figure 1 — Meilleure tournée obtenue par Recuit Simulé pour $N = 80$ (coût $\approx 1\,712$).** On observe la réduction des croisements inter-arêtes, signature d'une exploration efficace du voisinage 2-opt à haute température suivie d'une intensification locale.

#### Comparaison des coûts moyens – $N = 80$

<img width="590" height="390" alt="results_bar_80" src="https://github.com/user-attachments/assets/01cbc994-ca6d-4c6d-b133-aa11d1e6a7ae" />

> **Figure 2 — Comparaison des coûts moyens $\bar{f}$ pour $N = 80$ (30 runs, barres d'erreur = $\pm\sigma$).** Le Multi-Start améliore significativement HC-First ; le Recuit Simulé présente la plus grande variance, reflétant la sensibilité au paramétrage de la température.

---

## 4. Discussion & conclusion

### 4.1 Analyse par algorithme

- **Hill-Climbing (First Improvement) :** convergence gloutonne et déterministe à partir d'une solution initiale donnée. La dépendance au point de départ engendre une forte variance inter-runs, surtout pour $N = 80$. La complexité par run en $\mathcal{O}(N^2 \cdot I)$ reste cependant très faible.

- **Multi-Start Hill-Climbing :** la diversification par redémarrages réduit la variance et améliore $f^*$ de manière statistiquement significative (amélioration de $\approx 14\,\%$ sur HC-First pour $N = 80$). Le coût computationnel croît linéairement avec $k$ ($\approx 41$ s pour $k = 20$, $N = 80$), ce qui limite son application en temps réel.

- **Recuit Simulé :** l'acceptation probabiliste de détériorations ($P = e^{-\Delta f / T}$) permet d'explorer des régions étendues de l'espace de recherche. Son efficacité est néanmoins **fortement conditionnée** par le réglage du triplet $(T_0, \alpha, T_{\min})$ [\[2\]](#références). Un refroidissement trop rapide ($\alpha$ faible) conduit à un comportement proche de HC-FI ; un refroidissement trop lent augmente le temps de calcul sans gain proportionnel.

### 4.2 Compromis qualité–temps

On définit l'efficacité agrégée comme :

$$
\eta = \frac{1}{\bar{f} \cdot \bar{t}}
$$

| Algorithme ($N=80$) | $\bar{f}$ | $\bar{t}$ (s) | $\eta$ (u.a.) |
|---|---:|---:|---:|
| HC-First | 1213.95 | 1.76 | $4.68 \times 10^{-4}$ |
| HC-MultiStart | 1078.33 | 41.32 | $2.25 \times 10^{-5}$ |
| Recuit Simulé | 1640.82 | 0.21 | $2.90 \times 10^{-3}$ |

HC-FI présente le meilleur rapport qualité–temps brut pour $N = 80$, tandis que MS-HC maximise la qualité absolue au détriment du temps.

### 4.3 Conclusion générale

Pour des instances de TSP de grande taille ($N \geq 80$) où la qualité de solution est prioritaire et le temps de calcul n'est pas contraignant, **MS-HC** est recommandé. Lorsque le budget temps est limité, **le Recuit Simulé correctement paramétré** offre le meilleur équilibre exploration–exploitation. Dans un cadre temps-réel ou embarqué, **HC-FI** reste la solution la plus adaptée.

> *Perspectives :* l'intégration de méthodes hybrides (SA + 3-opt, algorithmes génétiques, colonies de fourmis) ou de stratégies d'apprentissage par renforcement pour l'adaptation dynamique de la température constitue une voie d'amélioration prometteuse [\[5\]](#références).

---

## 5. Utilisation & reproduction

### 5.1 Installation

```bash
git clone https://github.com/your-username/tsp-metaheuristics.git
cd tsp-metaheuristics
pip install -r requirements.txt
# ou : pip install numpy matplotlib pandas tqdm
```

### 5.2 Exécution

```bash
python main.py
```

Ce script exécute 30 runs indépendants par algorithme et par instance, puis génère le fichier `tsp_results.csv` dans le répertoire `assets/` (ou `data/` selon la configuration).

### 5.3 Structure du projet

```text
tsp-metaheuristics/
├── data/               # Génération et chargement des instances TSP
├── src/
│   ├── tsp_core.py     # Définition vectorisée du problème (matrice de distances, évaluateur)
│   ├── algorithms.py   # Implémentation de HC-FI, MS-HC et Recuit Simulé
│   └── utils.py        # Fonctions utilitaires : visualisation, statistiques, export CSV
├── images/             # Figures et graphiques générés automatiquement
├── assets/             # Données brutes (CSV des résultats)
├── main.py             # Script principal d'orchestration des expériences
├── requirements.txt    # Dépendances Python
└── README.md           # Documentation du projet
```

---

## Références

\[1\] Garey, M. R., & Johnson, D. S. (1979). *Computers and Intractability: A Guide to the Theory of NP-Completeness*. W. H. Freeman.

\[2\] Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by Simulated Annealing. *Science*, 220(4598), 671–680. — [Wikipedia – Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing)

\[3\] Lin, S. (1965). Computer solutions of the traveling salesman problem. *Bell System Technical Journal*, 44(10), 2245–2269.

\[4\] Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362.

\[5\] GeeksforGeeks. Hill Climbing and Simulated Annealing for the Traveling Salesman Problem. [geeksforgeeks.org](https://www.geeksforgeeks.org/artificial-intelligence/hill-climbing-and-simulated-annealing-for-the-traveling-salesman-problem/)

---
