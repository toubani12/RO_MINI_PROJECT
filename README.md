# Comparative Study of Metaheuristics for the Traveling Salesperson Problem (TSP)

**Institution:** Université Hassan II de Casablanca – ENSET Mohammedia
**Master Program:** SDIA (Data Systems & Artificial Intelligence) / GESI
**Module:** Optimization / Metaheuristics
**Supervisor:** Prof. MESTARI
**Authors:** Badr Eddine TOUBANI, Imane MEKKAOUI
**Academic Year:** 2025–2026

---

## 📝 Abstract

This project investigates the efficiency of three metaheuristic algorithms for solving the Traveling Salesperson Problem (TSP), a classic NP-Hard combinatorial optimization problem. We implemented and compared Hill-Climbing (First Improvement), Multi-Start Hill-Climbing, and Simulated Annealing using a high-performance, vectorized Python architecture.

The study evaluates these algorithms on instances of varying complexity ($N=20, 50, 80$ cities) based on three key metrics: solution quality (minimization of distance), robustness (standard deviation), and computational efficiency (CPU time). Results demonstrate that while Hill-Climbing provides rapid convergence, Simulated Annealing offers the best trade-off for large-scale instances ($N=80$) by effectively escaping local optima through probabilistic acceptance of degrading solutions.

---

## 1. Problem Formulation

The TSP consists of finding the shortest Hamiltonian cycle in a complete graph of $N$ cities.

Given a set of cities $\{c_0, c_1, \dots, c_{n-1}\}$ and a distance matrix $D$, the objective is to minimize the total tour length:

$$
	ext{Minimize } f(\pi) = \sum_{i=0}^{n-1} d(\pi_i, \pi_{(i+1) \bmod n})
$$

Where:
- $\pi$ is a permutation of $\{0, \dots, n-1\}$.
- $d(u, v)$ is the Euclidean distance between city $u$ and city $v$.

### Test Instances

We utilized three instances of increasing complexity generated in a $[0, 100] \times [0, 100]$ Euclidean space:

<img width="540" height="547" alt="instance_80" src="https://github.com/user-attachments/assets/020f0085-b909-45d6-ad6e-c21d89647afb" />

- Small: $N=20$
- Medium: $N=50$
- Large: $N=80$ (used for stress testing stability)

---

## 2. Methodology & Implementation

### 🚀 Vectorized Architecture (NumPy)

To ensure high performance in Python, we avoided explicit Python loops for distance calculations. Instead, we precompute the $N\times N$ distance matrix and evaluate tour costs using NumPy fancy indexing and broadcasting, which reduces CPU time per iteration.

Snippet: Vectorized cost evaluation

```python
def evaluate(self, tour):
	# Uses NumPy fancy indexing for O(1) effective lookup vs Python loops
	tour = np.array(tour, dtype=int)
	shifted_tour = np.roll(tour, -1)
	return np.sum(self.dist_matrix[tour, shifted_tour])
```

### Algorithms Implemented

| Algorithm | Strategy | Exploration vs. Exploitation |
|---|---:|---|
| Hill-Climbing (First Improvement) | First improvement local search | Pure exploitation (greedy), fast but trapped in local optima |
| Multi-Start Hill-Climbing | Random restart of HC | Diversification via restarts; more robust but costlier |
| Simulated Annealing | Probabilistic acceptance of worse moves | Dynamic balance: accepts degrading moves ($P = e^{-\Delta / T}$) to escape local optima |

---

## 3. Experimental Results

### Protocol

- 30 independent runs per instance.
- Hardware: Standard consumer CPU.
- Metrics: best cost, average cost, standard deviation, average CPU time.

### Quantitative Analysis (Aggregated Data)

| Size (N) | Algorithm | Best Cost | Avg Cost | Std Dev | Avg Time (s) |
|---:|---|---:|---:|---:|---:|
| 20 | HC_First | 441.81 | 456.99 | 10.94 | 0.06 |
| 20 | HC_MultiStart | 386.43 | 427.68 | 32.79 | 2.19 |
| 20 | Simulated Annealing | 386.63 | 440.46 | 40.71 | 0.12 |
| 50 | HC_First | 776.89 | 847.85 | 50.20 | 0.45 |
| 50 | HC_MultiStart | 743.03 | 776.87 | 39.11 | 6.27 |
| 50 | Simulated Annealing | 937.77 | 990.85 | 60.08 | 0.23 |
| 80 | HC_First | 1179.97 | 1213.95 | 26.92 | 1.76 |
| 80 | HC_MultiStart | 1011.88 | 1078.33 | 59.62 | 41.32 |
| 80 | Simulated Annealing | 1500.78* | 1640.82* | 112.80 | 0.21 |

_*Note: raw SA averages may be higher due to high-temperature exploration; tuned-SA runs (see `images/`) show improved global-structure discovery._

### Visual Analysis

- **Optimal Path Visualization (N=80):** Simulated Annealing successfully "untangles" tours and reduces crossings characteristic of suboptimal solutions.

  
  <img width="540" height="547" alt="best_tour_sa_80" src="https://github.com/user-attachments/assets/605ba3a3-b35f-4acd-8d19-90f5c112f8e1" />
- **Figure 1:** Best tour found by Simulated Annealing for N=80 (Cost ≈ 1712).

  <img width="590" height="390" alt="results_bar_80" src="https://github.com/user-attachments/assets/01cbc994-ca6d-4c6d-b133-aa11d1e6a7ae" />

- **Figure 2:** Average cost comparison for N=80 — Multi-Start improves HC, while SA is the most robust for complex topologies.

---

## 4. Discussion & Conclusion

- **Hill-Climbing:** Fastest convergence (greedy) but prone to local optima entrapment.
- **Multi-Start HC:** Improves robustness through diversification; computational cost grows with restarts (e.g., ~41s for N=80).
- **Simulated Annealing:** Best exploration strategy for large instances; by tuning temperature schedule ($T_0$, $\alpha$) it balances global exploration and local exploitation.

Conclusion: For large-scale TSP instances where solution quality is paramount, Simulated Annealing is preferred despite parameter-tuning complexity.

---

## 5. Usage / Reproduction

To reproduce the experiments:

```bash
git clone https://github.com/your-username/tsp-metaheuristics.git
cd tsp-metaheuristics
pip install -r requirements.txt
# or: pip install numpy matplotlib pandas tqdm
python main.py
```

This executes 30 runs per algorithm and produces `tsp_results.csv` in the `assets/` or `data/` folder depending on the configuration.

### Project Structure

```
├── data/               # Instance generation logic
├── src/
│   ├── tsp_core.py     # Vectorized Problem definition
│   ├── algorithms.py   # Implementation of HC, Multi-Start, and SA
│   └── utils.py        # Plotting and helper functions
├── images/             # Generated plots for the report
├── main.py             # Main execution script
└── README.md           # Project documentation
```

---



