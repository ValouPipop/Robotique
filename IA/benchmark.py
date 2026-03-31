"""
==========================================================================
 BENCHMARK : Navigation Classique (A*, Dijkstra, Greedy) vs IA (PPO)
==========================================================================
Compare les deux paradigmes de navigation sur les mêmes critères :
  - Efficacité du chemin (longueur)
  - Taux de succès
  - Évitement d'obstacles
  - Temps de calcul
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Pour sauvegarder sans afficher
import matplotlib.pyplot as plt
import time
import heapq
import json
import yaml
import math
import sys
import os
from PIL import Image
from scipy.ndimage import binary_dilation
from collections import deque

# Charger le modèle IA si disponible
try:
    from stable_baselines3 import PPO
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from robot_env import FastRobot2DEnv
    IA_AVAILABLE = True
except ImportError:
    IA_AVAILABLE = False
    print("⚠️ stable_baselines3 ou robot_env non disponible, test IA désactivé")

# =====================================================================
# 1. CHARGEMENT DE LA MAP ET DES DONNÉES
# =====================================================================
with open('../coordonées/destination.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('../map/map.yaml', 'r') as f:
    map_info = yaml.safe_load(f)

img = Image.open('../map/' + map_info["image"])
grid_pixels = np.array(img)
grille_brute = np.where(grid_pixels < 250, 1, 0)
res = map_info["resolution"]
origin_x = map_info["origin"][0]
origin_y = map_info["origin"][1]

# Inflation des obstacles
rayon_robot_m = 0.20
marge_securite_m = 0.05
rayon_total_m = rayon_robot_m + marge_securite_m
rayon_pixels = int(np.ceil(rayon_total_m / res))
y_g, x_g = np.ogrid[-rayon_pixels:rayon_pixels+1, -rayon_pixels:rayon_pixels+1]
masque = x_g**2 + y_g**2 <= rayon_pixels**2
grille_epaisse = binary_dilation(grille_brute, structure=masque).astype(int)
grid = grille_epaisse.T  # grid[col_x, row_y] pour les algos classiques
hauteur_image = grille_epaisse.shape[0]

x_start = data["start_position"]["x"]
y_start = data["start_position"]["y"]
tables = data["tables"]

def meters_to_grid(x_m, y_m):
    col_x = int(round((x_m - origin_x) / res))
    row_y = int(round((y_m - origin_y) / res))
    row_y_image = (hauteur_image - 1) - row_y
    return (col_x, row_y_image)

start_grid = meters_to_grid(x_start, y_start)

# =====================================================================
# 2. ALGORITHMES DE PATHFINDING CLASSIQUES
# =====================================================================

def get_neighbors(node):
    x, y = node
    neighbors = []
    width, height = grid.shape
    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height and grid[nx, ny] == 0:
            neighbors.append((nx, ny))
    return neighbors

def heuristic_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start, goal):
    openlist = []
    closedlist = set()
    g = {start: 0}
    parent = {}
    heapq.heappush(openlist, (heuristic_manhattan(start, goal), heuristic_manhattan(start, goal), start))
    while openlist:
        _, _, current = heapq.heappop(openlist)
        if current in closedlist:
            continue
        if current == goal:
            path = []
            cost = g[current]
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            path.reverse()
            return path, len(closedlist), cost
        closedlist.add(current)
        for neighbor in get_neighbors(current):
            if neighbor in closedlist:
                continue
            tentative_g = g[current] + 1
            if tentative_g < g.get(neighbor, float('inf')):
                parent[neighbor] = current
                g[neighbor] = tentative_g
                f_score = tentative_g + heuristic_manhattan(neighbor, goal)
                heapq.heappush(openlist, (f_score, heuristic_manhattan(neighbor, goal), neighbor))
    return [], 0, 0

def dijkstra(start, goal):
    openlist = []
    closedlist = set()
    g = {start: 0}
    parent = {}
    heapq.heappush(openlist, (0, start))
    while openlist:
        current_g, current = heapq.heappop(openlist)
        if current in closedlist:
            continue
        if current == goal:
            path = []
            cost = g[current]
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            return path[::-1], len(closedlist), cost
        closedlist.add(current)
        for neighbor in get_neighbors(current):
            if neighbor in closedlist:
                continue
            tentative_g = g[current] + 1
            if tentative_g < g.get(neighbor, float('inf')):
                parent[neighbor] = current
                g[neighbor] = tentative_g
                heapq.heappush(openlist, (tentative_g, neighbor))
    return [], 0, 0

def greedy(start, goal):
    openlist = []
    closedlist = set()
    parent = {}
    heapq.heappush(openlist, (heuristic_manhattan(start, goal), start))
    while openlist:
        _, current = heapq.heappop(openlist)
        if current in closedlist:
            continue
        if current == goal:
            path = []
            temp = current
            while temp in parent:
                path.append(temp)
                temp = parent[temp]
            path.append(start)
            path.reverse()
            return path, len(closedlist), len(path) - 1
        closedlist.add(current)
        for neighbor in get_neighbors(current):
            if neighbor in closedlist:
                continue
            if neighbor not in parent and neighbor != start:
                parent[neighbor] = current
                heapq.heappush(openlist, (heuristic_manhattan(neighbor, goal), neighbor))
    return [], 0, 0

# =====================================================================
# 3. TEST DE L'IA (PPO)
# =====================================================================

def test_ia_for_table(env, model, table):
    """Teste l'IA sur une table spécifique et retourne les métriques."""
    import random
    # On force la table cible dans le reset
    obs, _ = env.reset()
    env.goal_x = table["delivery_point"]["x"]
    env.goal_y = table["delivery_point"]["y"]
    env.distance_map = env.all_distance_maps[table["id"]]
    row, col = env._meters_to_grid(env.robot_x, env.robot_y)
    env.previous_real_dist = env.distance_map[row, col] * env.res
    obs = env._get_state()
    
    done = False
    truncated = False
    total_steps = 0
    path_length_m = 0.0
    prev_x, prev_y = env.robot_x, env.robot_y
    min_obstacle_dist = float('inf')
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_steps += 1
        
        # Calcul de la distance parcourue
        dx = env.robot_x - prev_x
        dy = env.robot_y - prev_y
        path_length_m += math.sqrt(dx*dx + dy*dy)
        prev_x, prev_y = env.robot_x, env.robot_y
        
        # Distance minimale aux obstacles (via LiDAR)
        scan = env._simulate_lidar()
        min_scan = min(scan)
        if min_scan < min_obstacle_dist:
            min_obstacle_dist = min_scan
        
        if total_steps > 2000:
            break
    
    dist_finale = np.hypot(env.goal_x - env.robot_x, env.goal_y - env.robot_y)
    success = dist_finale < 0.4 and not env.crash
    
    return {
        "success": success,
        "crash": env.crash,
        "path_length_m": path_length_m,
        "steps": total_steps,
        "min_obstacle_dist": min_obstacle_dist,
        "dist_finale": dist_finale
    }

# =====================================================================
# 4. BENCHMARK PRINCIPAL
# =====================================================================

def run_benchmark():
    print("=" * 70)
    print("   BENCHMARK : Navigation Classique vs IA (PPO Actor-Critic)")
    print("=" * 70)
    
    results = {
        "A*": {"paths": [], "times": [], "nodes": [], "costs": [], "successes": 0},
        "Dijkstra": {"paths": [], "times": [], "nodes": [], "costs": [], "successes": 0},
        "Greedy": {"paths": [], "times": [], "nodes": [], "costs": [], "successes": 0},
    }
    
    if IA_AVAILABLE:
        results["IA (PPO)"] = {"paths_m": [], "times": [], "steps": [], "successes": 0, "min_obstacles": [], "crashes": 0}
    
    # --- Test sur chaque table ---
    for table in tables:
        tid = table["id"]
        gx = table["delivery_point"]["x"]
        gy = table["delivery_point"]["y"]
        goal_grid = meters_to_grid(gx, gy)
        
        print(f"\n{'─'*50}")
        print(f"  Table : {tid} → ({gx}, {gy})")
        print(f"{'─'*50}")
        
        # --- A* ---
        t0 = time.time()
        path, nodes, cost = a_star(start_grid, goal_grid)
        dt = time.time() - t0
        success = len(path) > 0
        results["A*"]["paths"].append(path)
        results["A*"]["times"].append(dt)
        results["A*"]["nodes"].append(nodes)
        results["A*"]["costs"].append(cost)
        if success: results["A*"]["successes"] += 1
        path_m = cost * res
        print(f"  A*       | {'✅' if success else '❌'} | {path_m:.2f}m | {dt*1000:.1f}ms | {nodes} nœuds")
        
        # --- Dijkstra ---
        t0 = time.time()
        path, nodes, cost = dijkstra(start_grid, goal_grid)
        dt = time.time() - t0
        success = len(path) > 0
        results["Dijkstra"]["paths"].append(path)
        results["Dijkstra"]["times"].append(dt)
        results["Dijkstra"]["nodes"].append(nodes)
        results["Dijkstra"]["costs"].append(cost)
        if success: results["Dijkstra"]["successes"] += 1
        path_m = cost * res
        print(f"  Dijkstra | {'✅' if success else '❌'} | {path_m:.2f}m | {dt*1000:.1f}ms | {nodes} nœuds")
        
        # --- Greedy ---
        t0 = time.time()
        path, nodes, cost = greedy(start_grid, goal_grid)
        dt = time.time() - t0
        success = len(path) > 0
        results["Greedy"]["paths"].append(path)
        results["Greedy"]["times"].append(dt)
        results["Greedy"]["nodes"].append(nodes)
        results["Greedy"]["costs"].append(cost)
        if success: results["Greedy"]["successes"] += 1
        path_m = cost * res
        print(f"  Greedy   | {'✅' if success else '❌'} | {path_m:.2f}m | {dt*1000:.1f}ms | {nodes} nœuds")
        
        # --- IA (PPO) ---
        if IA_AVAILABLE:
            env = FastRobot2DEnv()
            model = PPO.load("best_model/best_model.zip", env=env, device="cpu")
            
            # On fait N essais pour mesurer le taux de succès
            n_trials = 5
            trial_results = []
            for _ in range(n_trials):
                t0 = time.time()
                r = test_ia_for_table(env, model, table)
                dt = time.time() - t0
                r["time"] = dt
                trial_results.append(r)
            
            successes = sum(1 for r in trial_results if r["success"])
            avg_path = np.mean([r["path_length_m"] for r in trial_results])
            avg_time = np.mean([r["time"] for r in trial_results])
            avg_steps = np.mean([r["steps"] for r in trial_results])
            avg_min_obs = np.mean([r["min_obstacle_dist"] for r in trial_results])
            crashes = sum(1 for r in trial_results if r["crash"])
            
            results["IA (PPO)"]["paths_m"].append(avg_path)
            results["IA (PPO)"]["times"].append(avg_time)
            results["IA (PPO)"]["steps"].append(avg_steps)
            results["IA (PPO)"]["successes"] += successes
            results["IA (PPO)"]["min_obstacles"].append(avg_min_obs)
            results["IA (PPO)"]["crashes"] += crashes
            
            print(f"  IA (PPO) | {successes}/{n_trials} ✅ | {avg_path:.2f}m | {avg_time*1000:.1f}ms | min_obs={avg_min_obs:.2f}m")
    
    # =====================================================================
    # 5. TABLEAU RÉCAPITULATIF
    # =====================================================================
    n_tables = len(tables)
    n_ia_trials = n_tables * 5 if IA_AVAILABLE else 0
    
    print("\n")
    print("=" * 80)
    print("                    TABLEAU RÉCAPITULATIF")
    print("=" * 80)
    print(f"{'Critère':<30} {'A*':>10} {'Dijkstra':>10} {'Greedy':>10}", end="")
    if IA_AVAILABLE:
        print(f" {'IA (PPO)':>10}", end="")
    print()
    print("─" * 80)
    
    # Taux de succès
    print(f"{'Taux de succès':<30}", end="")
    for algo in ["A*", "Dijkstra", "Greedy"]:
        rate = results[algo]["successes"] / n_tables * 100
        print(f" {rate:>8.0f}% ", end="")
    if IA_AVAILABLE:
        rate = results["IA (PPO)"]["successes"] / n_ia_trials * 100
        print(f" {rate:>8.0f}% ", end="")
    print()
    
    # Longueur moyenne du chemin (en mètres)
    print(f"{'Longueur moy. chemin (m)':<30}", end="")
    for algo in ["A*", "Dijkstra", "Greedy"]:
        costs = results[algo]["costs"]
        avg = np.mean(costs) * res if costs else 0
        print(f" {avg:>9.2f} ", end="")
    if IA_AVAILABLE:
        avg = np.mean(results["IA (PPO)"]["paths_m"])
        print(f" {avg:>9.2f} ", end="")
    print()
    
    # Temps de calcul moyen
    print(f"{'Temps calcul moy. (ms)':<30}", end="")
    for algo in ["A*", "Dijkstra", "Greedy"]:
        avg = np.mean(results[algo]["times"]) * 1000
        print(f" {avg:>9.1f} ", end="")
    if IA_AVAILABLE:
        avg = np.mean(results["IA (PPO)"]["times"]) * 1000
        print(f" {avg:>9.1f} ", end="")
    print()
    
    # Nœuds explorés / Steps
    print(f"{'Nœuds explorés (moy.)':<30}", end="")
    for algo in ["A*", "Dijkstra", "Greedy"]:
        avg = np.mean(results[algo]["nodes"])
        print(f" {avg:>9.0f} ", end="")
    if IA_AVAILABLE:
        avg = np.mean(results["IA (PPO)"]["steps"])
        print(f" {avg:>9.0f} ", end="")
    print()
    
    # Évitement d'obstacles (IA seulement)
    if IA_AVAILABLE:
        print(f"{'Dist. min obstacle (m)':<30}", end="")
        for algo in ["A*", "Dijkstra", "Greedy"]:
            # Les algos classiques ont une marge fixe de 25cm (inflation)
            print(f" {'≥0.25':>9} ", end="")
        avg_obs = np.mean(results["IA (PPO)"]["min_obstacles"])
        print(f" {avg_obs:>9.2f} ", end="")
        print()
        
        print(f"{'Crashes':<30}", end="")
        for algo in ["A*", "Dijkstra", "Greedy"]:
            print(f" {'0':>9} ", end="")
        print(f" {results['IA (PPO)']['crashes']:>9} ", end="")
        print()
    
    print("─" * 80)
    
    # =====================================================================
    # 6. GÉNÉRATION DES GRAPHIQUES
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Benchmark : Navigation Classique vs IA (PPO)", fontsize=14, fontweight='bold')
    
    algos = ["A*", "Dijkstra", "Greedy"]
    if IA_AVAILABLE:
        algos.append("IA (PPO)")
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    table_names = [t["id"] for t in tables]
    
    # Graph 1 : Longueur du chemin par table
    ax = axes[0, 0]
    x = np.arange(len(tables))
    width = 0.2
    for i, algo in enumerate(algos):
        if algo == "IA (PPO)":
            vals = results[algo]["paths_m"]
        else:
            vals = [c * res for c in results[algo]["costs"]]
        ax.bar(x + i * width, vals, width, label=algo, color=colors[i])
    ax.set_xlabel("Table")
    ax.set_ylabel("Longueur du chemin (m)")
    ax.set_title("Efficacité du chemin")
    ax.set_xticks(x + width * (len(algos)-1) / 2)
    ax.set_xticklabels(table_names, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Graph 2 : Temps de calcul par table
    ax = axes[0, 1]
    for i, algo in enumerate(algos):
        vals = [t * 1000 for t in results[algo]["times"]]
        ax.bar(x + i * width, vals, width, label=algo, color=colors[i])
    ax.set_xlabel("Table")
    ax.set_ylabel("Temps (ms)")
    ax.set_title("Temps de calcul")
    ax.set_xticks(x + width * (len(algos)-1) / 2)
    ax.set_xticklabels(table_names, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Graph 3 : Taux de succès global
    ax = axes[1, 0]
    success_rates = []
    for algo in algos:
        if algo == "IA (PPO)":
            rate = results[algo]["successes"] / n_ia_trials * 100
        else:
            rate = results[algo]["successes"] / n_tables * 100
        success_rates.append(rate)
    bars = ax.bar(algos, success_rates, color=colors[:len(algos)])
    ax.set_ylabel("Taux de succès (%)")
    ax.set_title("Taux de réussite")
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2, f'{val:.0f}%', ha='center', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Graph 4 : Nœuds explorés / Steps
    ax = axes[1, 1]
    node_vals = []
    for algo in algos:
        if algo == "IA (PPO)":
            node_vals.append(np.mean(results[algo]["steps"]))
        else:
            node_vals.append(np.mean(results[algo]["nodes"]))
    bars = ax.bar(algos, node_vals, color=colors[:len(algos)])
    ax.set_ylabel("Nœuds explorés / Steps IA")
    ax.set_title("Coût computationnel")
    for bar, val in zip(bars, node_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(node_vals)*0.02, f'{val:.0f}', ha='center', fontweight='bold', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("benchmark_comparaison.png", dpi=150)
    print(f"\n📊 Graphique sauvegardé dans 'benchmark_comparaison.png'")
    
    return results

if __name__ == '__main__':
    run_benchmark()
