import numpy as np
import matplotlib.pyplot as plt
import time
import heapq
import json
import yaml
from PIL import Image

# --- 1. CHARGEMENT DES DONNÉES DE DESTINATION ---
with open('../coordonées/destination.json','r', encoding='utf-8') as f:
    data = json.load(f)

x_start_m = data["start_position"]["x"]
y_start_m = data["start_position"]["y"]

# Extraction des coordonnées de la table 1
table_1 = next(t for t in data["tables"] if t["id"] == "table_1")
x_goal_m = table_1["delivery_point"]["x"]
y_goal_m = table_1["delivery_point"]["y"]

# --- 2. CHARGEMENT DE LA CARTE (YAML + PGM/PNG) ---
with open('../map/map.yaml', 'r') as file:
    map_info = yaml.safe_load(file)

img = Image.open('../map/' + map_info["image"])
grid_pixels = np.array(img)

# Conversion en grille binaire : 0 = Libre, 1 = Obstacle
grille_brute = np.where(grid_pixels < 250, 1, 0) 
grid = grille_brute.T  # Transpose pour le calcul [x, y]

# Paramètres de transformation
origin_x = map_info["origin"][0]
origin_y = map_info["origin"][1]
res = map_info["resolution"]
hauteur_image = grille_brute.shape[0]

def meters_to_grid(x_meters, y_meters):
    col_x = int(round((x_meters - origin_x) / res))
    row_y = int(round((y_meters - origin_y) / res))
    row_y_image = (hauteur_image - 1) - row_y
    return (col_x, row_y_image)

start = meters_to_grid(x_start_m, y_start_m)
goal = meters_to_grid(x_goal_m, y_goal_m)

# --- 3. ALGORITHME GREEDY (BEST-FIRST SEARCH) ---

def heuristic_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def Neighbors(node):
    x, y = node
    neighbors = []
    width, height = grid.shape
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height and grid[nx, ny] == 0:
            neighbors.append((nx, ny))
    return neighbors

def greedy(grid, start, goal, heuristic_func):
    openlist = []
    closedlist = set()
    parent = {}
    path = []

    # Dans Greedy, on trie uniquement par l'heuristique (distance estimée au but)
    heapq.heappush(openlist, (heuristic_func(start, goal), start))

    while openlist:
        _, current = heapq.heappop(openlist)

        if current in closedlist:
            continue
            
        if current == goal:
            # Reconstruction du chemin en remontant les parents
            temp = current
            while temp in parent:
                path.append(temp)
                temp = parent[temp]
            path.append(start)
            path.reverse() 
            # Retourne : chemin, noeuds explorés, et coût (longueur du chemin)
            return path, len(closedlist), len(path) - 1
            
        closedlist.add(current)

        for neighbor in Neighbors(current):
            if neighbor in closedlist:
                continue
                
            # Si le voisin n'a pas encore été visité ou n'est pas dans la pile
            if neighbor not in parent and neighbor != start:
                parent[neighbor] = current
                h_score = heuristic_func(neighbor, goal)
                heapq.heappush(openlist, (h_score, neighbor))
                
    return [], len(closedlist), 0

# --- 4. EXÉCUTION ET AFFICHAGE ---
print(f"Lancement de Greedy de {start} vers {goal}...")
start_time = time.time()

# Appel de la fonction avec les 3 variables de retour
path_result, num_nodes, cost = greedy(grid, start, goal, heuristic_manhattan)

execution_time = time.time() - start_time

print(f"Terminé en {execution_time:.4f} secondes !")
print(f"Coût du chemin : {cost} | Nœuds explorés : {num_nodes}")

# --- 5. VISUALISATION ---
plt.figure(figsize=(10, 8))
plt.imshow(grille_brute, cmap='binary', origin='upper')

if path_result:
    path_x = [p[0] for p in path_result]
    path_y = [p[1] for p in path_result]
    plt.plot(path_x, path_y, color='orange', linewidth=2, label="Chemin Greedy")
else:
    print("⚠️ Aucun chemin trouvé !")

plt.scatter(start[0], start[1], color='green', marker='o', s=100, label="Départ", zorder=5)
plt.scatter(goal[0], goal[1], color='red', marker='x', s=100, label="Arrivée (Table 1)", zorder=5)

plt.title(f"Pathfinding Greedy (Heuristique seule)\nExplorés: {num_nodes} | Temps: {execution_time:.4f}s")
plt.legend()
plt.show()