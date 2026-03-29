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
# Note : On accède directement via la liste pour plus de simplicité
table_1 = next(t for t in data["tables"] if t["id"] == "table_1")
x_goal_m = table_1["delivery_point"]["x"]
y_goal_m = table_1["delivery_point"]["y"]

# --- 2. CHARGEMENT DE LA CARTE (YAML + PGM/PNG) ---
with open('../map/map.yaml', 'r') as file:
    map_info = yaml.safe_load(file)

img = Image.open('../map/' + map_info["image"])
grid_pixels = np.array(img)

# Conversion en grille binaire : 0 = Libre, 1 = Obstacle
# On considère souvent que < 250 (gris/noir) est un obstacle
grille_brute = np.where(grid_pixels < 250, 1, 0) 
grid = grille_brute.T  # Transpose pour avoir grid[x, y]

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

# --- 3. ALGORITHME DE DIJKSTRA ---

def Neighbors(node, grid):
    x, y = node
    neighbors = []
    width, height = grid.shape
    # Déplacements cardinaux (Haut, Bas, Droite, Gauche)
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            if grid[nx, ny] == 0:  # Si la case est libre
                neighbors.append((nx, ny))
    return neighbors

def dijkstra(grid, start, goal):
    openlist = []
    closedlist = set()
    g = {start: 0}
    parent = {}
    
    # Dans Dijkstra, on trie uniquement par le coût cumulé g
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

        for neighbor in Neighbors(current, grid):
            if neighbor in closedlist:
                continue
                
            tentative_g = g[current] + 1 # Coût de 1 par case
            
            if tentative_g < g.get(neighbor, float('inf')):
                parent[neighbor] = current
                g[neighbor] = tentative_g
                heapq.heappush(openlist, (tentative_g, neighbor))

    return [], len(closedlist), 0

# --- 4. EXÉCUTION ET AFFICHAGE ---
print(f"Lancement de Dijkstra de {start} vers {goal}...")
start_time = time.time()
path_result, num_nodes, cost = dijkstra(grid, start, goal)
execution_time = time.time() - start_time

print(f"Terminé en {execution_time:.4f} secondes !")
print(f"Coût : {cost} | Nœuds explorés : {num_nodes}")

# Affichage graphique
plt.figure(figsize=(10, 10))
plt.imshow(grille_brute, cmap='binary', origin='upper')

if path_result:
    path_x = [p[0] for p in path_result]
    path_y = [p[1] for p in path_result]
    plt.plot(path_x, path_y, color='blue', linewidth=2, label="Chemin Dijkstra")
else:
    print("⚠️ Aucun chemin trouvé !")

plt.scatter(start[0], start[1], color='green', marker='o', s=100, label="Départ")
plt.scatter(goal[0], goal[1], color='red', marker='x', s=100, label="Table 1")

plt.title(f"Pathfinding Dijkstra\nExplorés: {num_nodes} | Temps: {execution_time:.4f}s")
plt.legend()
plt.show()