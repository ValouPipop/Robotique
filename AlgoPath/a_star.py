import numpy as np
import matplotlib.pyplot as plt
import time
import heapq
import json
import yaml
from PIL import Image




with open('../coordonées/destination.json','r', encoding='utf-8') as f:
    data = json.load(f)

x_start = data["start_position"]["x"]
y_start = data["start_position"]["y"]

positions_tables = {}

for table in data["tables"]:
    id_table = table["id"]
    nom_x = id_table + "_x"
    nom_y = id_table + "_y"
    
    positions_tables[nom_x] = table["delivery_point"]["x"]
    positions_tables[nom_y] = table["delivery_point"]["y"]

# --- CHARGEMENT DE LA GRILLE PGM EN ARRAY NUMPY ---
with open('../map/map.yaml', 'r') as file:
    map_info = yaml.safe_load(file)

img = Image.open('../map/' + map_info["image"])
grid_pixels = np.array(img)
grille_brute = np.where(grid_pixels < 250, 1, 0) # 0 = Libre, 1 = Obstacle

# On TRANSPOSE la grille pour A* ! 
# (Ainsi grid[x, y] est la colonne X et ligne Y de l'image)
grid = grille_brute.T

origin_x = map_info["origin"][0]
origin_y = map_info["origin"][1]
res = map_info["resolution"]
hauteur_image = grille_brute.shape[0]

def meters_to_grid(x_meters, y_meters):
    col_x = int(round((x_meters - origin_x) / res))
    row_y = int(round((y_meters - origin_y) / res))
    row_y_image = (hauteur_image - 1) - row_y
    return (col_x, row_y_image)

start = meters_to_grid(x_start, y_start)
goal = meters_to_grid(positions_tables["table_1_x"], positions_tables["table_1_y"])


def heuristic_manhattan(a,b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def Neighbors(node):
    x,y = node
    neighbors = []
    width, height = grid.shape
    for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        nx,ny = x+dx,y+dy
        if 0 <= nx < width and 0 <= ny < height and grid[nx,ny] == 0:
            neighbors.append((nx,ny))
    return neighbors

def a_star(grid,start,goal):
    openlist = []
    closedlist = set()
    g = {}
    parent = {}
    path = []

    g[start] = 0
    heapq.heappush(openlist, (heuristic_manhattan(start, goal), heuristic_manhattan(start, goal), start))

    while openlist:
        _, _, current = heapq.heappop(openlist)

        if current in closedlist:
            continue

        if current == goal:
            cost = g[current]
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            path.reverse() 
            return path, len(closedlist), cost
            
        closedlist.add(current)

        neighbors = Neighbors(current)
        for neighbor in neighbors:
            if neighbor in closedlist:
                continue
                
            tentative_g = g[current] + 1
            
            if tentative_g < g.get(neighbor, float('inf')):
                parent[neighbor] = current
                g[neighbor] = tentative_g
                f_score = tentative_g + heuristic_manhattan(neighbor, goal)
                h_score = heuristic_manhattan(neighbor, goal)
                heapq.heappush(openlist, (f_score, h_score, neighbor))

    return [], len(closedlist), 0

# --- EXECUTION DE L'ALGO ---
print(f"Lancement de A* de {start} vers {goal}...")
start_time = time.time()
path_result, num_nodes, cost = a_star(grid, start, goal)
print(f"Terminé en {time.time() - start_time:.4f} secondes !")
print(f"Coût : {cost} | Nœuds explorés : {num_nodes}")

# --- AFFICHAGE DU CHEMIN ---
plt.imshow(grille_brute, cmap='binary', origin='upper')

if path_result:
    # Récupération de tous les X et Y pour le tracé
    path_x = [p[0] for p in path_result]
    path_y = [p[1] for p in path_result]
    plt.plot(path_x, path_y, color='red', linewidth=2, label="Chemin A*")
else:
    print("⚠️ Aucun chemin trouvé !")

plt.scatter(start[0], start[1], color='green', marker='o', s=100, label="Départ", zorder=5)
plt.scatter(goal[0], goal[1], color='blue', marker='x', s=100, label="Arrivée (Table 1)", zorder=5)

plt.title("Pathfinding A*")
plt.legend()
plt.show()
