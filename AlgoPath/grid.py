import yaml
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt # <-- Le nouvel import

def load_map_to_grid(yaml_path):
    with open(yaml_path, 'r') as file:
        map_info = yaml.safe_load(file)
    
    image_filename = map_info['image']
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    full_image_path = os.path.join(yaml_dir, image_filename)
    
    img = Image.open(full_image_path)
    grid_pixels = np.array(img)
    
    # 0 = Libre, 1 = Obstacle
    pathfinding_grid = np.where(grid_pixels < 250, 1, 0)
    
    return pathfinding_grid, map_info

def meters_to_pixels(x_meters, y_meters, origin_x, origin_y, res, hauteur_image):
    # 1. on décale de "origin" 
    # 2. on divise par la résolution pour avoir le nombre de pixels
    col_x = int(round((x_meters - origin_x) / res))
    
    # 3. Y_ROS monte vers le HAUT, Y_IMAGE descend vers le BAS !
    # Du coup on doit l'inverser par rapport à la hauteur de l'image.
    row_y = int(round((y_meters - origin_y) / res))
    row_y_image = (hauteur_image - 1) - row_y
    
    return col_x, row_y_image
if __name__ == "__main__":
    import json

    # --- Utilisation ---
    yaml_file = "../map/map.yaml" # Ton fichier YAML
    grille, infos = load_map_to_grid(yaml_file)

    # --- AFFICHAGE DE LA GRILLE ---
    print("Génération de l'affichage...")

    plt.imshow(grille, cmap='binary', origin='upper')

    origin_x = infos["origin"][0]
    origin_y = infos["origin"][1]
    res = infos["resolution"]
    hauteur_image = grille.shape[0]

    # On va tester de dessiner nos points !
    with open('../coordonées/destination.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Départ
    start_x, start_y = meters_to_pixels(data["start_position"]["x"], data["start_position"]["y"], origin_x, origin_y, res, hauteur_image)
    plt.scatter(start_x, start_y, color='green', marker='o', s=150, label="Start")

    # Tables
    for tbl in data["tables"]:
        tx, ty = meters_to_pixels(tbl["delivery_point"]["x"], tbl["delivery_point"]["y"], origin_x, origin_y, res, hauteur_image)
        plt.scatter(tx, ty, color='blue', marker='x', s=100)

    plt.title("Grille & Coordonnées (Destinations/Depart)")
    plt.legend()

    # On sauvegarde l'image dans le dossier map (en haute définition avec dpi=300)
    output_path = "../map/map_pathfinding.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Image sauvegardée sous : {output_path}")

    plt.show()