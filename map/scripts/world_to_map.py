#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import yaml
import sys

def parse_pose(pose_str):
    if pose_str is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return [float(x) for x in pose_str.split()]

def convert_world_to_map(world_file, output_prefix, resolution=0.05):
    try:
        tree = ET.parse(world_file)
        root = tree.getroot()
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier XML/SDF : {e}")
        return

    # On définit une zone de taille fixe pour accueillir le restaurant global
    # (le monde fait environ 20x10, on prend une marge pour être tranquille)
    width_m = 25.0
    height_m = 15.0
    width_px = int(width_m / resolution)
    height_px = int(height_m / resolution)
    
    # 255 = vide (blanc), 0 = obstacle (noir)
    map_image = np.ones((height_px, width_px), dtype=np.uint8) * 255
    
    # On décale le 0,0 du world Gazebo pour qu'il soit à x=2, y=2 sur la grille
    offset_x = 2.0
    offset_y = 2.0
    
    def m_to_px(x, y):
        # Repère cartésien (Gazebo) -> Repère image (OpenCV où Y descend)
        px = int((x + offset_x) / resolution)
        py = height_px - int((y + offset_y) / resolution)
        return px, py

    print("Extraction des obstacles du fichier world...")
    
    # --- 1. Extraction des murs (boites) ---
    for model in root.findall('.//model'):
        for link in model.findall('link'):
            pose_elem = link.find('pose')
            pose = parse_pose(pose_elem.text if pose_elem is not None else None)
            x, y, yaw = pose[0], pose[1], pose[5]
            
            box = link.find('.//collision/geometry/box/size')
            if box is not None:
                size = [float(s) for s in box.text.split()]
                w, h = size[0], size[1]
                
                # Gestion de la rotation des murs au cas où
                corns = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
                c, s = np.cos(yaw), np.sin(yaw)
                R = np.array([[c, -s], [s, c]])
                
                pts = []
                for pt in corns:
                    rp = np.dot(R, pt)
                    pts.append(m_to_px(x + rp[0], y + rp[1]))
                
                cv2.fillPoly(map_image, [np.array(pts, np.int32)], 0)

    # --- 2. Extraction des tables ---
    for inc in root.findall('.//include'):
        uri = inc.find('uri')
        if uri is not None and ('cafe_table' in uri.text):
            pose_elem = inc.find('pose')
            pose = parse_pose(pose_elem.text if pose_elem is not None else None)
            px, py = m_to_px(pose[0], pose[1])
            # Une table de café fait environ ~40 cm de rayon
            radius_px = int(0.40 / resolution)
            cv2.circle(map_image, (px, py), radius_px, 0, -1)

    # Sauvegarde des fichiers
    pgm_file = f"{output_prefix}.pgm"
    yaml_file = f"{output_prefix}.yaml"
    cv2.imwrite(pgm_file, map_image)
    
    # Définition des métadonnées du SLAM
    origin_x = -offset_x
    origin_y = -height_m + offset_y + height_m # simpliste : origine Y du coin en bas à gauche
    origin_y = -offset_y
    
    yaml_dict = {
        'image': pgm_file.split('/')[-1], # Nom du fichier PGM (sans chemin)
        'resolution': resolution,
        'origin': [origin_x, origin_y, 0.0],
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196
    }
    
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_dict, f, default_flow_style=None)
        
    print(f"Mission accomplie ! Les fichiers SLAM pour le Pathfinding ont été générés : ")
    print(f"-> {pgm_file}")
    print(f"-> {yaml_file}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Utilisation: python3 world_to_map.py <fichier.world> <nom_de_sortie>")
        sys.exit(1)
    
    convert_world_to_map(sys.argv[1], sys.argv[2])
