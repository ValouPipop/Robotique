#!/usr/bin/env python3
"""
==========================================================================
 CONTRÔLEUR IA (PPO) POUR GAZEBO / ROS
==========================================================================
Charge le modèle PPO entraîné et pilote le TurtleBot3 dans Gazebo.
Utilise le vrai LiDAR ROS (/scan) et l'odométrie (/odom).

Usage :
  python3 controller_ia.py [table_id]
  python3 controller_ia.py table_1
  python3 controller_ia.py table_3
"""
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import tf
import numpy as np
import math
import sys
import os
import yaml
import json
from PIL import Image
from scipy.ndimage import binary_dilation
from collections import deque
from stable_baselines3 import PPO

# =====================================================================
# 1. CHARGEMENT DE LA MAP ET DISTANCE MAP
# =====================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

with open(os.path.join(project_dir, 'map/map.yaml'), 'r') as f:
    map_info = yaml.safe_load(f)

img = Image.open(os.path.join(project_dir, 'map', map_info["image"]))
grid_pixels = np.array(img)
grille_brute = np.where(grid_pixels < 250, 1, 0)

res = map_info["resolution"]
origin_x = map_info["origin"][0]
origin_y = map_info["origin"][1]

# Inflation des obstacles (même rayon que l'entraînement : 20cm)
rayon_pixels = int(np.ceil(0.20 / res))
y_g, x_g = np.ogrid[-rayon_pixels:rayon_pixels+1, -rayon_pixels:rayon_pixels+1]
masque = x_g**2 + y_g**2 <= rayon_pixels**2
grille_epaisse = binary_dilation(grille_brute, structure=masque).astype(int)

grid = grille_epaisse
hauteur_image = grille_epaisse.shape[0]
largeur_image = grille_epaisse.shape[1]

# Charger les tables
with open(os.path.join(project_dir, 'coordonées/destination.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)
tables = data["tables"]

def meters_to_grid(x_m, y_m):
    col_x = int(round((x_m - origin_x) / res))
    row_y = int(round((y_m - origin_y) / res))
    row_y_image = (hauteur_image - 1) - row_y
    return row_y_image, col_x

def calculate_distance_map(goal_x, goal_y):
    """BFS/Dijkstra flood-fill depuis le goal (identique à robot_env.py)."""
    row_g, col_g = meters_to_grid(goal_x, goal_y)
    dist_map = np.full((hauteur_image, largeur_image), 9999.0, dtype=np.float32)
    row_g = max(0, min(hauteur_image - 1, row_g))
    col_g = max(0, min(largeur_image - 1, col_g))
    dist_map[row_g, col_g] = 0.0
    queue = deque([(row_g, col_g)])
    while queue:
        r, c = queue.popleft()
        current_dist = dist_map[r, c]
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < hauteur_image and 0 <= nc < largeur_image:
                if grid[nr, nc] == 0:
                    weight = 1.414 if abs(dr) + abs(dc) == 2 else 1.0
                    new_dist = current_dist + weight
                    if new_dist < dist_map[nr, nc]:
                        dist_map[nr, nc] = new_dist
                        queue.append((nr, nc))
    return dist_map

# =====================================================================
# 2. NOEUD ROS : CONTRÔLEUR IA
# =====================================================================
class IAController:
    def __init__(self, table_id):
        rospy.init_node('ia_controller_node', anonymous=True)
        
        # --- Trouver la table cible ---
        table_cible = next((t for t in tables if t["id"] == table_id), None)
        if table_cible is None:
            rospy.logerr(f"Table '{table_id}' introuvable ! Tables disponibles : {[t['id'] for t in tables]}")
            sys.exit(1)
        
        self.goal_x = table_cible["delivery_point"]["x"]
        self.goal_y = table_cible["delivery_point"]["y"]
        rospy.loginfo(f"Table cible : {table_id} → ({self.goal_x}, {self.goal_y})")
        
        # --- Calculer la distance map pour cette table ---
        rospy.loginfo("Calcul de la distance map...")
        self.distance_map = calculate_distance_map(self.goal_x, self.goal_y)
        rospy.loginfo("Distance map prête !")
        
        # --- Charger le modèle IA ---
        model_path = os.path.join(script_dir, "best_model.zip")
        rospy.loginfo(f"Chargement du modèle IA : {model_path}")
        self.model = PPO.load(model_path, device="cpu")
        rospy.loginfo("Modèle chargé !")
        
        # --- ROS Pub/Sub ---
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        # --- État du robot ---
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.scan_ranges = None
        self.scan_angle_min = 0.0
        self.scan_angle_increment = 0.0
        self.scan_n_rays = 0
        
        # --- Paramètres (identiques à l'entraînement) ---
        self.num_lidar_rays = 24  # L'IA attend 24 rayons
        self.max_range = 3.5
        # Les 24 angles utilisés pendant l'entraînement (relatifs au robot)
        self.training_angles = np.linspace(-math.pi, math.pi, self.num_lidar_rays, endpoint=False)
        
        # --- Fréquence de contrôle ---
        self.rate = rospy.Rate(10)  # 10 Hz = dt=0.1s comme l'entraînement
        
        rospy.loginfo("En attente de /odom et /scan...")
        try:
            rospy.wait_for_message('/odom', Odometry, timeout=5)
            scan_msg = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            rospy.loginfo(f"LiDAR reçu : {len(scan_msg.ranges)} rayons, "
                          f"angle_min={math.degrees(scan_msg.angle_min):.1f}°, "
                          f"angle_max={math.degrees(scan_msg.angle_max):.1f}°, "
                          f"increment={math.degrees(scan_msg.angle_increment):.2f}°")
            rospy.loginfo("Données reçues ! Démarrage de l'IA...")
        except rospy.ROSException:
            rospy.logerr("Timeout ! Vérifiez que Gazebo est lancé avec le robot.")
            sys.exit(1)
    
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.theta = euler[2]
    
    def scan_callback(self, msg):
        self.scan_ranges = np.array(msg.ranges, dtype=np.float32)
        self.scan_angle_min = msg.angle_min
        self.scan_angle_increment = msg.angle_increment
        self.scan_n_rays = len(msg.ranges)
    
    def build_observation(self):
        """
        Construit le vecteur d'observation (26 valeurs) identique à celui
        utilisé pendant l'entraînement dans FastRobot2DEnv._get_state().
        
        Format : [24 rayons LiDAR] + [distance_dijkstra] + [heading]
        
        MAPPING CRITIQUE :
        - L'entraînement utilise 24 rayons à angles = linspace(-π, +π, 24)
          Ces angles sont RELATIFS au robot (0 = devant).
        - Le vrai LiDAR TurtleBot3 a 360 rayons avec angle_min et angle_increment.
          On doit retrouver l'index du vrai rayon pour chaque angle d'entraînement.
        """
        raw = self.scan_ranges.copy()
        # Remplacer nan/inf par max_range
        raw = np.where(np.isnan(raw) | np.isinf(raw), self.max_range, raw)
        raw = np.clip(raw, 0.0, self.max_range)
        
        # --- Mapper les 24 angles d'entraînement vers les indices du vrai LiDAR ---
        lidar_24 = []
        for train_angle in self.training_angles:
            # train_angle est relatif au robot (0 = devant, -π = derrière)
            # Le vrai LiDAR aussi est relatif au robot, mais indexé différemment
            # Index dans le vrai scan = (angle - angle_min) / angle_increment
            
            # Normaliser l'angle dans la plage du LiDAR
            scan_angle = train_angle
            # Ramener dans [angle_min, angle_min + 2π)
            while scan_angle < self.scan_angle_min:
                scan_angle += 2 * math.pi
            while scan_angle >= self.scan_angle_min + 2 * math.pi:
                scan_angle -= 2 * math.pi
            
            idx = int(round((scan_angle - self.scan_angle_min) / self.scan_angle_increment))
            idx = idx % self.scan_n_rays  # Sécurité wrapping
            
            lidar_24.append(float(raw[idx]))
        
        # --- 2. Distance Dijkstra (depuis la distance map) ---
        row, col = meters_to_grid(self.x, self.y)
        row = max(0, min(hauteur_image - 1, row))
        col = max(0, min(largeur_image - 1, col))
        raw_dist = self.distance_map[row, col]
        real_dist = min(raw_dist, 2000.0) * res
        
        # --- 3. Heading (angle relatif vers le goal) ---
        target_angle = math.atan2(self.goal_y - self.y, self.goal_x - self.x)
        heading = target_angle - self.theta
        heading = (heading + math.pi) % (2 * math.pi) - math.pi
        
        # --- 4. Assembler le vecteur (26 valeurs) ---
        obs = lidar_24 + [real_dist, heading]
        return np.array(obs, dtype=np.float32)
    
    def run(self):
        rospy.loginfo(f"=== IA EN ROUTE VERS ({self.goal_x}, {self.goal_y}) ===")
        step_count = 0
        
        while not rospy.is_shutdown():
            if self.scan_ranges is None:
                self.rate.sleep()
                continue
            
            # Distance euclidienne au goal
            dist_to_goal = np.hypot(self.x - self.goal_x, self.y - self.goal_y)
            
            # Arrivé ? (même seuil que l'entraînement : 0.35m)
            if dist_to_goal < 0.40:
                rospy.loginfo(f"🎉 ARRIVÉ ! Distance finale : {dist_to_goal:.2f}m en {step_count} steps.")
                self.stop_robot()
                break
            
            # Sécurité : timeout
            if step_count > 3000:
                rospy.logwarn(f"⚠️ Timeout ({step_count} steps). Robot arrêté.")
                self.stop_robot()
                break
            
            # Construire l'observation et prédire l'action
            obs = self.build_observation()
            action, _ = self.model.predict(obs, deterministic=True)
            
            v = float(np.clip(action[0], 0.0, 0.5))
            w = float(np.clip(action[1], -1.0, 1.0))
            
            # Publier l'action
            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = w
            self.cmd_vel_pub.publish(cmd)
            
            step_count += 1
            if step_count % 50 == 0:
                # Debug : afficher les premières valeurs LiDAR + distance + heading
                rospy.loginfo(
                    f"Step {step_count} | Pos: ({self.x:.1f}, {self.y:.1f}) θ={math.degrees(self.theta):.0f}° | "
                    f"Dist: {dist_to_goal:.2f}m | v={v:.2f} w={w:.2f} | "
                    f"obs[dist]={obs[-2]:.1f} obs[heading]={math.degrees(obs[-1]):.0f}° | "
                    f"lidar_front={obs[12]:.2f}m lidar_back={obs[0]:.2f}m"
                )
            
            self.rate.sleep()
    
    def stop_robot(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        rospy.sleep(0.5)
        self.cmd_vel_pub.publish(cmd)

# =====================================================================
# 3. MAIN
# =====================================================================
if __name__ == '__main__':
    try:
        table_id = sys.argv[1] if len(sys.argv) > 1 else 'table_1'
        
        print("=" * 60)
        print(f"  CONTRÔLEUR IA (PPO) - Navigation vers {table_id}")
        print("=" * 60)
        
        controller = IAController(table_id)
        controller.run()
        
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Erreur : {e}")
        import traceback
        traceback.print_exc()
