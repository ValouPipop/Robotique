#!/usr/bin/env python3
"""
==========================================================================
 CONTRÔLEUR IA (PPO) POUR GAZEBO / ROS
==========================================================================
Charge le modèle PPO entraîné et pilote le TurtleBot3 dans Gazebo.
Utilise le vrai LiDAR ROS (/scan) et l'odométrie (/odom).

Quand le robot se bloque (mur, etc.), il est reset à sa position de
départ via le service Gazebo, comme pendant l'entraînement (crash = reset).

Usage :
  python3 controller_ia.py [table_id]
  python3 controller_ia.py table_1
  python3 controller_ia.py table_3
"""
import rospy
from geometry_msgs.msg import Twist, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
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

# Charger les tables et position de départ
with open(os.path.join(project_dir, 'coordonées/destination.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)
tables = data["tables"]
start_pos = data["start_position"]

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
        
        # --- Position de départ (identique à l'entraînement) ---
        self.start_x = start_pos["x"]
        self.start_y = start_pos["y"]
        self.start_yaw = start_pos.get("yaw", 0.0)
        
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
        
        # --- Service Gazebo pour reset le robot ---
        rospy.loginfo("Connexion au service Gazebo set_model_state...")
        rospy.wait_for_service('/gazebo/set_model_state', timeout=10)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        rospy.loginfo("Service Gazebo connecté !")
        
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
        # Index 0 = -π (arrière), index 6 = -π/2 (droite),
        # Index 12 = 0 (avant), index 18 = π/2 (gauche)
        self.training_angles = np.linspace(-math.pi, math.pi, self.num_lidar_rays, endpoint=False)
        
        # --- Détection de blocage ---
        self.stuck_threshold = 30    # Steps sans mouvement → reset (~3s)
        
        # --- Sécurité : distance d'évitement ---
        self.safety_dist = 0.30
        self.critical_dist = 0.20
        
        # --- Fréquence de contrôle ---
        self.rate = rospy.Rate(10)  # 10 Hz = dt=0.1s comme l'entraînement
        
        # ============================================================
        # ATTENTE DE GAZEBO : données stables avant de démarrer
        # ============================================================
        self._wait_for_stable_data()
    
    def _wait_for_stable_data(self):
        """Attend que Gazebo soit prêt avec des données LiDAR stables."""
        rospy.loginfo("⏳ Attente de Gazebo (odom + LiDAR stables)...")
        
        # 1. Attendre le premier message odom
        try:
            rospy.wait_for_message('/odom', Odometry, timeout=10)
        except rospy.ROSException:
            rospy.logerr("❌ Timeout /odom ! Vérifiez que Gazebo est lancé.")
            sys.exit(1)
        
        # 2. Attendre plusieurs scans LiDAR stables (pas de valeurs aberrantes)
        stable_scans = 0
        required_stable = 5  # 5 scans stables consécutifs
        
        while stable_scans < required_stable and not rospy.is_shutdown():
            try:
                scan_msg = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except rospy.ROSException:
                rospy.logwarn("⏳ Pas de scan LiDAR, on attend...")
                continue
            
            raw = np.array(scan_msg.ranges, dtype=np.float32)
            n_valid = np.sum((raw > 0.01) & np.isfinite(raw))
            pct_valid = n_valid / len(raw) * 100
            
            if pct_valid > 80:  # Au moins 80% de rayons valides
                stable_scans += 1
            else:
                stable_scans = 0
                rospy.loginfo(f"⏳ LiDAR pas encore stable ({pct_valid:.0f}% valides), on attend...")
            
            rospy.sleep(0.1)
        
        # Afficher les infos du dernier scan
        scan_msg = rospy.wait_for_message('/scan', LaserScan, timeout=5)
        raw = np.array(scan_msg.ranges, dtype=np.float32)
        n_zeros = np.sum(raw == 0.0)
        n_nan = np.sum(np.isnan(raw))
        n_inf = np.sum(np.isinf(raw))
        rospy.loginfo(f"✅ LiDAR stable : {len(scan_msg.ranges)} rayons, "
                      f"angle_min={math.degrees(scan_msg.angle_min):.1f}°, "
                      f"angle_max={math.degrees(scan_msg.angle_max):.1f}°, "
                      f"increment={math.degrees(scan_msg.angle_increment):.2f}°")
        rospy.loginfo(f"   Diagnostic : {n_zeros} zéros, {n_nan} NaN, {n_inf} inf "
                      f"sur {len(scan_msg.ranges)} rayons")
        rospy.loginfo("✅ Gazebo prêt ! Démarrage de l'IA...")
    
    def reset_robot_gazebo(self):
        """Téléporte le robot à sa position de départ via Gazebo."""
        self.stop_robot()
        rospy.sleep(0.3)
        
        # Convertir yaw en quaternion
        q = tf.transformations.quaternion_from_euler(0, 0, self.start_yaw)
        
        model_state = ModelState()
        model_state.model_name = 'turtlebot3_burger'
        model_state.pose.position.x = self.start_x
        model_state.pose.position.y = self.start_y
        model_state.pose.position.z = 0.0
        model_state.pose.orientation = Quaternion(*q)
        model_state.reference_frame = 'world'
        
        try:
            resp = self.set_model_state(model_state)
            if resp.success:
                rospy.loginfo(f"🔄 Robot reset à ({self.start_x}, {self.start_y}, θ={math.degrees(self.start_yaw):.0f}°)")
            else:
                rospy.logwarn(f"⚠️ Reset échoué : {resp.status_message}")
        except rospy.ServiceException as e:
            rospy.logerr(f"❌ Service Gazebo indisponible : {e}")
        
        # Attendre que les données se stabilisent après le reset
        rospy.sleep(1.0)
        # Vider le scan actuel pour forcer la réception de nouvelles données
        self.scan_ranges = None
        while self.scan_ranges is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
    
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
        """
        raw = self.scan_ranges.copy()
        # Remplacer nan/inf/0.0 par max_range
        # Le LiDAR réel et Gazebo renvoient 0.0 quand un rayon est hors portée
        raw = np.where(np.isnan(raw) | np.isinf(raw) | (raw < 0.01), self.max_range, raw)
        raw = np.clip(raw, 0.0, self.max_range)
        
        # --- Mapper les 24 angles d'entraînement vers les indices du vrai LiDAR ---
        lidar_24 = []
        for train_angle in self.training_angles:
            scan_angle = train_angle
            while scan_angle < self.scan_angle_min:
                scan_angle += 2 * math.pi
            while scan_angle >= self.scan_angle_min + 2 * math.pi:
                scan_angle -= 2 * math.pi
            
            idx = int(round((scan_angle - self.scan_angle_min) / self.scan_angle_increment))
            idx = idx % self.scan_n_rays
            
            lidar_24.append(float(raw[idx]))
        
        # --- 2. Distance Dijkstra (depuis la distance map) ---
        row, col = meters_to_grid(self.x, self.y)
        row = max(0, min(hauteur_image - 1, row))
        col = max(0, min(largeur_image - 1, col))
        raw_dist = self.distance_map[row, col]
        # Cap à 1000 cellules (50m) : pendant l'entraînement, le robot
        # ne se retrouve JAMAIS en zone obstacle (raw_dist ≈ 9999)
        real_dist = min(raw_dist, 1000.0) * res
        
        # --- 3. Heading (angle relatif vers le goal) ---
        target_angle = math.atan2(self.goal_y - self.y, self.goal_x - self.x)
        heading = target_angle - self.theta
        heading = (heading + math.pi) % (2 * math.pi) - math.pi
        
        # --- 4. Assembler le vecteur (26 valeurs) ---
        obs = lidar_24 + [real_dist, heading]
        return np.array(obs, dtype=np.float32)
    
    def run(self):
        rospy.loginfo(f"=== IA EN ROUTE VERS ({self.goal_x}, {self.goal_y}) ===")
        
        max_attempts = 5
        
        for attempt in range(1, max_attempts + 1):
            if rospy.is_shutdown():
                break
            
            if attempt > 1:
                rospy.logwarn(f"🔄 Tentative {attempt}/{max_attempts} — Reset du robot...")
                self.reset_robot_gazebo()
            
            success = self._navigate()
            
            if success:
                break
            elif attempt < max_attempts:
                rospy.logwarn(f"❌ Tentative {attempt} échouée.")
            else:
                rospy.logerr(f"❌ Échec après {max_attempts} tentatives.")
    
    def _navigate(self):
        """
        Boucle de navigation principale.
        Retourne True si arrivé, False si bloqué/timeout.
        """
        step_count = 0
        stuck_counter = 0
        last_pos = (self.x, self.y)
        
        while not rospy.is_shutdown():
            if self.scan_ranges is None:
                self.rate.sleep()
                continue
            
            # Distance euclidienne au goal
            dist_to_goal = np.hypot(self.x - self.goal_x, self.y - self.goal_y)
            
            # Arrivé ?
            if dist_to_goal < 0.40:
                rospy.loginfo(f"🎉 ARRIVÉ ! Distance finale : {dist_to_goal:.2f}m en {step_count} steps.")
                self.stop_robot()
                return True
            
            # Timeout
            if step_count > 3000:
                rospy.logwarn(f"⚠️ Timeout ({step_count} steps).")
                self.stop_robot()
                return False
            
            # Construire l'observation et prédire l'action
            obs = self.build_observation()
            action, _ = self.model.predict(obs, deterministic=True)
            
            v = float(np.clip(action[0], 0.0, 0.5))
            w = float(np.clip(action[1], -1.0, 1.0))
            
            # --- Détection de blocage → reset ---
            current_pos = (self.x, self.y)
            dist_moved = np.hypot(current_pos[0] - last_pos[0],
                                  current_pos[1] - last_pos[1])
            if dist_moved < 0.005:
                stuck_counter += 1
            else:
                stuck_counter = 0
            last_pos = current_pos
            
            if stuck_counter >= self.stuck_threshold:
                rospy.logwarn(f"⚠️ Robot bloqué à ({self.x:.1f}, {self.y:.1f}) depuis {stuck_counter} steps → RESET")
                self.stop_robot()
                return False  # Sortir pour reset
            
            # --- Sécurité minimale : évitement front ---
            front_indices = [10, 11, 12, 13, 14]
            front_min = min(obs[i] for i in front_indices)
            
            if front_min < self.critical_dist:
                v = 0.0
                left_space = float(np.mean([obs[i] for i in range(15, 21)]))
                right_space = float(np.mean([obs[i] for i in range(3, 9)]))
                w = 1.0 if left_space > right_space else -1.0
            elif front_min < self.safety_dist:
                v = min(v, 0.1)
            
            # Publier l'action
            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = w
            self.cmd_vel_pub.publish(cmd)
            
            step_count += 1
            if step_count % 50 == 0:
                mode = "SECU" if front_min < self.safety_dist else "IA"
                rospy.loginfo(
                    f"Step {step_count} [{mode}] | Pos: ({self.x:.1f}, {self.y:.1f}) θ={math.degrees(self.theta):.0f}° | "
                    f"Dist: {dist_to_goal:.2f}m | v={v:.2f} w={w:.2f} | "
                    f"obs[dist]={obs[-2]:.1f} obs[heading]={math.degrees(obs[-1]):.0f}° | "
                    f"front={front_min:.2f}m | stuck={stuck_counter}"
                )
            
            self.rate.sleep()
        
        return False
    
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
