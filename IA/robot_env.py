import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import yaml
import json
import random
from PIL import Image
from scipy.ndimage import binary_dilation
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class FastRobot2DEnv(gym.Env):
    def __init__(self):
        super(FastRobot2DEnv, self).__init__()
        
        # --- 1. CHARGEMENT DE LA MAP (Comme dans ton A*) ---
        with open('../map/map.yaml', 'r') as file:
            self.map_info = yaml.safe_load(file)
            
        img = Image.open('../map/' + self.map_info["image"])
        grid_pixels = np.array(img)
        grille_brute = np.where(grid_pixels < 250, 1, 0)
        
        self.res = self.map_info["resolution"]
        self.origin_x = self.map_info["origin"][0]
        self.origin_y = self.map_info["origin"][1]
        
        # Inflation des obstacles (20cm)
        rayon_pixels = int(np.ceil(0.20 / self.res))
        y_g, x_g = np.ogrid[-rayon_pixels:rayon_pixels+1, -rayon_pixels:rayon_pixels+1]
        masque = x_g**2 + y_g**2 <= rayon_pixels**2
        self.grille_epaisse = binary_dilation(grille_brute, structure=masque).astype(int)
        
        self.grid = self.grille_epaisse.T 
        self.hauteur_image = self.grille_epaisse.shape[0]
        self.largeur_image = self.grille_epaisse.shape[1]

        # --- 2. ESPACES GYM (Identique à Gazebo) ---
        # Action : [Vitesse linéaire, Vitesse angulaire]
        self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([0.5, 1.0]), dtype=np.float32)
        # Observation : 10 rayons LiDAR + distance + angle_relatif = 12 valeurs
        self.observation_space = spaces.Box(low=np.array([-np.inf]*12), high=np.array([np.inf]*12), dtype=np.float32)

        # --- 3. PARAMÈTRES DU ROBOT ---
        self.dt = 0.1 # On simule des pas de 0.1 seconde
        self.max_range = 3.5 # Portée max du LiDAR en mètres
        self.min_range = 0.20 # Distance de collision
        
        # Positions (On les réinitialisera dans reset())
        self.robot_x, self.robot_y, self.robot_theta = 0.0, 0.0, 0.0
        self.goal_x, self.goal_y = 2.0, 2.0 
        self.previous_distance = 0.0
        self.crash = False
        self.steps = 0

    def _meters_to_grid(self, x_m, y_m):
        """Convertit les mètres en indices de la matrice Numpy"""
        col_x = int(round((x_m - self.origin_x) / self.res))
        row_y = int(round((y_m - self.origin_y) / self.res))
        row_y_image = (self.hauteur_image - 1) - row_y
        return col_x, row_y_image

    def _simulate_lidar(self):
        """Simule 10 rayons laser mathématiquement (Raycasting)"""
        ranges = []
        # 10 rayons répartis sur 360 degrés
        angles = np.linspace(-math.pi, math.pi, 10, endpoint=False)
        
        step_size = self.res / 2.0 # On avance sur le rayon par demi-pixel pour être précis
        
        for angle in angles:
            ray_angle = self.robot_theta + angle
            dist = 0.0
            hit = False
            
            while dist < self.max_range:
                dist += step_size
                check_x = self.robot_x + dist * math.cos(ray_angle)
                check_y = self.robot_y + dist * math.sin(ray_angle)
                
                col, row = self._meters_to_grid(check_x, check_y)
                
                # Vérifie si le rayon sort de la carte ou tape un mur
                if col < 0 or col >= self.largeur_image or row < 0 or row >= self.hauteur_image:
                    hit = True
                    break
                if self.grid[col, row] == 1:
                    hit = True
                    break
                    
            ranges.append(dist)
        return ranges

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Charger les données de destination (Start et Tables)
        with open('../coordonées/destination.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.tables = data["tables"]
        start_pos = data["start_position"]

        # Position de départ
        self.robot_x = start_pos["x"]
        self.robot_y = start_pos["y"]
        self.robot_theta = start_pos.get("yaw", 0.0)
        
        # Choisir une table au hasard comme objectif pour généraliser l'IA
        table_cible = random.choice(self.tables)
        delivery = table_cible["delivery_point"]
        self.goal_x = delivery["x"]
        self.goal_y = delivery["y"]
        
        self.crash = False
        self.previous_distance = np.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
        self.steps = 0
        
        return self._get_state(), {}

    def step(self, action):
        self.steps += 1
        v = float(action[0])
        w = float(action[1])
        
        # --- LE MOTEUR PHYSIQUE MATHÉMATIQUE ---
        self.robot_x += v * math.cos(self.robot_theta) * self.dt
        self.robot_y += v * math.sin(self.robot_theta) * self.dt
        self.robot_theta += w * self.dt
        
        # Normalisation de l'angle
        while self.robot_theta > math.pi: self.robot_theta -= 2*math.pi
        while self.robot_theta < -math.pi: self.robot_theta += 2*math.pi
        
        state = self._get_state()
        terminated = self._check_if_done()
        reward = self._compute_reward()
        
        # Timeout pour éviter de tourner en rond éternellement (1000 steps = 100 sec)
        truncated = self.steps >= 1000
        
        self.previous_distance = np.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
        
        return state, reward, terminated, truncated, {}

    def _get_state(self):
        scan = self._simulate_lidar()
        
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        dist = np.hypot(dx, dy)
        
        target_angle = math.atan2(dy, dx)
        heading = target_angle - self.robot_theta
        
        # Normalisation heading
        while heading > math.pi: heading -= 2*math.pi
        while heading < -math.pi: heading += 2*math.pi
        
        state = scan + [dist, heading]
        return np.array(state, dtype=np.float32)

    def _check_if_done(self):
        dist = np.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
        if dist < 0.3: # Rayon d'arrivée
            return True # Gagné !
            
        # Collision : Si on est littéralement SUR un pixel noir (mur)
        col, row = self._meters_to_grid(self.robot_x, self.robot_y)
        if col < 0 or col >= self.largeur_image or row < 0 or row >= self.hauteur_image or self.grid[col, row] == 1:
            self.crash = True
            return True
            
        # Collision : Si le faux LiDAR détecte un mur trop près
        # On évite de recalculer le lidar ici pour la perf, on pourrait utiliser le dernier scan
        # Mais pour la cohérence, on va utiliser une min_range
        return False

    def _compute_reward(self):
        dist = np.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
        
        if self.crash:
            return -500.0 # Grosse pénalité pour le crash
        if dist < 0.3:
            return 2000.0 # Grosse récompense pour l'objectif
            
        # Récompense basée sur la progression vers l'objectif
        reward = (self.previous_distance - dist) * 100.0 
        
        # Petite pénalité constante pour encourager la rapidité
        reward -= 0.1
        
        return float(reward)

if __name__ == '__main__':
    env = FastRobot2DEnv()
    print("Vérification de l'environnement 2D...")
    check_env(env, warn=True)
    
    print("Lancement de l'entraînement Actor-Critic (PPO)...")
    # On utilise PPO qui est un algorithme Actor-Critic très robuste
    # On force l'utilisation du CPU pour éviter les erreurs CUDA sur certains systèmes
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, device="cpu")
    
    # Entraînement sur 300 000 pas pour bien généraliser toutes les tables
    model.learn(total_timesteps=300000)
    
    model.save("mon_robot_intelligent_tables")
    print("Entraînement terminé et modèle sauvegardé !")