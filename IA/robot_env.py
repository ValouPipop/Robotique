import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import yaml
import json
import random
import os
from PIL import Image
from scipy.ndimage import binary_dilation
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from collections import deque
import matplotlib.pyplot as plt

class FastRobot2DEnv(gym.Env):
    def __init__(self):
        super(FastRobot2DEnv, self).__init__()
        
        # --- 1. CHARGEMENT DE LA MAP ---
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
        
        self.grid = self.grille_epaisse 
        self.hauteur_image = self.grille_epaisse.shape[0]
        self.largeur_image = self.grille_epaisse.shape[1]

        # --- 2. ESPACES GYM ---
        self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([0.5, 1.0]), dtype=np.float32)
        self.num_lidar_rays = 24
        self.observation_space = spaces.Box(low=np.array([-np.inf]*(self.num_lidar_rays + 2)), 
                                            high=np.array([np.inf]*(self.num_lidar_rays + 2)), 
                                            dtype=np.float32)

        # --- 3. PARAMÈTRES DU ROBOT ---
        self.dt = 0.1 
        self.max_range = 3.5 
        self.min_range = 0.20 
        
        self.robot_x, self.robot_y, self.robot_theta = 0.0, 0.0, 0.0
        self.goal_x, self.goal_y = 0.0, 0.0 
        self.distance_map = None
        self.previous_real_dist = 0.0
        self.crash = False
        self.steps = 0
        self.last_w = 0.0

        # --- 4. PRÉ-CALCUL DES DISTANCE MAPS (une seule fois !) ---
        with open('../coordonées/destination.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.tables = data["tables"]
        self.start_pos = data["start_position"]
        
        print("Pré-calcul des cartes de distances pour les 5 tables...")
        self.all_distance_maps = {}
        for table in self.tables:
            tid = table["id"]
            gx = table["delivery_point"]["x"]
            gy = table["delivery_point"]["y"]
            self.all_distance_maps[tid] = self._calculate_distance_map(gx, gy)
            print(f"  -> {tid} OK")
        print("Pré-calcul terminé !")

    def _meters_to_grid(self, x_m, y_m):
        col_x = int(round((x_m - self.origin_x) / self.res))
        row_y = int(round((y_m - self.origin_y) / self.res))
        row_y_image = (self.hauteur_image - 1) - row_y
        return row_y_image, col_x

    def _calculate_distance_map(self, goal_x, goal_y):
        row_g, col_g = self._meters_to_grid(goal_x, goal_y)
        dist_map = np.full((self.hauteur_image, self.largeur_image), 9999.0, dtype=np.float32)
        row_g = max(0, min(self.hauteur_image - 1, row_g))
        col_g = max(0, min(self.largeur_image - 1, col_g))
        dist_map[row_g, col_g] = 0.0
        queue = deque([(row_g, col_g)])
        while queue:
            r, c = queue.popleft()
            current_dist = dist_map[r, c]
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.hauteur_image and 0 <= nc < self.largeur_image:
                    if self.grid[nr, nc] == 0: 
                        weight = 1.414 if abs(dr) + abs(dc) == 2 else 1.0
                        new_dist = current_dist + weight
                        if new_dist < dist_map[nr, nc]:
                            dist_map[nr, nc] = new_dist
                            queue.append((nr, nc))
        return dist_map

    def _simulate_lidar(self):
        ranges = []
        angles = np.linspace(-math.pi, math.pi, self.num_lidar_rays, endpoint=False)
        step_size = self.res / 2.0
        for angle in angles:
            ray_angle = self.robot_theta + angle
            dist = 0.0
            while dist < self.max_range:
                dist += step_size
                check_x = self.robot_x + dist * math.cos(ray_angle)
                check_y = self.robot_y + dist * math.sin(ray_angle)
                row, col = self._meters_to_grid(check_x, check_y)
                if row < 0 or row >= self.hauteur_image or col < 0 or col >= self.largeur_image or self.grid[row, col] == 1:
                    break
            ranges.append(dist)
        return ranges

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Position de départ
        self.robot_x = self.start_pos["x"]
        self.robot_y = self.start_pos["y"]
        self.robot_theta = self.start_pos.get("yaw", 0.0)
        # Choisir une table au hasard et récupérer sa distance map pré-calculée
        table_cible = random.choice(self.tables)
        self.goal_x = table_cible["delivery_point"]["x"]
        self.goal_y = table_cible["delivery_point"]["y"]
        self.distance_map = self.all_distance_maps[table_cible["id"]]
        row, col = self._meters_to_grid(self.robot_x, self.robot_y)
        self.previous_real_dist = self.distance_map[row, col] * self.res
        self.crash = False
        self.steps = 0
        self.last_w = 0.0
        return self._get_state(), {}

    def step(self, action):
        self.steps += 1
        v, w = float(action[0]), float(action[1])
        self.last_w = w  # On mémorise pour la pénalité de rotation
        self.robot_x += v * math.cos(self.robot_theta) * self.dt
        self.robot_y += v * math.sin(self.robot_theta) * self.dt
        self.robot_theta += w * self.dt
        self.robot_theta = (self.robot_theta + math.pi) % (2 * math.pi) - math.pi
        state = self._get_state()
        terminated = self._check_if_done()
        reward = self._compute_reward()
        truncated = self.steps >= 1500
        row, col = self._meters_to_grid(self.robot_x, self.robot_y)
        row = max(0, min(self.hauteur_image - 1, row))
        col = max(0, min(self.largeur_image - 1, col))
        self.previous_real_dist = self.distance_map[row, col] * self.res
        return state, reward, terminated, truncated, {}

    def _get_state(self):
        scan = self._simulate_lidar()
        row, col = self._meters_to_grid(self.robot_x, self.robot_y)
        row = max(0, min(self.hauteur_image - 1, row))
        col = max(0, min(self.largeur_image - 1, col))
        raw_dist = self.distance_map[row, col]
        real_dist = min(raw_dist, 2000.0) * self.res
        target_angle = math.atan2(self.goal_y - self.robot_y, self.goal_x - self.robot_x)
        heading = target_angle - self.robot_theta
        heading = (heading + math.pi) % (2 * math.pi) - math.pi
        state = scan + [real_dist, heading]
        return np.array(state, dtype=np.float32)

    def _check_if_done(self):
        dist_eucl = np.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
        if dist_eucl < 0.35:
            return True
        row, col = self._meters_to_grid(self.robot_x, self.robot_y)
        if row < 0 or row >= self.hauteur_image or col < 0 or col >= self.largeur_image or self.grid[row, col] == 1:
            self.crash = True
            return True
        return False

    def _compute_reward(self):
        dist_eucl = np.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
        if self.crash:
            return -500.0
        if dist_eucl < 0.35:
            return 2000.0
        row, col = self._meters_to_grid(self.robot_x, self.robot_y)
        row = max(0, min(self.hauteur_image - 1, row))
        col = max(0, min(self.largeur_image - 1, col))
        current_raw_dist = min(self.distance_map[row, col], 2000.0)
        current_real_dist = current_raw_dist * self.res
        prev_real_dist_saturated = min(self.previous_real_dist / self.res, 2000.0) * self.res
        # Progression vers l'objectif
        reward = (prev_real_dist_saturated - current_real_dist) * 50.0 
        # Pénalité de temps plus forte → chemins plus courts
        reward -= 0.5
        # Pénalité de rotation → empêche le zigzag
        reward -= abs(self.last_w) * 0.3
        return float(reward)

def plot_final_results(log_dir="./logs/"):
    eval_file = os.path.join(log_dir, "evaluations.npz")
    if os.path.exists(eval_file):
        data = np.load(eval_file)
        timesteps = data['timesteps']
        results = data['results']
        mean_rewards = np.mean(results, axis=1)
        plt.figure(figsize=(10, 5))
        plt.plot(timesteps, mean_rewards, label="Reward Moyenne")
        plt.title("Analyse de l'Entraînement de l'IA")
        plt.xlabel("Pas de temps (steps)")
        plt.ylabel("Récompense")
        plt.grid(True)
        plt.savefig("resultats_entrainement_final.png")
        print("Graphique de synthèse sauvegardé dans 'resultats_entrainement_final.png'")

if __name__ == '__main__':
    env = FastRobot2DEnv()
    
    # Dossiers de sauvegarde
    os.makedirs("./best_model/", exist_ok=True)
    os.makedirs("./checkpoints/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)

    # Callbacks
    eval_callback = EvalCallback(env, best_model_save_path="./best_model/",
                                 log_path="./logs/", eval_freq=10000,
                                 n_eval_episodes=5, deterministic=True)
    
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path="./checkpoints/",
                                             name_prefix="robot_model")

    print("Lancement de l'Entraînement Final (1.5M steps)...")
    
    model = PPO("MlpPolicy", env, verbose=1, 
                learning_rate=0.0001, 
                n_steps=4096, 
                batch_size=128,
                ent_coef=0.01,
                target_kl=0.015,
                device="cpu")
    
    try:
        model.learn(total_timesteps=1000000, callback=[eval_callback, checkpoint_callback])
    except KeyboardInterrupt:
        print("Entraînement interrompu par l'utilisateur.")
    
    # Toujours sauvegarder et tracer à la fin
    model.save("mon_robot_intelligent_complet")
    plot_final_results()
    print("Fini !")