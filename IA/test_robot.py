import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from robot_env import FastRobot2DEnv
from PIL import Image
import yaml

def test_model():
    # 1. Charger l'environnement
    env = FastRobot2DEnv()
    
    # 2. Charger le modèle entraîné
    model_path = "best_model/best_model.zip"
    print(f"Chargement du modèle : {model_path}")
    model = PPO.load(model_path, env=env)

    # 3. Charger la map pour la visualisation
    with open('../map/map.yaml', 'r') as file:
        map_info = yaml.safe_load(file)
    img = Image.open('../map/' + map_info["image"])
    map_data = np.array(img)
    
    # Paramètres de conversion
    res = map_info["resolution"]
    origin_x = map_info["origin"][0]
    origin_y = map_info["origin"][1]
    hauteur_image = map_data.shape[0]

    def meters_to_pixel(x_m, y_m):
        col_x = int(round((x_m - origin_x) / res))
        row_y = int(round((y_m - origin_y) / res))
        row_y_image = (hauteur_image - 1) - row_y
        return col_x, row_y_image

    # 4. Faire quelques tests
    num_tests = 3
    plt.figure(figsize=(10, 10))
    plt.imshow(map_data, cmap='gray')
    
    colors = ['r', 'g', 'b', 'y', 'm']

    for i in range(num_tests):
        print(f"\n--- Test {i+1} ---")
        obs, info = env.reset()
        done = False
        truncated = False
        path_x, path_y = [], []
        
        # Enregistrer le point de départ et d'arrivée
        start_px, start_py = meters_to_pixel(env.robot_x, env.robot_y)
        goal_px, goal_py = meters_to_pixel(env.goal_x, env.goal_y)
        
        plt.plot(start_px, start_py, 'go', markersize=10, label='Départ' if i==0 else "")
        plt.plot(goal_px, goal_py, 'x', color=colors[i % len(colors)], markersize=12, markeredgewidth=3, label=f'Table {i+1}')

        step_count = 0
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            px, py = meters_to_pixel(env.robot_x, env.robot_y)
            path_x.append(px)
            path_y.append(py)
            step_count += 1
            
            if step_count > 2000: break # Sécurité

        plt.plot(path_x, path_y, color=colors[i % len(colors)], linewidth=2)
        
        if env.crash:
            print("Résultat : CRASH s'est produit.")
        else:
            dist_final = np.hypot(env.robot_x - env.goal_x, env.robot_y - env.goal_y)
            if dist_final < 0.4:
                print(f"Résultat : SUCCÈS ! Arrivé à {dist_final:.2f}m de la table.")
            else:
                print(f"Résultat : ÉCHEC (Temps écoulé ou bloqué). Distance finale : {dist_final:.2f}m")

    plt.legend()
    plt.title("Trajectoires de l'IA vers les tables")
    save_path = "resultat_ia_test.png"
    plt.savefig(save_path)
    print(f"\nVisualisation sauvegardée dans : {save_path}")
    plt.show()

if __name__ == '__main__':
    test_model()
