#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import tf
import numpy as np
import sys

# --- 1. SELECTION DE L'ALGORITHME ---
algo_name = sys.argv[1] if len(sys.argv) > 1 else 'a_star'

if algo_name == 'a_star':
    from a_star import a_star as find_path, grid, start, goal, res, origin_x, origin_y, hauteur_image
    planner = lambda: find_path(grid, start, goal)
elif algo_name == 'greedy':
    from greedy import greedy as find_path, grid, start, goal, res, origin_x, origin_y, hauteur_image, heuristic_manhattan
    planner = lambda: find_path(grid, start, goal, heuristic_manhattan)
elif algo_name == 'dijkstra':
    from djikstra import dijkstra as find_path, grid, start, goal, res, origin_x, origin_y, hauteur_image
    planner = lambda: find_path(grid, start, goal)
else:
    print(f"Algorithme inconnu: {algo_name}. Utilisation de a_star par défaut.")
    from a_star import a_star as find_path, grid, start, goal, res, origin_x, origin_y, hauteur_image
    planner = lambda: find_path(grid, start, goal)

# --- 2. CONVERSION PIXELS -> MÈTRES ---
def grid_to_meters(col_x, row_y_image):
    """Convertit le chemin A* (pixels) en coordonnées Gazebo (mètres)."""
    # row_y_image est l'index dans le tableau (0 en haut)
    # On doit repasser en coordonnées cartésiennes (Y=0 en bas)
    row_y = (hauteur_image - 1) - row_y_image
    x_meters = (col_x * res) + origin_x
    y_meters = (row_y * res) + origin_y
    return (x_meters, y_meters)

# --- 3. FONCTIONS DWA ---
def simulate(x, y, theta, v, w, dt=0.2):
    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + w * dt
    return x_new, y_new, theta_new

def distance_to_path(x, y, path):
    if not path: return 0.0
    return min(np.hypot(x - p[0], y - p[1]) for p in path)

def dwa_step(x, y, theta, v_current, w_current, path):
    dt = 0.2
    horizon = 5
    v_max = 0.5  # Réduit un peu pour la sécurité
    w_max = 1.0
    a_v = 0.5
    a_w = 1.0

    v_min_dw = max(0, v_current - a_v * dt)
    v_max_dw = min(v_max, v_current + a_v * dt)
    w_min_dw = max(-w_max, w_current - a_w * dt)
    w_max_dw = min(w_max, w_current + a_w * dt)

    v_samples = np.linspace(v_min_dw, v_max_dw, 5)
    w_samples = np.linspace(w_min_dw, w_max_dw, 7)

    best_cost = float('inf')
    best_u = (0.0, 0.0)

    for v in v_samples:
        for w in w_samples:
            xn, yn, thetan = x, y, theta
            
            # Simulation sur l'horizon
            for _ in range(horizon):
                xn, yn, thetan = simulate(xn, yn, thetan, v, w, dt)

            dist = distance_to_path(xn, yn, path)

            # Trouver le point local sur le chemin
            closest_idx = 0
            min_dist = float('inf')
            for i_p, p in enumerate(path):
                d = np.hypot(x - p[0], y - p[1])
                if d < min_dist:
                    min_dist = d
                    closest_idx = i_p
            
            # On vise un point un peu plus loin sur le chemin
            target_idx = min(len(path) - 1, closest_idx + 5)
            goal_pt = path[target_idx]

            # Calcul du cap (heading) vers le point cible
            goal_theta = np.arctan2(goal_pt[1] - yn, goal_pt[0] - xn)
            diff = goal_theta - thetan
            heading = abs(np.arctan2(np.sin(diff), np.cos(diff)))

            # Coûts : 
            # 1. Distance au chemin (doit rester proche)
            # 2. Orientation vers la cible
            # 3. Vitesse (doit avancer)
            velocity_cost = (v_max - v)
            
            J = 2.0 * dist + 1.0 * heading + 0.5 * velocity_cost

            if J < best_cost:
                best_cost = J
                best_u = (v, w)

    return best_u

# --- 4. NOEUD ROS ---
class DWARobotController:
    def __init__(self, path):
        rospy.init_node('dwa_controller_node', anonymous=True)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.rate = rospy.Rate(5) # 5 Hz correspond au dt=0.2s utilisé dans DWA
        
        self.path = path
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v_current = 0.0
        self.w_current = 0.0
        
        rospy.loginfo("En attente de l'odométrie sur /odom...")
        try:
            rospy.wait_for_message('/odom', Odometry, timeout=5)
            rospy.loginfo("Odométrie reçue !")
        except rospy.ROSException:
            rospy.logerr("Timeout : Aucune donnée reçue sur /odom. Vérifiez que Gazebo est lancé.")
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
        
        self.v_current = msg.twist.twist.linear.x
        self.w_current = msg.twist.twist.angular.z

    def run(self):
        if not self.path:
            rospy.logerr("Chemin vide, le robot ne bougera pas.")
            return

        goal_x, goal_y = self.path[-1][0], self.path[-1][1]
        rospy.loginfo(f"Démarrage du mouvement vers ({goal_x:.2f}, {goal_y:.2f})")

        while not rospy.is_shutdown():
            # Vérifier si on est arrivé (tolérance de 15 cm)
            dist_to_goal = np.hypot(self.x - goal_x, self.y - goal_y)
            if dist_to_goal < 0.15:
                rospy.loginfo("Destination atteinte avec succès !")
                self.stop_robot()
                break

            # Calcul des vitesses DWA
            (v, w) = dwa_step(self.x, self.y, self.theta, 
                              self.v_current, self.w_current, self.path)
            
            # Envoi des commandes
            cmd_msg = Twist()
            cmd_msg.linear.x = v
            cmd_msg.angular.z = w
            self.cmd_vel_pub.publish(cmd_msg)
            
            self.rate.sleep()

    def stop_robot(self):
        cmd_msg = Twist()
        self.cmd_vel_pub.publish(cmd_msg)

# --- 5. EXECUTION PRINCIPALE ---
if __name__ == '__main__':
    try:
        rospy.loginfo(f"Calcul du chemin avec {algo_name}...")
        path_pixels, num_nodes, cost = planner()
        
        if not path_pixels:
            rospy.logerr(f"L'algorithme {algo_name} n'a pas trouvé de chemin !")
        else:
            rospy.loginfo(f"Chemin trouvé ({len(path_pixels)} points). Conversion en mètres...")
            path_meters = [grid_to_meters(p[0], p[1]) for p in path_pixels]
            
            # On lance le contrôleur
            controller = DWARobotController(path_meters)
            controller.run()
            
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Erreur inattendue : {e}")