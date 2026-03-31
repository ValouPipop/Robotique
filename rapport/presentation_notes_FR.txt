# 📋 NOTES DE PRÉSENTATION — Navigation Autonome

> **Projet** : Robot serveur dans un restaurant  
> **Robot** : TurtleBot3 Burger (Gazebo/ROS)  
> **But** : Comparer navigation classique vs IA pour servir 5 tables

---

## 🔄 1. PIPELINE GLOBAL

```
Vue aérienne restaurant
        ↓
Fichier .world (Gazebo) + destination.json (coordonnées)
        ↓
SLAM (slam_toolbox) → map.pgm + map.yaml
        ↓
    ┌──────────────────────────────┐
    │   APPROCHE CLASSIQUE         │     APPROCHE IA (RL)
    │                              │
    │  Path Planning               │  Entraînement PPO
    │  (A*, Dijkstra, Greedy)      │  (2M steps, ~30min)
    │         ↓                    │         ↓
    │  Contrôleur DWA              │  Modèle best_model.zip
    │  (suivi de chemin)           │  (réseau de neurones)
    │         ↓                    │         ↓
    │  cmd_vel (v, ω)              │  cmd_vel (v, ω)
    └──────────────────────────────┘
                    ↓
            Robot se déplace dans Gazebo
```

**Ce qu'on dit à la présentation :**
> "On a d'abord récupéré une vue aérienne d'un vrai restaurant. On l'a modélisée dans Gazebo pour créer l'environnement de simulation. Ensuite on a fait du SLAM avec slam_toolbox pour récupérer la carte de l'environnement sous forme d'une grille d'occupation. À partir de cette carte, on a implémenté deux approches de navigation complètement différentes qu'on va comparer."

---

## 🗺️ 2. CRÉATION DE LA MAP

### Tentatives SLAM :
- **explore_lite + gmapping** → ❌ Échec (glissement des roues du TurtleBot)  
- **slam_toolbox** → ✅ Succès avec un script Python dédié

### Paramètres de la carte :
| Paramètre | Valeur | Signification |
|---|---|---|
| Resolution | 0.05 m/pixel | 1 pixel = 5 cm |
| Origin | (-2.0, -2.0, 0.0) | Coin bas-gauche de la carte en mètres |
| Occupied threshold | 0.65 | Pixel noir >= 65% = obstacle |
| Free threshold | 0.196 | Pixel blanc <= 19.6% = libre |

### Conversion mètres → pixels :
```
col_x = round((x_metres - origin_x) / resolution)
row_y = (hauteur_image - 1) - round((y_metres - origin_y) / resolution)
```

**Ce qu'on dit :**
> "La carte est stockée en PGM, c'est une image en niveaux de gris. Chaque pixel fait 5cm. On a aussi un fichier YAML qui donne l'origine et la résolution. Pour passer des coordonnées Gazebo en mètres aux pixels de la carte, on fait cette formule simple."

### Inflation des obstacles :
On dilate les murs pour que le robot (rayon 20cm) + marge de sécurité (5cm) ne puisse pas s'approcher trop des obstacles.

```
r_inflation = ceil((r_robot + r_marge) / resolution)
            = ceil((0.20 + 0.05) / 0.05)
            = ceil(5.0) = 5 pixels
```

On crée un masque circulaire de 5 pixels de rayon et on dilate la grille avec `scipy.binary_dilation`.

**Ce qu'on dit :**
> "Avant de faire du pathfinding, on grossit les murs de 5 pixels dans toutes les directions. Comme ça le centre du robot ne peut jamais s'approcher à moins de 25cm d'un mur, ce qui garantit la sécurité."

---

## 📐 3. ALGORITHMES DE PATH PLANNING

### 3.1 — A* (A-Star)

**Principe :** Recherche dans un graphe en combinant le coût réel + une estimation heuristique.

```
f(n) = g(n) + h(n)
```
- `g(n)` = coût réel du départ jusqu'au noeud n (nombre de cases parcourues)
- `h(n)` = heuristique = estimation de la distance restante
- `f(n)` = coût total estimé → on visite toujours le noeud avec le plus petit f

**Heuristique Manhattan :**
```
h(n) = |x_n - x_goal| + |y_n - y_goal|
```
C'est la distance en "bloc de ville" (pas de diagonales).

**Propriétés :**
- ✅ **Optimal** : trouve toujours le plus court chemin
- ✅ **Complet** : s'il existe un chemin, il le trouve
- ❌ Plus lent que Greedy (explore plus de noeuds)

**Ce qu'on dit :**
> "A* c'est l'algorithme le plus connu en pathfinding. Il est optimal parce qu'il combine le coût réel g(n) avec une heuristique h(n) qui estime la distance restante. On utilise la distance de Manhattan comme heuristique. C'est admissible parce qu'elle ne surestime jamais la vraie distance."

---

### 3.2 — Dijkstra

**Principe :** Cas particulier de A* sans heuristique.

```
f(n) = g(n)     (h(n) = 0)
```

Il explore dans toutes les directions depuis le départ, en cercles concentriques, jusqu'à trouver le but.

**Propriétés :**
- ✅ **Optimal** (comme A*)
- ✅ **Complet**
- ❌ Beaucoup plus lent : explore environ 2x plus de noeuds qu'A* car il n'a aucun "sens de direction"

**Ce qu'on dit :**
> "Dijkstra c'est A* sans heuristique. Il explore tout autour de lui en cercles, sans savoir où est la cible. Il est garanti optimal mais il explore beaucoup plus de noeuds — 30 000 en moyenne contre 15 000 pour A*."

---

### 3.3 — Greedy Best-First Search

**Principe :** Utilise UNIQUEMENT l'heuristique, ignore le coût réel.

```
f(n) = h(n)     (g(n) ignoré)
```

Il fonce droit vers le but sans se soucier de la longueur du chemin.

**Propriétés :**
- ✅ **Le plus rapide** (~15ms vs 40ms pour A*)
- ❌ **PAS optimal** : le chemin trouvé peut être plus long (43.3m vs 42.5m)
- ❌ Pas complet dans certains cas (peut se bloquer)

**Ce qu'on dit :**
> "Greedy c'est le plus rapide parce qu'il utilise seulement l'heuristique, il fonce vers le but. Mais il est pas optimal — il peut trouver des chemins un peu plus longs. Sur notre carte, il donne des chemins 1 à 2% plus longs que A*."

---

### Comparaison path planning — TOUTES les tables

| Table | A* (m) | A* noeuds | A* temps | Dijkstra (m) | Dijkstra noeuds | Dijkstra temps | Greedy (m) | Greedy noeuds | Greedy temps |
|---|---|---|---|---|---|---|---|---|---|
| table_1 | 36.60 | 10 151 | 30.1 ms | 36.60 | 19 887 | 45.1 ms | 36.60 | 4 252 | 10.8 ms |
| table_2 | 35.20 | 12 135 | 49.4 ms | 35.20 | 17 234 | 43.8 ms | 35.20 | 4 654 | 8.7 ms |
| table_3 | 43.90 | 19 396 | 56.4 ms | 43.90 | 33 614 | 77.4 ms | 44.60 | 10 323 | 21.8 ms |
| table_4 | 45.90 | 16 688 | 48.3 ms | 45.90 | 37 564 | 85.9 ms | 47.30 | 11 149 | 28.9 ms |
| table_5 | 50.90 | 19 635 | 44.5 ms | 50.90 | 44 719 | 96.1 ms | 52.80 | 12 942 | 22.1 ms |
| **MOY.** | **42.50** | **15 601** | **45.8** | **42.50** | **30 604** | **69.7** | **43.30** | **8 664** | **18.4** |

**Observations clés :**
- A* et Dijkstra trouvent toujours le **même chemin** (optimal) mais Dijkstra explore 2x plus de noeuds
- Greedy est **2.5x plus rapide** que A* mais ses chemins sont **1-4% plus longs**
- Plus la table est loin, plus il y a de noeuds à explorer (logique)

---

## 🎮 4. CONTRÔLEUR DWA (Dynamic Window Approach)

### Rôle
Le path planning donne un chemin en pixels. Mais le robot a besoin de **vitesses** (v, w) pour avancer. Le DWA fait le pont entre les deux.

### Modèle cinématique du robot (unicycle) :
```
x(t+1) = x(t) + v * cos(theta) * dt
y(t+1) = y(t) + v * sin(theta) * dt
theta(t+1) = theta(t) + w * dt
```
- `v` = vitesse linéaire (m/s) → le robot avance
- `w` = vitesse angulaire (rad/s) → le robot tourne
- `dt = 0.2 s` → on met a jour 5 fois par seconde

### Paramètres DWA :
| Paramètre | Valeur | Description |
|---|---|---|
| v_max | 0.5 m/s | Vitesse linéaire maximale du TurtleBot3 |
| w_max | 1.0 rad/s | Vitesse angulaire maximale |
| a_v | 0.5 m/s2 | Accélération linéaire maximale |
| a_w | 1.0 rad/s2 | Accélération angulaire maximale |
| dt | 0.2 s | Pas de temps (5 Hz) |
| Horizon | 5 steps | On simule 5 pas dans le futur (1 seconde) |
| v_samples | 5 | Nombre d'échantillons en vitesse linéaire |
| w_samples | 7 | Nombre d'échantillons en vitesse angulaire |
| alpha (dist_path) | 2.0 | Poids : rester sur le chemin |
| beta (heading) | 1.0 | Poids : orientation vers la cible |
| gamma (velocity) | 0.5 | Poids : aller vite |

### Fenêtre dynamique :
A chaque instant, on calcule les vitesses **physiquement réalisables** :
```
v in [v_actuel - a_v*dt,  v_actuel + a_v*dt]   (borné par [0, v_max])
w in [w_actuel - a_w*dt,  w_actuel + a_w*dt]   (borné par [-w_max, w_max])
```

Ça veut dire : le robot ne peut pas passer instantanément de 0 a 0.5 m/s. Il accélère par paliers.

### Fonction de coût :
On teste les 5x7 = **35 combinaisons** (v, w), et pour chacune on simule la trajectoire sur 5 pas (1 seconde dans le futur). Puis on calcule :
```
J(v, w) = 2.0 * dist_au_chemin  +  1.0 * erreur_cap  +  0.5 * (v_max - v)
```

- **dist_au_chemin** : distance entre le point simulé et le chemin A* → on veut rester sur le chemin
- **erreur_cap** : angle entre l'orientation du robot et le prochain point du chemin → on veut regarder dans la bonne direction
- **v_max - v** : pénalité si le robot va lentement → on veut avancer vite

On prend le (v, w) qui **minimise** J.

**Ce qu'on dit :**
> "Le DWA c'est un contrôleur local. A chaque itération il teste 35 paires de vitesses dans une fenêtre réaliste, simule chaque trajectoire sur 1 seconde, et choisit celle qui minimise un coût. Le coût c'est : rester proche du chemin, regarder vers la cible, et avancer le plus vite possible."

---

## 🤖 5. IA — PPO ACTOR-CRITIC

### Pourquoi Actor-Critic ?
- **DQN** = actions DISCRÈTES (tourne gauche, avance, tourne droite) → mouvement saccadé
- **PPO Actor-Critic** = actions CONTINUES (v in [0, 0.5], w in [-1.0, 1.0]) → mouvement fluide

Le TurtleBot3 accepte des vitesses continues → on DOIT utiliser un algorithme Actor-Critic.

### Architecture du réseau :
```
Observation (26 valeurs)
      |
  MLP Partagé
  |-- Couche 1 : 64 neurones (ReLU)
  |-- Couche 2 : 64 neurones (ReLU)
      |                    |
   ACTOR (pi)          CRITIC (V)
   |                      |
   mu, sigma → N(mu,sigma²)    V(s) → Retour attendu
   |
   Action (v, w)
```

### États (Observation) — 26 valeurs :
| # | Valeur | Description |
|---|---|---|
| 1-24 | 24 rayons LiDAR | Distance aux obstacles dans 24 directions (360 degres, un rayon tous les 15 degres). Portée max : 3.5m |
| 25 | Distance Dijkstra | Vraie distance au but via le chemin le plus court (calculée par BFS/Dijkstra). Tient compte des murs ! |
| 26 | Heading (cap) | Angle relatif entre l'orientation du robot et la direction du but. phi = atan2(y_but - y_robot, x_but - x_robot) - theta_robot, normalisé dans [-pi, pi] |

**Ce qu'on dit :**
> "Le robot voit 24 rayons laser autour de lui qui lui donnent la distance aux murs. En plus il reçoit la vraie distance au but — pas la distance en ligne droite mais la distance en suivant les couloirs, calculée par Dijkstra. Et le cap, c'est l'angle vers le but pour savoir de quel coté tourner."

### Actions — 2 valeurs continues :
```
a = (v, w) in [0.0, 0.5] x [-1.0, 1.0]
```
- `v` = vitesse linéaire (borné par le max du TurtleBot3 : 0.5 m/s)
- `w` = vitesse angulaire (borné a +/-1.0 rad/s)

### Distance Map (Dijkstra Flood Fill) :
Au lieu d'utiliser la distance euclidienne (en ligne droite, ignore les murs), on pré-calcule une **carte de distances** par BFS/Dijkstra depuis chaque table.

```
D(r, c) = min_chemin SUM(w_i)
```
- `w_i = 1.0` pour un déplacement cardinal (haut/bas/gauche/droite)
- `w_i = sqrt(2) = 1.414` pour un déplacement diagonal
- Cellules obstacles = 9999 (inaccessible)

On pré-calcule les 5 cartes (une par table) au lancement. Ça prend ~5 secondes, mais après chaque reset() est instantané.

### Système de récompenses :

| Condition | Récompense | But |
|---|---|---|
| Arrivé (distance < 0.35m) | **+2000** | Fortement encourager l'arrivée |
| Crash (collision) | **-500** | Fortement décourager les collisions |
| Progression | **+50 * (D(t-1) - D(t))** | Encourager le robot a se rapprocher via le vrai chemin |
| Temps | **-0.5** par step | Forcer l'efficacité, éviter les boucles infinies |
| Rotation | **-0.3 * |w|** | Éviter le zigzag, trajectoires plus lisses |

**Formule complète :**
```
R(s, a) = +2000                                  si distance_but < 0.35m  (succes)
         = -500                                   si collision  (crash)
         = 50*(D_prev - D_current) - 0.5 - 0.3*|w|    sinon
```

**Ce qu'on dit :**
> "La récompense est positive si le robot se rapproche du but par le vrai chemin Dijkstra, et négative s'il perd du temps ou tourne trop. Le gros bonus a +2000 pour l'arrivée garantit que l'agent préfère toujours arriver plutôt que d'éviter les risques."

### PPO — Formules mathématiques :

**1. Ratio de probabilité :**
```
r_t(theta) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)
```
C'est le rapport entre la nouvelle politique et l'ancienne. Si > 1, l'action est devenue plus probable. Si < 1, moins probable.

**2. Avantage (GAE - Generalized Advantage Estimation) :**
```
A_hat_t = delta_t + (gamma*lambda)*delta_{t+1} + (gamma*lambda)^2 * delta_{t+2} + ...
delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
```
- `gamma = 0.99` (discount factor → les récompenses futures valent presque autant que les immédiates)
- `lambda = 0.95` (GAE lambda → compromis biais/variance)

L'avantage dit : "est-ce que cette action a été mieux que ce qu'on attendait ?"

**3. Loss CLIP (Actor) :**
```
L_CLIP(theta) = E_t [ min( r_t(theta)*A_hat_t,  clip(r_t(theta), 1-epsilon, 1+epsilon)*A_hat_t ) ]
```
avec `epsilon = 0.2` (clip range)

Le clip empêche les mises a jour trop brutales : si r_t sort de [0.8, 1.2], on le ramène dans cette plage. Ça stabilise l'entraînement.

**4. Loss Value (Critic) :**
```
L_VF(phi) = E_t [ (V_phi(s_t) - V_target)^2 ]
```
Simple erreur quadratique entre la valeur prédite et la valeur réelle.

**5. Entropie :**
```
H(pi_theta) = -E[ log pi_theta(a|s) ]
```
Encourage l'exploration : si l'entropie est élevée, l'agent essaie des actions variées.

**6. Loss totale :**
```
L(theta, phi) = -L_CLIP(theta) + 0.5 * L_VF(phi) - 0.01 * H(pi_theta)
```
- On maximise L_CLIP (d'ou le signe moins)
- On minimise L_VF (erreur du critique)
- On maximise H (d'ou le signe moins, encourage l'exploration)

**7. Target KL :**
```
Si KL(pi_old || pi_new) > 0.015 → on arrête les updates de cette epoch
```
Sécurité supplémentaire : si la politique change trop vite, on stoppe. Ça évite l'effondrement de la politique (quand le modèle "oublie" tout d'un coup).

### Hyperparamètres d'entraînement :
| Paramètre | Valeur | Explication |
|---|---|---|
| Algorithme | PPO | Proximal Policy Optimization |
| Réseau | MLP, 2x64 neurones | Petit réseau, suffisant pour 26 entrées |
| Learning rate | 1e-4 | Pas d'apprentissage (assez petit pour la stabilité) |
| n_steps | 4096 | Nombre de steps avant chaque update du réseau |
| Batch size | 128 | Taille des mini-batches pendant les updates |
| Clip range (epsilon) | 0.2 | Limite du changement de politique |
| Entropy coeff | 0.01 | Encourage un peu d'exploration |
| Target KL | 0.015 | Sécurité : arrête l'update si la politique change trop |
| Total timesteps | 2 000 000 | Nombre total de pas d'entrainement |
| Episode timeout | 1500 steps | Temps max par épisode (~150 secondes) |
| Device | CUDA (GPU) | Accélération GPU |

---

## 📊 6. COMPARAISON FINALE (Classique vs IA)

| Critère | A* | Dijkstra | Greedy | IA (PPO) |
|---|---|---|---|---|
| Taux de succès | **100%** | **100%** | **100%** | **100%** |
| Longueur moy. chemin | 42.50 m | 42.50 m | 43.30 m | **36.38 m** |
| Temps de calcul | 37.8 ms | 60.5 ms | **15.5 ms** | 1371 ms |
| Noeuds/Steps | 15 601 | 30 604 | 8 664 | **733** |
| Distance min. obstacles | >= 0.25 m | >= 0.25 m | >= 0.25 m | 0.07 m |
| Temps d'entraînement | Aucun | Aucun | Aucun | ~30 min |

**Ce qu'on dit :**
> "L'IA trouve des chemins 14% plus courts parce qu'elle se déplace en continu avec des courbes lisses, pas case par case. Par contre elle est 90 fois plus lente a l'exécution et elle frôle les murs a 7 cm, ce qui est dangereux pour un vrai robot. Les classiques garantissent 25 cm de marge."

---

## 🚀 7. AMÉLIORATIONS POSSIBLES

### 7.1 — Hybride : A* + RL (meilleure idée)
- Utiliser A* pour calculer le chemin global (garanti sûr)
- Placer des **waypoints** (points de passage) tous les X mètres le long du chemin A*
- L'IA navigue entre les waypoints successifs → mouvement fluide + sécurité
- On combine le meilleur des deux mondes !

### 7.2 — Safety Layer
- A chaque step, vérifier si un rayon LiDAR < 15 cm
- Si oui → override l'action de l'IA et reculer/tourner
- Garantit qu'on ne crashe jamais

### 7.3 — Curriculum Learning
- Commencer l'entraînement avec des tables proches
- Progressivement augmenter la distance
- L'agent apprend plus vite

### 7.4 — Domain Randomization
- Ajouter du bruit sur les capteurs LiDAR pendant l'entraînement
- Varier légèrement les positions des murs
- Améliore le transfert sim → réel

### 7.5 — Plus d'entraînement + récompenses
- Augmenter la pénalité de temps (actuellement -0.5 → essayer -1.0)
- Ajouter une récompense pour rester loin des murs
- Entrainer plus longtemps (5M steps)

**Ce qu'on dit :**
> "L'amélioration la plus prometteuse c'est l'approche hybride : on fait A* pour le chemin global qui est garanti sûr, et on utilise l'IA comme un contrôleur local qui suit les waypoints avec des mouvements fluides. On combine la fiabilité de A* avec la fluidité de l'IA."
