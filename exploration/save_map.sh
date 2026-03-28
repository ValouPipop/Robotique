#!/bin/bash
# Repertoire de destination des cartes
MAP_DIR="/home/valou/Robotique/map"

# On demande le nom de la carte a sauvegarder
read -p "Entrez un nom pour la carte (ex: ma_carte) : " map_name

if [ -z "$map_name" ]
then
    echo "Nom vide, annulation..."
    exit 1
fi

echo "Sauvegarde de la carte dans le dossier $MAP_DIR sous le nom $map_name..."

# Utilise map_saver pour sauvegarder
rosrun map_server map_saver -f "$MAP_DIR/$map_name"

echo "Carte sauvegardée ! Vous trouverez les fichiers .yaml et .pgm dans le dossier map."
