# Version de Reference

## Version stable retenue

La version de reference de demonstration correspond a l'etat du depot local base sur le commit :

`aa2d14535adf54f2665da01c1114549359ca2018`

Cette version inclut :

- authentification etudiante UCA par comptes locaux Django
- espace chat protege
- historique personnel des conversations
- affichage des sources et du niveau de confiance
- health checks durcis
- documentation et packaging de deploiement de demonstration

## Perimetre de la version stable

Ce qui est considere stable pour la demo :

- flux `signup -> login -> chat -> historique`
- API chat protegee
- dashboard admin existant
- pipeline RAG commande via les management commands
- verification de readiness

## Evolutions explicitement hors perimetre de la version stable

- SSO UCA reel
- orchestration serveur avancee
- refactor complet des gros modules RAG
- gestion conversationnelle avancee type favoris/feedback
