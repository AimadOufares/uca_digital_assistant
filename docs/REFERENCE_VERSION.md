# Version de reference

## Version stable actuelle

La version de reference actuelle correspond au commit :

```text
ce68ee202cfc9c9ff3d064b50976eca6ebba60cf
```

Commit court :

```text
ce68ee2
```

Date de reference : `2026-05-15`

Cette version correspond a la preparation finale de demonstration et de soutenance.

## Contenu de la version

Cette version inclut :

- application web Django ;
- inscription et connexion etudiante ;
- restriction possible par domaines email UCA ;
- interface chat protegee ;
- historique des conversations ;
- gestion multi-conversations ;
- affichage des sources et du niveau de confiance ;
- dashboard administrateur ;
- health checks `live` et `ready` ;
- module RAG avec ingestion, processing, indexing, retrieval et generation ;
- recherche hybride FAISS + BM25 ;
- contexte conversationnel ;
- evaluation Drive ;
- evaluation contextuelle ;
- documents de reunion et de soutenance dans `reunion/`.

## Resultats de validation

Derniers resultats utiles :

| Element | Resultat |
|---|---:|
| Tests Django cibles | 59 tests OK |
| Healthcheck RAG | ready = true |
| Service top-1 Drive | 92,31 % |
| Reponses utiles generation | 61,54 % |
| Reecriture contextuelle | 93,75 % |
| Utilisation correcte du contexte | 93,75 % |

## Perimetre stable pour la demonstration

Ce qui est considere stable :

- flux `signup -> login -> chat -> historique` ;
- API chat protegee ;
- affichage des sources ;
- dashboard admin ;
- healthcheck RAG ;
- benchmark Drive ;
- scenario contextuel court ;
- documentation de soutenance dans `reunion/`.

## Hors perimetre de cette version

Les elements suivants restent des perspectives :

- SSO UCA reel ;
- deploiement production durci ;
- orchestration serveur institutionnelle ;
- optimisation GPU/serveur pour la generation LLM ;
- feedback utilisateur utile / non utile ;
- sources entierement cliquables ;
- migration complete vers Qdrant/PostgreSQL/VPS.

