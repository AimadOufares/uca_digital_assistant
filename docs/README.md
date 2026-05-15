# Documentation technique

## Role du dossier `docs/`

Ce dossier contient la documentation technique du projet **UCA Digital Assistant**.

Il sert surtout a :

- comprendre l'architecture ;
- garder une trace de l'evolution du projet ;
- expliquer le pipeline RAG ;
- preparer une demonstration technique.

Pour la soutenance finale et les supports oraux, utiliser en priorite le dossier `reunion/`.

## Fichiers utiles

| Fichier | Role actuel |
|---|---|
| `RAG_ARCHITECTURE.md` | Reference technique principale sur l'architecture RAG |
| `DEMO_GUIDE.md` | Guide court pour lancer une demonstration |
| `REFERENCE_VERSION.md` | Commit et perimetre de la version stable actuelle |
| `SOUTENANCE_TECHNIQUE.md` | Ancien support technique, encore utile comme annexe |
| `ETAT_AVANCEMENT_PFE_2026-05-05.md` | Rapport historique du 05/05/2026, chiffres depasses |

## Etat actuel a retenir

Reference actuelle :

```text
ce68ee2
```

Resultats principaux :

| Element | Resultat |
|---|---:|
| Tests Django cibles | 59 tests OK |
| Healthcheck RAG | ready = true |
| Benchmark Drive - service top-1 | 92,31 % |
| Benchmark Drive - reponses utiles | 61,54 % |
| Benchmark contexte - reecriture correcte | 93,75 % |
| Benchmark contexte - utilisation correcte du contexte | 93,75 % |

## Relation avec `reunion/`

Le dossier `reunion/` contient les documents les plus recents pour :

- le rapport ;
- la presentation ;
- les questions de demonstration ;
- les reponses au jury ;
- l'evaluation finale.

En resume :

```text
docs/     -> documentation technique et historique
reunion/  -> supports finaux de rapport, reunion et soutenance
```

