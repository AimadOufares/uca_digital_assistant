# Plan de presentation - UCA Digital Assistant

## Objectif

Ce document sert de base pour preparer les slides de soutenance. Il ne remplace pas le rapport ; il donne une structure orale claire et defendable.

## Slide 1 - Titre

**UCA Digital Assistant**

Conception et realisation d'un assistant universitaire intelligent base sur une architecture RAG.

Elements a afficher :

- nom du projet ;
- nom de l'etudiant ;
- filiere / encadrant ;
- Universite Cadi Ayyad ;
- annee universitaire.

Message oral :

> Mon projet consiste a developper un assistant numerique intelligent capable d'aider les etudiants a retrouver rapidement des informations fiables sur les services et plateformes de l'Universite Cadi Ayyad.

## Slide 2 - Probleme

Probleme traite :

- les informations universitaires sont dispersees ;
- les plateformes ont parfois des noms proches ;
- l'etudiant ne sait pas toujours ou chercher ;
- un chatbot LLM seul peut donner une reponse non verifiee.

Message oral :

> Le probleme principal n'est pas seulement de repondre a une question, mais de retrouver une information fiable dans un corpus universitaire reel.

## Slide 3 - Objectifs

Objectifs du projet :

- fournir une interface de chat simple pour l'etudiant ;
- exploiter les documents Drive et les sources UCA ;
- utiliser une architecture RAG pour limiter les hallucinations ;
- afficher les sources et un niveau de confiance ;
- conserver l'historique des conversations ;
- evaluer objectivement le comportement du module.

## Slide 4 - Pourquoi RAG ?

Comparer trois approches :

| Approche | Limite |
|---|---|
| Recherche classique | ne comprend pas toujours l'intention |
| LLM seul | risque d'hallucination |
| RAG | cherche dans les documents puis genere une reponse |

Message oral :

> Le RAG permet de relier la reponse du modele a des documents reels. C'est ce qui rend la reponse plus defendable.

## Slide 5 - Architecture globale

Schema conseille :

```text
Etudiant
  -> Interface chat
  -> Backend Django
  -> Module RAG
  -> FAISS + BM25
  -> LM Studio
  -> Reponse + sources + confiance
```

Elements a expliquer :

- Django pour l'application web ;
- `rag_module` pour la partie IA/documentaire ;
- FAISS pour la recherche vectorielle ;
- BM25 pour la recherche lexicale ;
- LM Studio pour la generation locale.

## Slide 6 - Pipeline offline

Pipeline de preparation :

```text
Documents Drive / UCA
  -> extraction
  -> nettoyage
  -> chunking
  -> metadonnees
  -> embeddings
  -> index FAISS
  -> corpus BM25
```

Message oral :

> Cette phase transforme les documents bruts en une base documentaire exploitable par le module RAG.

## Slide 7 - Pipeline online

Pipeline utilisateur :

```text
Question
  -> analyse de la question
  -> recherche hybride
  -> garde-fous
  -> prompt final
  -> generation / fallback
  -> reponse avec sources
```

Message oral :

> Quand l'etudiant pose une question, le systeme ne repond pas directement. Il cherche d'abord les passages pertinents, puis construit une reponse a partir de ces passages.

## Slide 8 - Fonctionnalites developpees

Fonctionnalites :

- inscription et connexion ;
- chat protege ;
- historique des conversations ;
- multi-conversations ;
- affichage des sources ;
- niveau de confiance ;
- dashboard administrateur ;
- healthcheck RAG ;
- benchmark Drive et contexte conversationnel.

## Slide 9 - Evaluation

Resultats a afficher :

| Evaluation | Resultat |
|---|---:|
| Tests Django | 59 tests OK |
| Healthcheck RAG | ready = true |
| Service top-1 sur Drive | 92,31 % |
| Reponses utiles | 61,54 % |
| Reecriture contextuelle | 93,75 % |
| Utilisation correcte du contexte | 93,75 % |

Message oral :

> L'evaluation montre que le retrieval est le point fort du projet. La generation fonctionne, mais elle reste limitee par la latence de LM Studio et la qualite des chunks.

## Slide 10 - Contexte materiel

Configuration de test :

- Intel Core i7-8665U ;
- 16 Go RAM ;
- Intel UHD Graphics 620 ;
- environnement local ;
- LM Studio local.

Message oral :

> Les tests ont ete faits sur un PC personnel sans GPU dedie. Les latences de generation doivent donc etre interpretees comme une limite de l'environnement local, pas comme une faiblesse de l'architecture RAG.

## Slide 11 - Limites

Limites a assumer :

- generation lente ;
- certaines reponses trop extractives ;
- corpus encore a enrichir ;
- metadonnees a harmoniser ;
- version locale, non encore production ;
- SSO institutionnel non integre.

## Slide 12 - Perspectives

Perspectives :

- enrichir le corpus officiel ;
- nettoyer les metadonnees ;
- ameliorer le chunking ;
- ajouter feedback utilisateur ;
- rendre les sources plus cliquables ;
- tester un modele LLM plus rapide ;
- migration future vers PostgreSQL + Qdrant + VPS ;
- integration SSO UCA.

## Slide 13 - Demonstration

Scenario court :

1. connexion ;
2. ouverture du chat ;
3. question sur UC@Student ;
4. question sur PEDOC ;
5. question contextuelle ;
6. affichage des sources ;
7. dashboard admin.

## Slide 14 - Conclusion

Message final :

> UCA Digital Assistant est un prototype avance, evalué et demonstrable. Il combine une application web complete, un module RAG, une exploitation des documents Drive, un contexte conversationnel et une evaluation mesurable. Les resultats montrent une base solide, avec des perspectives claires vers une version institutionnelle plus robuste.

