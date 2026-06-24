# Script oral - Presentation PFE

Version alignee avec la nouvelle structure en 18 slides, selon les consignes d'encadrement.

Le principe est simple : les slides affichent les idees cles, et l'oral explique en phrases courtes.

## Slide 1 - Page de garde

Bonjour, je vais vous presenter mon projet de fin d'etudes intitule **UCA Digital Assistant**.

Il s'agit d'un assistant intelligent base sur une architecture RAG, destine a faciliter l'acces des etudiants aux informations universitaires.

Ce sujet est propose par le **Pole Digitalisation de la Presidence de l'Universite Cadi Ayyad**.

## Slide 2 - Plan de la presentation

Ma presentation est organisee en plusieurs parties.

Je vais commencer par le contexte general et l'organisme d'accueil.

Ensuite, je presenterai le sujet, la problematique et les exigences.

Puis je passerai a la methodologie, au planning, a l'architecture technique, a la realisation, a l'evaluation et enfin a la demonstration avec les perspectives.

## Slide 3 - Introduction generale

Aujourd'hui, les universites utilisent de plus en plus de services numeriques.

On trouve des sites officiels, des plateformes pedagogiques, des documents administratifs et plusieurs canaux de communication.

Cette transformation numerique est importante, mais elle cree aussi une difficulte : l'information existe, mais elle est parfois dispersee.

Pour un etudiant, trouver rapidement une information fiable peut donc devenir complique.

## Slide 4 - Organisme d'accueil : Presidence de l'UCA

Le projet s'inscrit dans le cadre de l'Universite Cadi Ayyad.

Plus precisement, le sujet est propose par le Pole Digitalisation de la Presidence.

Ce pole s'interesse a l'amelioration des services numeriques et a la valorisation de l'information universitaire.

Le projet repond donc a un besoin concret : faciliter l'acces aux informations utiles pour les etudiants.

## Slide 5 - Contexte numerique UCA

Dans le contexte de l'UCA, l'etudiant peut trouver l'information dans plusieurs sources.

Il y a les sites web officiels, les plateformes universitaires, les documents administratifs, les services administratifs et les canaux de communication.

Le probleme principal n'est donc pas l'absence d'information.

Le probleme est plutot la dispersion de cette information entre plusieurs espaces.

## Slide 6 - Description du sujet

Le sujet consiste a developper une application web intelligente.

L'etudiant pose une question en langage naturel.

Le systeme recherche les passages pertinents dans les documents de l'UCA.

Ensuite, il genere une reponse claire, avec des sources et un niveau de confiance.

L'application contient aussi un dashboard administrateur pour suivre les documents, les audits et les resultats.

## Slide 7 - Problematique

La problematique principale est la suivante.

Comment permettre aux etudiants d'acceder rapidement a une information universitaire fiable, alors que les sources sont nombreuses, dispersees et parfois difficiles a exploiter ?

Cette problematique est importante, car une mauvaise information peut causer une perte de temps ou une mauvaise orientation.

La solution doit donc etre a la fois utile, fiable et verifiable.

## Slide 8 - Exigences fonctionnelles

Les exigences fonctionnelles sont organisees autour de deux profils.

Pour l'etudiant, le systeme doit permettre de poser une question, recevoir une reponse, consulter les sources, voir le niveau de confiance, garder l'historique et donner un feedback.

Pour l'administrateur, le systeme doit permettre de gerer les documents, suivre le pipeline RAG, consulter les audits, analyser les conversations et visualiser les metriques.

## Slide 9 - Exigences non fonctionnelles

En plus des fonctionnalites, le systeme doit respecter plusieurs qualites.

Il doit etre fiable, car les reponses doivent etre basees sur des documents.

Il doit etre tracable, avec les sources, l'historique, les feedbacks et les audits.

Il doit aussi etre securise, ergonomique, performant et maintenable.

Ces exigences sont importantes pour une solution utilisable dans un contexte universitaire.

## Slide 10 - Methodologie de developpement

Le projet a ete realise de maniere progressive et iterative.

J'ai commence par analyser le besoin et le contexte.

Ensuite, j'ai collecte et prepare les documents.

Puis j'ai concu l'architecture, developpe le pipeline RAG, l'application Django, l'interface chat et le dashboard administrateur.

La derniere partie a ete consacree aux tests, a l'evaluation et aux ameliorations.

## Slide 11 - Planning Gantt

Ce planning resume l'organisation du travail sur quatre mois.

Le premier mois a ete consacre a l'analyse du besoin, a l'etude du contexte et au debut de la collecte des documents.

Les deuxieme et troisieme mois ont ete consacres a la conception, au pipeline RAG et au developpement de l'application.

Le quatrieme mois a ete consacre au dashboard, aux tests, a l'evaluation, au rapport et a la preparation de la soutenance.

## Slide 12 - Architecture globale du systeme

L'architecture globale est organisee autour d'une application Django.

Django gere l'authentification, les conversations, l'historique, le feedback et le dashboard administrateur.

Le moteur RAG est connecte a cette application pour rechercher les documents pertinents et produire des reponses contextualisees.

La base de donnees conserve les utilisateurs, les conversations, les documents, les feedbacks et les informations d'audit.

## Slide 13 - Pipeline RAG offline

La phase offline prepare les documents avant l'utilisation du chatbot.

Les documents sont extraits, nettoyes, decoupes en chunks, puis enrichis avec des metadonnees.

Ensuite, le systeme genere les embeddings et construit les index de recherche.

Cette phase permet de rendre le corpus documentaire exploitable par le moteur RAG.

## Slide 14 - Pipeline RAG online

La phase online commence lorsque l'etudiant pose une question.

Le systeme analyse la question et utilise le contexte de la conversation si necessaire.

Ensuite, il recherche les passages pertinents dans les index.

Ces passages sont transmis au modele de langage pour generer une reponse avec sources et niveau de confiance.

## Slide 15 - Recherche hybride : FAISS + BM25

La recherche hybride combine deux approches complementaires.

FAISS permet de retrouver les passages proches du sens de la question.

BM25 permet de mieux detecter les mots exacts, comme les noms de services ou de plateformes.

Les resultats des deux moteurs sont ensuite fusionnes pour ameliorer la qualite du contexte donne au modele.

## Slide 16 - Application realisee

L'application realisee contient deux espaces principaux.

Le premier est l'espace etudiant, avec le chat, les reponses, les sources, le niveau de confiance, l'historique et le feedback.

Le deuxieme est le dashboard administrateur, qui permet de suivre les documents, les audits, les conversations et les metriques.

Les deux interfaces reposent sur le meme moteur RAG.

## Slide 17 - Evaluation et resultats

Le projet a ete evalue avec plusieurs indicateurs.

Les tests Django cibles sont valides avec **59 tests OK**.

Le healthcheck indique que le systeme RAG est operationnel.

Les resultats montrent aussi un **Top-1 service Drive de 92,31 %**, un **BM25 hit@k de 84,62 %** et un **context rewrite/use de 93,75 %**.

Ces resultats montrent que la partie retrieval est solide, meme si elle depend encore de la qualite du corpus et des chunks.

## Slide 18 - Demonstration, conclusion et perspectives

Pour terminer, je vais presenter une courte demonstration du prototype.

La demonstration montre le parcours principal : poser une question, obtenir une reponse, consulter les sources et verifier le niveau de confiance.

En conclusion, le projet a permis de developper une application web fonctionnelle basee sur une architecture RAG, avec recherche hybride, dashboard administrateur et evaluation mesurable.

Les perspectives concernent l'enrichissement du corpus, l'amelioration du chunking, l'ajout de nouvelles sources officielles, le renforcement de l'evaluation et un futur deploiement institutionnel.

Merci pour votre attention.
