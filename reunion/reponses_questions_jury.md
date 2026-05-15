# Reponses aux questions probables du jury

## Pourquoi avoir choisi une architecture RAG ?

Reponse :

> J'ai choisi le RAG parce qu'un LLM seul peut produire des reponses generales ou hallucinees. Dans mon projet, le modele commence par recuperer des passages depuis les documents UCA, puis il genere une reponse a partir de ce contexte. Cela rend la reponse plus controlee, plus verifiable et plus adaptee au contexte universitaire.

## Pourquoi ne pas utiliser seulement un moteur de recherche ?

Reponse :

> Un moteur de recherche retourne surtout des documents ou des liens. L'objectif du projet est different : l'etudiant pose une question en langage naturel et recoit une reponse synthetique, avec des sources. Le RAG combine donc recherche documentaire et generation controlee.

## Pourquoi FAISS + BM25 ?

Reponse :

> FAISS permet une recherche semantique par embeddings, utile lorsque la question n'utilise pas exactement les memes mots que le document. BM25 est utile pour les noms exacts de services comme PEDOC, UC@Student ou HPC UCA. La combinaison des deux rend le retrieval plus robuste.

## Pourquoi LM Studio ?

Reponse :

> LM Studio permet de tester un modele local via une API compatible OpenAI. C'est pratique pour une demonstration locale et cela evite de dependre totalement d'un service externe. En revanche, la latence depend fortement des capacites du PC utilise.

## Pourquoi la generation est lente ?

Reponse :

> Les tests ont ete realises sur un PC personnel avec un Intel Core i7-8665U, 16 Go de RAM et une carte graphique integree Intel UHD Graphics 620. Ce n'est pas une machine optimisee pour l'inference LLM. Le retrieval reste relativement rapide, mais la generation avec LM Studio est plus couteuse. Sur un serveur avec GPU dedie, la latence serait probablement plus faible.

## Pourquoi certaines reponses sont incomplètes ?

Reponse :

> La qualite finale depend de trois elements : la qualite du document source, la qualite du chunk retrouve et la capacite du modele a synthetiser. Si un chunk est partiel ou mal aligne, la reponse finale peut etre incomplete. C'est pour cela que j'ai separe l'evaluation du retrieval et celle de la generation.

## Quelle est la difference entre retrieval et generation ?

Reponse :

> Le retrieval cherche les passages pertinents dans les documents. La generation transforme ces passages en reponse lisible. Dans mon evaluation, le retrieval est plus solide que la generation : le service correct est retrouve en top-1 dans 92,31 % des cas, tandis que les reponses utiles sont autour de 61,54 %.

## Comment avez-vous evalue le module ?

Reponse :

> J'ai utilise plusieurs niveaux de validation : tests Django, healthcheck RAG, benchmark Drive, benchmark contextuel et evaluation generation. Le benchmark Drive mesure si le bon service est retrouve. Le benchmark contextuel mesure la capacite a garder ou changer le contexte dans une conversation.

## Pourquoi le projet est-il plus qu'une simple demo ?

Reponse :

> Le projet contient une application web complete avec authentification, chat, historique, dashboard admin, module RAG, ingestion documentaire, indexation FAISS, BM25, generation, sources et evaluation. Ce n'est donc pas seulement une interface qui appelle un modele, mais une chaine complete de traitement et de recherche d'information.

## Quelles sont les limites actuelles ?

Reponse :

> Les limites principales sont la latence de generation, certaines reponses trop extractives, un corpus encore a enrichir, des metadonnees a harmoniser et une version encore locale. Ces limites sont identifiees et deviennent les perspectives naturelles du projet.

## Comment ameliorer le projet en production ?

Reponse :

> Pour une version production, je proposerais d'enrichir le corpus officiel, nettoyer les metadonnees, ameliorer le chunking, utiliser PostgreSQL, migrer le vector store vers Qdrant, deployer sur VPS ou serveur dedie, ajouter un modele plus rapide et integrer le SSO institutionnel.

## Pourquoi ne pas avoir deploye directement en production ?

Reponse :

> L'objectif du PFE est d'abord de concevoir, implementer et evaluer un prototype fonctionnel. Le deploiement production demande des choix institutionnels : SSO, securite, serveur, supervision, gouvernance des donnees et maintenance du corpus. Je le presente donc comme une perspective.

## Quelle note donneriez-vous au projet ?

Reponse :

> Je le presenterais comme un prototype avance autour de 16/20. Le retrieval est solide, l'application est complete et les evaluations sont defendables. Ce qui empeche une note plus elevee, c'est surtout la generation encore lente, le corpus a enrichir et l'absence de deploiement production.

