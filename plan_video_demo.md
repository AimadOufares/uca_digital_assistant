# Plan de video de demonstration

## Objectif

Montrer en 3 a 5 minutes que le projet est deja fonctionnel, que l'architecture est claire et que plusieurs fonctionnalites sont deja developpees.

## Version de reference

La video doit etre preparee sur la **version stable de reference** :

- commit `c25f5769e105aac35962c4cac8dd9451c2e01f83`

La refonte plus recente du pipeline RAG vers un backend hybride Qdrant doit etre mentionnee uniquement comme **travail en cours**, sans en faire le coeur de la demonstration si elle n'est pas encore stable.

## Duree recommandee

- **video courte** : 3 a 5 minutes
- **format** : capture d'ecran avec commentaire vocal simple

## Message a faire passer

L'idee principale n'est pas de montrer toutes les modifications recentes, mais de prouver que :

- le projet fonctionne deja sur une version stable
- l'assistant peut repondre a des questions universitaires
- le systeme gere le contexte, l'ambiguite et le hors perimetre
- une evolution technique plus avancee est en cours, mais pas encore stabilisee

## Scenario conseille

### 1. Introduction rapide

Dire en 20 a 30 secondes :

"Bonjour Monsieur, je vais vous presenter l'avancement de mon projet de fin d'etudes. Le projet s'appelle UCA Digital Assistant. Il s'agit d'un assistant intelligent base sur une architecture RAG, concu pour repondre aux questions des etudiants a partir de documents officiels de l'Universite Cadi Ayyad. Dans cette demonstration, je vais d'abord presenter rapidement la logique principale du projet et le role des modules, puis montrer le fonctionnement general de l'application."

### 2. Presentation rapide de la structure du projet

Montrer rapidement :

- le dossier du projet
- les modules `api_app`
- le module `rag_module`
- le dossier `data_storage`

Phrase conseillee :

"Le projet est organise en un backend Django, une interface web de chat et un module RAG qui gere l'ingestion, le traitement, l'indexation et la generation des reponses."

### Script bref propose pour cette partie

"Voici le dossier principal du projet, `uca_digital_assistant`. Le projet est organise autour d'un backend Django. Je vais expliquer rapidement le role de chaque module principal.

Le dossier `api_app` correspond a la partie application web. Il contient l'interface utilisateur du chatbot, les vues, les routes API et les pages HTML.

Le dossier `core` correspond au point d'entree principal du projet Django. Il contient la configuration generale, les settings, les URLs principales et le lancement du projet.

Le dossier `data_storage` regroupe toutes les donnees utilisees par le systeme. On y retrouve les documents collectes, les donnees traitees, la base vectorielle, les index, ainsi que certains fichiers de cache et de rapports.

Enfin, le module `rag_module` contient la logique principale du système RAG. Il est structuré en plusieurs sous-dossiers : `offline` pour le traitement et l'indexation des données, `retrieval` pour la recherche d'informations, `generation` pour la création des réponses par l'IA, `audit` et `evaluation` pour le suivi qualité, et `shared` pour les utilitaires partagés."

### 3. Demonstration de l'interface chat

Montrer :

- la page chat
- le champ de saisie
- le selecteur d'etablissement

Faire une premiere question simple, par exemple :

- "Quels documents sont necessaires pour une inscription administrative ?"

Expliquer :

"Ici, l'utilisateur peut poser une question libre. Le systeme recherche les passages pertinents dans la base documentaire avant de generer une reponse."

### 4. Demonstration du contexte par etablissement

Choisir un etablissement dans la liste, par exemple `FSSM`, puis poser une question :

- "Comment se passe l'inscription ?"

Expliquer :

"Le systeme peut utiliser le contexte de l'etablissement pour limiter la recherche et rendre la reponse plus precise."

### 5. Demonstration d'un cas ambigu

Poser une question volontairement ambigue sans contexte clair.

Exemple :

- "Comment se passe l'inscription ?"

Expliquer :

"Lorsque la question est trop generale, l'assistant peut demander une clarification au lieu de produire une reponse non fiable."

### 6. Demonstration d'un cas hors perimetre

Poser une question hors UCA.

Exemple :

- "Comment s'inscrire a l'Universite Mohammed V ?"

Expliquer :

"Le systeme detecte egalement les questions en dehors du perimetre cible et retourne une reponse de limitation."

### 7. Presentation du dashboard administrateur

Montrer :

- la page dashboard
- les indicateurs visibles
- les actions d'audit disponibles

Expliquer :

"J'ai egalement mis en place un tableau de bord d'administration pour suivre l'etat des donnees, des audits et de certaines metriques du systeme."

### 8. Conclusion

Conclure en 20 secondes :

"A ce stade, le projet est deja fonctionnel au niveau de l'ingestion, du retrieval, de la generation, de l'API et de l'interface utilisateur. En parallele, je travaille sur une evolution plus avancee du pipeline RAG vers un backend hybride Qdrant, mais cette partie est encore en cours de stabilisation."

## Ce qu'il vaut mieux eviter pendant la video

- ne pas baser la demo sur la branche actuelle si elle est instable
- ne pas lancer une sequence technique risquee pendant l'enregistrement
- ne pas promettre que la refonte hybride est finalisee si ce n'est pas le cas
- ne pas surcharger la video avec trop de details internes

## Conseils pratiques

- utiliser une resolution lisible
- zoomer un peu si le texte est petit
- ne pas aller trop vite
- preparer les questions a l'avance
- fermer les fenetres inutiles avant l'enregistrement
- ne pas afficher de secrets ou de cles API
- verifier avant l'enregistrement que tu es bien sur la version de demonstration retenue

## Questions conseillees pour la demo

- "Quels documents sont necessaires pour une inscription administrative ?"
- "Comment se passe l'inscription ?"
- "Quelles sont les conditions d'inscription en licence ?"
- "Comment s'inscrire a l'Universite Mohammed V ?"

## Phrase utile si on te demande ce qui a change apres

"Apres cette version stable, j'ai lance une refonte technique du pipeline RAG vers une architecture hybride basee sur Qdrant. Cette partie est en cours de stabilisation, donc je ne la prends pas comme support principal de la demonstration d'aujourd'hui."
