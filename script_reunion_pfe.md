# Script oral pour la reunion

## Version courte

"Mon projet s'intitule UCA Digital Assistant. Il s'agit d'un assistant intelligent base sur une architecture RAG pour aider les etudiants a trouver rapidement des informations universitaires a partir de documents officiels de l'Universite Cadi Ayyad.

Au niveau de l'avancement, j'ai deja mis en place la structure principale du systeme. J'ai developpe le backend avec Django, une API de chat, une interface web pour poser les questions, ainsi qu'un module RAG qui prend en charge l'ingestion des documents, leur traitement, l'indexation et la recherche.

J'ai egalement ajoute un mecanisme de resolution de contexte par etablissement, ce qui permet au systeme de mieux cibler les reponses, ainsi qu'un tableau de bord administrateur pour suivre les metriques et lancer des audits.

Concernant les donnees, j'ai deja collecte plus de 2000 fichiers documentaires, et le pipeline a permis de traiter ces donnees et de generer les chunks necessaires pour l'indexation. Le systeme est donc deja operationnel en mode prototype.

Actuellement, je travaille surtout sur l'amelioration de la pertinence des reponses, l'evaluation du retrieval, l'affichage des sources et certaines fonctionnalites complementaires comme l'historique ou l'authentification."

## Version un peu plus detaillee

"Le but de mon projet est de construire un assistant universitaire intelligent capable de repondre aux questions des etudiants en s'appuyant sur des documents officiels, au lieu de generer des reponses sans base documentaire.

Pour cela, j'ai adopte une architecture RAG. Dans une premiere phase, le systeme recupere et traite les documents. Ensuite, ces documents sont indexes dans une base vectorielle afin de permettre une recherche semantique. Enfin, lorsqu'un utilisateur pose une question, le systeme recupere les passages pertinents et genere une reponse contextualisee.

Sur le plan technique, j'utilise Python, Django, Django REST Framework, SQLite, Qdrant, Sentence-Transformers et une generation via LM Studio compatible OpenAI. J'ai aussi developpe une interface web simple en HTML, CSS et JavaScript pour la demonstration.

En termes d'avancement concret, plusieurs fonctionnalites sont deja disponibles : le chat, l'API, la resolution de contexte par etablissement, la gestion des questions ambiguës, le filtrage des questions hors UCA, le pipeline d'ingestion, ainsi qu'un dashboard administrateur.

Les resultats obtenus montrent que le systeme est deja fonctionnel, mais qu'il reste des ameliorations a faire au niveau de la precision et de la couverture des reponses. Les prochaines etapes portent donc sur l'optimisation du retrieval, l'ajout des citations des sources et la finalisation de l'experience utilisateur."

## Reponses courtes si on te demande "ou en es-tu ?"

- "Le projet est deja fonctionnel sous forme de prototype."
- "La chaine complete ingestion, indexation, retrieval et chat est deja en place."
- "Je suis actuellement dans une phase d'amelioration de la precision et de finalisation."
- "La base technique est developpee, et je travaille maintenant sur la qualite des resultats et les fonctionnalites de finition."

## Si on te demande "qu'est-ce qui reste ?"

- "Il reste surtout l'amelioration de la pertinence des reponses."
- "Je dois encore exposer clairement les sources dans l'interface."
- "Certaines fonctionnalites utilisateur comme l'historique et l'authentification ne sont pas encore finalisees."
- "Je dois aussi consolider l'evaluation experimentale avant la version finale."
