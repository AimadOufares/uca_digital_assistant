# Script de reunion - UCA Digital Assistant

## 1. Objectif de la reunion

L'objectif de cette reunion est de presenter clairement l'etat actuel du projet **UCA Digital Assistant**, montrer que la solution est maintenant demonstrable, expliquer les resultats obtenus avec les documents Drive, puis demander l'avis de l'encadrant sur les choix restants avant de se concentrer sur le rapport et la presentation finale.

Message principal a faire passer :

> Le projet n'est plus seulement une idee ou une maquette. C'est une application web demonstrable basee sur une architecture RAG, avec authentification, chat, historique, sources, confiance, corpus Drive et evaluation.

## 2. Introduction orale proposee

Bonjour Professeur.

Aujourd'hui, je vais vous presenter l'avancement actuel de mon projet **UCA Digital Assistant**. L'objectif du projet est de proposer un assistant intelligent pour aider les etudiants de l'Universite Cadi Ayyad a retrouver rapidement des informations fiables sur les plateformes, services et procedures numeriques de l'UCA.

La solution repose sur une architecture **RAG**. Cela signifie que le systeme ne repond pas uniquement avec les connaissances generales d'un modele de langage. Il commence d'abord par rechercher les passages pertinents dans une base documentaire construite a partir de documents UCA, notamment les fichiers partages sur Drive, puis il produit une reponse appuyee sur ces sources.

Depuis la derniere etape, j'ai avance sur trois axes :

- la finalisation de l'application web etudiante ;
- l'integration des documents Drive dans le corpus RAG ;
- l'evaluation du module avec des questions/reponses de reference.

## 3. Presentation courte du projet

L'application se presente comme un espace etudiant avec une interface de chat.

L'etudiant peut :

- creer un compte ou se connecter ;
- poser une question en langage naturel ;
- recevoir une reponse basee sur les documents indexes ;
- consulter les sources utilisees ;
- voir un niveau de confiance ;
- garder l'historique de ses conversations.

Du cote technique, le projet contient :

- un backend Django ;
- une API avec Django REST Framework ;
- un module RAG separe ;
- une indexation FAISS ;
- une recherche lexicale BM25 ;
- des embeddings avec `BAAI/bge-m3` ;
- une generation locale via LM Studio ;
- des guardrails pour eviter les reponses non supportees ;
- un dashboard admin et des health checks.

## 4. Architecture a expliquer simplement

Schema oral :

```text
Documents UCA / Drive
  -> extraction du texte
  -> nettoyage
  -> decoupage en chunks
  -> enrichissement avec metadata
  -> indexation FAISS + BM25
  -> question utilisateur
  -> retrieval des passages pertinents
  -> generation ou fallback extractif
  -> reponse avec sources et confiance
```

Phrase a dire :

> L'interet de cette architecture est de separer la preparation de la base documentaire et l'utilisation en temps reel. Cela permet d'avoir des reponses plus justifiables qu'un chatbot classique, car chaque reponse est liee a des documents sources.

## 5. Ce qui a ete realise

### Application web

- interface de chat etudiante ;
- authentification locale ;
- restriction possible par domaines email UCA ;
- historique des conversations ;
- creation et archivage de conversations ;
- affichage des sources ;
- niveau de confiance ;
- dashboard administrateur.

### Module RAG

- ingestion de fichiers HTML, PDF, DOCX, TXT et MD ;
- traitement et nettoyage du texte ;
- chunking ;
- indexation FAISS ;
- recherche hybride FAISS + BM25 ;
- guardrails de pertinence ;
- abstention si les sources sont insuffisantes ;
- integration de LM Studio ;
- fallback extractif si le LLM local ne repond pas.

### Contexte conversationnel

Le systeme garde un contexte de conversation pour mieux gerer les questions de suivi.

Exemple :

```text
Question 1 : Comment obtenir mon attestation sur UC@Student ?
Question 2 : Et pour les delais ?
```

Le systeme peut reformuler la deuxieme question en :

```text
Quel est le delai pour attestation sur UC@Student ?
```

Cela ameliore le retrieval, car la question devient plus complete.

## 6. Exploitation des documents Drive

J'ai exploite les documents partages sur Drive comme source documentaire principale pour tester l'assistant sur les services numeriques UCA.

Services couverts :

- UC@Student ;
- PEDOC ;
- CIP ;
- UCAPLAT ;
- Espace Diplomes ;
- Mobilite Internationale ;
- Soutien-Recherche ;
- HPC UCA ;
- PUCAStaff ;
- Centre de Conferences ;
- Appels a Projets ;
- Clubs des etudiants.

Phrase a dire :

> Les documents Drive ne sont pas seulement stockes dans le projet. Ils sont traites, decoupes, indexes et utilises par le module RAG pour repondre aux questions.

## 7. Evaluation realisee

J'ai prepare un fichier d'evaluation `drive_QR.md` avec 20 questions de reference.

Pour chaque question, j'ai compare :

- la question posee ;
- la reponse de reference ;
- le document retrouve par le retrieval ;
- la reponse produite par le module RAG.

Resultats actuels :

- documents pertinents retrouves par le retrieval : `18/20`, soit `90 %` ;
- score global des reponses : `40/60`, soit `66,7 %` ;
- reponses acceptables : `14/20`, soit `70 %` ;
- principales erreurs : confusion entre `Club UCA` et `Clubs des etudiants`, et certains chunks UCAPLAT mal alignes.

Interpretation :

> Le retrieval fonctionne globalement bien : il retrouve souvent les bons documents. La limite principale est au niveau de la generation finale, surtout avec LM Studio qui est parfois lent ou instable.

## 8. Limites actuelles a assumer

Il faut presenter les limites calmement, comme des points de maturite technique, pas comme des echecs.

Limites principales :

- LM Studio fonctionne, mais il est lent sur certains prompts RAG ;
- certaines reponses fallback sont trop extractives ;
- quelques documents proches peuvent etre confondus ;
- certaines metadata doivent etre ameliorees ;
- le corpus reste limite et doit etre enrichi ;
- l'application est encore une version de demonstration locale.

Phrase a dire :

> Les resultats montrent que le coeur RAG fonctionne, mais aussi que la qualite finale depend fortement de la qualite des chunks, des metadata et du modele de generation. C'est justement ce que l'evaluation m'a permis d'identifier.

## 9. Demonstration conseillee

### Etape 1 : ouvrir l'application

Dire :

> Voici l'interface de l'application. L'acces au chat est protege par authentification pour se rapprocher d'un vrai espace etudiant.

### Etape 2 : montrer le chat

Dire :

> L'utilisateur peut poser une question en langage naturel. Le systeme va chercher dans les documents indexes avant de produire une reponse.

### Etape 3 : poser une question qui marche bien

Questions conseillees :

- `Quel est le role de la plateforme CIP ?`
- `A quoi sert la plateforme HPC UCA ?`
- `A quoi sert la plateforme Mobilite Internationale ?`
- `A quoi sert PUCA Staff ?`

Eviter pour la demo :

- `A quoi sert UCAPLAT ?`
- `Comment les activites des clubs etudiants sont-elles validees ?`

Ces deux questions sont utiles pour l'analyse, mais pas ideales pour une demonstration fluide.

### Etape 4 : montrer les sources

Dire :

> La reponse est accompagnee des sources retrouvees. Cela permet de garder une trace documentaire et de reduire le risque d'hallucination.

### Etape 5 : montrer l'historique

Dire :

> Les conversations sont conservees, ce qui permet a l'etudiant de retrouver ses anciennes questions.

### Etape 6 : montrer le health check ou dashboard admin

Dire :

> J'ai aussi ajoute une partie de supervision pour verifier que l'index, le backend vectoriel et le fournisseur LLM sont disponibles.

## 10. Mes idees et points ou je veux votre avis

Pendant la reunion, je peux demander l'avis du professeur sur ces points :

### Idee 1 : se concentrer maintenant sur le rapport et la presentation

> Comme la version est deja demonstrable, je pense limiter les grands changements techniques et me concentrer sur la redaction du rapport, la presentation et la preparation de la soutenance. Est-ce que vous validez cette priorite ?

### Idee 2 : presenter le projet comme prototype avance

> Je propose de presenter la solution comme un prototype avance et fonctionnel, avec des resultats d'evaluation honnetes. Le retrieval est fort, mais la generation reste perfectible. Est-ce une bonne formulation pour le rapport ?

### Idee 3 : garder FAISS + BM25 pour la version de demonstration

> Pour eviter d'introduire des risques avant la soutenance, je prefere garder la version actuelle basee sur FAISS + BM25 pour la demo. Les evolutions comme Qdrant peuvent etre presentees comme perspectives. Est-ce que cela vous semble pertinent ?

### Idee 4 : ameliorer la base documentaire apres la soutenance

> Les erreurs restantes viennent souvent de documents proches ou de metadata insuffisantes. Je pense proposer comme perspective l'enrichissement du corpus et l'amelioration des metadata. Est-ce que vous voulez que je detaille cette partie dans le rapport ?

### Idee 5 : evaluation dans le rapport

> J'ai prepare une evaluation sur 20 questions. Je peux l'inclure dans le rapport pour montrer la methode experimentale, avec les forces et les limites du systeme. Est-ce que je dois la mettre dans le corps du rapport ou en annexe ?

## 11. Questions possibles du professeur et reponses proposees

### Est-ce que les documents Drive ont ete exploites ?

Oui. Ils ont ete integres dans le corpus Drive, traites, decoupes en chunks, indexes et utilises pour repondre aux questions sur les services UCA.

### Quelle est la difference avec un chatbot classique ?

Un chatbot classique repond surtout a partir de son modele interne. Ici, le systeme recupere d'abord des passages depuis les documents UCA, puis repond a partir de ces passages. Cela rend la reponse plus controlee et plus justifiable.

### Est-ce que les reponses sont toujours correctes ?

Pas toujours. L'evaluation montre que le retrieval retrouve les bons documents dans 90 % des cas, mais la reponse finale atteint environ 66,7 %. Cela montre que le module est fonctionnel mais perfectible, surtout au niveau generation et qualite des chunks.

### Pourquoi utiliser LM Studio ?

LM Studio permet de tester un LLM local via une API compatible OpenAI, sans dependre obligatoirement d'un service externe. C'est pratique pour une demonstration locale, meme si la latence reste une limite.

### Pourquoi certaines reponses sont extractives ?

Quand le modele local est indisponible ou trop lent, le systeme utilise un fallback extractif. Cela garantit qu'une reponse issue des documents peut quand meme etre retournee, mais elle est parfois moins naturelle qu'une reponse generee par LLM.

### Qu'est-ce qui reste a faire ?

Les priorites restantes sont :

- ameliorer les metadata ;
- enrichir le corpus ;
- rendre les sources plus propres dans l'interface ;
- stabiliser la generation LLM ;
- finaliser le rapport et la presentation.

## 12. Conclusion orale

Pour conclure, la solution actuelle est une version demonstrable de l'UCA Digital Assistant. Elle integre une application web, une authentification, un chat, un historique, un pipeline RAG, les documents Drive, des sources, un niveau de confiance et une evaluation.

Les resultats montrent que le retrieval est deja solide, avec 90 % de documents pertinents retrouves. La generation finale reste perfectible, notamment a cause de la latence de LM Studio et de certains chunks mal alignes.

Mon objectif maintenant est de stabiliser cette version, finaliser le rapport, preparer la presentation et presenter les ameliorations restantes comme perspectives.

## 13. Reponse email courte si necessaire

Objet : Preparation de la reunion

Bonjour Professeur,

Merci pour votre retour.

J'ai bien avance sur la programmation de l'application UCA Digital Assistant. La solution integre maintenant une interface de chat etudiante, une authentification, un historique des conversations, un pipeline RAG, l'exploitation des documents Drive, l'affichage des sources et une evaluation sur des questions de reference.

Je pourrai vous presenter la version actuelle, les resultats obtenus, les limites identifiees et les prochaines etapes pour le rapport et la soutenance.

Cordialement,

Aimad Oufares
