# Guide de demonstration

## Etat actuel

- Depot : `uca_digital_assistant`
- Commit de reference actuel : `ce68ee2`
- Date de mise a jour : `2026-05-15`
- Statut : version locale de demonstration, avec application web, module RAG, evaluation Drive et contexte conversationnel.

Ce guide est une documentation technique courte. Pour la preparation orale finale, utiliser aussi :

- `reunion/plan_presentation_soutenance.md`
- `reunion/questions_demo_sures.md`
- `reunion/reponses_questions_jury.md`

## Verification avant demonstration

Commandes conseillees :

```bash
python manage.py check
python manage.py rag_healthcheck --json
python manage.py test api_app.tests --keepdb
```

Etat attendu :

- Django check : aucun probleme ;
- healthcheck : `ready=true` ;
- vector store FAISS : OK ;
- LM Studio : joignable ;
- tests applicatifs : 59 tests OK.

## Flux de demonstration recommande

1. Ouvrir l'inscription etudiante : `/signup/`.
2. Creer ou utiliser un compte avec email UCA autorise.
3. Se connecter via `/login/`.
4. Ouvrir `/chat/`.
5. Poser une question sure, par exemple :

```text
Ou consulter mes notes sur UC@Student ?
```

6. Montrer :

- la reponse ;
- les sources ;
- le niveau de confiance ;
- l'historique dans la sidebar ;
- la creation d'une nouvelle conversation.

7. Poser une question PEDOC :

```text
Comment candidater sur PEDOC ?
```

8. Montrer le contexte conversationnel avec :

```text
Et les documents necessaires ?
```

9. Ouvrir le dashboard admin : `/admin-dashboard/`.
10. Montrer l'etat RAG ou le healthcheck.

## Questions conseillees

Questions stables pour la demonstration :

- `Ou consulter mes notes sur UC@Student ?`
- `Comment candidater sur PEDOC ?`
- `A quoi sert le CIP ?`
- `Comment acceder au calcul haute performance de UCA ?`
- `Ou trouver un accompagnement pour monter un projet de recherche ?`

Questions a eviter comme premiere demonstration live :

- `A quoi sert UCAPLAT ?`
- `Comment deposer des devoirs sur UCAPLAT ?`
- `Comment suivre l'etat de mon diplome ?`

Ces questions sont utiles pour parler des limites, mais elles peuvent produire des reponses plus fragiles selon les chunks retrouves.

## Messages a porter en soutenance

- Le projet n'est pas un simple chatbot.
- Le systeme s'appuie sur une chaine RAG complete : ingestion, chunking, metadata, FAISS, BM25, retrieval, generation et sources.
- Le retrieval est le point fort : `92,31 %` de bon service en top-1 sur le benchmark Drive final.
- Le contexte conversationnel est fonctionnel : `93,75 %` de reecritures correctes.
- La generation reste perfectible, notamment a cause de la latence LM Studio sur un PC sans GPU dedie.
- La version actuelle est un prototype avance et demonstrable, pas encore une solution production institutionnelle.

