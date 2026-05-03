# Guide de Demonstration

## Version de reference

- Depot : `uca_digital_assistant`
- Snapshot de reference : `aa2d14535adf54f2665da01c1114549359ca2018`
- Date de stabilisation locale : `2026-05-03`

## Flux de demonstration recommande

1. Ouvrir l'inscription etudiante : `/signup/`
2. Creer un compte avec un email UCA autorise.
3. Se connecter puis ouvrir `/chat/`.
4. Montrer :
   - le hero de l'espace etudiant
   - l'historique recent dans la sidebar
   - une nouvelle conversation
   - une question avec reponse, sources et confiance
5. Ouvrir une conversation precedente pour montrer la persistence.
6. Ouvrir `/api/health/ready/` ou `python manage.py rag_healthcheck --json`.
7. Ouvrir le dashboard admin si besoin : `/admin-dashboard/`

## Messages a porter en soutenance

- Le systeme est centre sur les etudiants UCA authentifies.
- Le chat s'appuie sur un pipeline RAG avec sources et abstention.
- Les conversations sont personnelles et persistantes.
- La version actuelle est la version stable de demonstration.
- Les evolutions SSO UCA et deploiement institutionnel restent preparees mais non imposees a cette v1.
