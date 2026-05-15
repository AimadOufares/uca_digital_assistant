# Questions / Réponses de référence - corpus Drive

Objectif : utiliser ces questions pour interroger le module RAG, puis comparer sa sortie avec la réponse de référence.  
Nombre de questions : 20, couvrant les principales plateformes présentes dans `data_storage/raw/drive`.

Note de test : LM Studio est joignable via `http://127.0.0.1:1234/v1`. Après réduction du prompt compact, seules les questions précédemment en abstention ont été retestées avec LM Studio : Q10 a produit une génération LLM, tandis que Q16 a encore expiré et est restée en fallback extractif.

| # | question | repense | retrieval_doc_pertinent | repense_module_rag |
|---|---|---|---|---|
| 1 | À quoi sert la plateforme UC@Student ? | UC@Student est une plateforme institutionnelle destinée aux étudiants de l'Université Cadi Ayyad. Elle centralise les services académiques et administratifs : emploi du temps, résultats, demandes administratives, suivi des diplômes, notifications et accès mobile via l'application Android. | Oui - UC@Student | D'apres les informations retrouvees dans les documents UCA : - La plateforme est complétée par une application mobile Android dédiée aux étudiants, qui leur permet d’accéder, depuis leur smartphone, aux principales fonctionnalités de UC@Student (emploi du temps, résultats, demandes administratives, notifications, suivi des diplômes, etc.), offrant ainsi une expérience mobile moderne, continue et accessible à tout moment. - Espace d’administration UC@Student L’espace d’administration permet aux établissements de : Gérer l’ensemble des modules de la plateforme Suivre les activités des étudiants Mettre à jour les données académiques Superviser les demandes et les délais de traitement Disposer d’outils de pilotage et de suivi Il constitue un outil opérationnel quotidien pour les services de scolarité et les responsables administratifs. - Objectifs de la plateforme |
| 2 | Quels services administratifs un étudiant peut-il demander via UC@Student ? | UC@Student permet de centraliser les demandes administratives des étudiants, notamment les attestations, les relevés de notes et d'autres documents administratifs. L'étudiant peut suivre l'état de la demande : en attente, en cours de traitement ou traitée. | Oui - UC@Student | D'apres les informations retrouvees dans les documents UCA : - 🔹 Diplômes Suivi administratif des diplômes délivrés Gestion des différentes étapes du processus : En traitement Prêt Retiré Mise à jour en temps réel de l’état d’avancement Consultation du statut par l’étudiant Facilitation du travail des services administratifs Ce module garantit transparence, fiabilité et organisation dans la gestion des diplômes. - Elle constitue un point d’accès unique pour les étudiants et un outil de gestion intégré pour les services administratifs des établissements, permettant une communication fluide, une réduction des procédures papier et une amélioration significative de la qualité de service. - # Votre portail numérique de la scolarité Uc@Student Accédez à tous vos services académiques et administratifs en un seul endroit. |
| 3 | Comment UC@Student gère-t-elle les résultats académiques ? | Les résultats sont importés par l'administration, organisés par matière, semestre et année universitaire, validés avant publication, puis publiés de manière sécurisée dans l'espace personnel de l'étudiant. | Oui - UC@Student | D'apres les informations retrouvees dans les documents UCA : - Espace d’administration UC@Student L’espace d’administration permet aux établissements de : Gérer l’ensemble des modules de la plateforme Suivre les activités des étudiants Mettre à jour les données académiques Superviser l ... - 🔹 Résultats académiques Importation des résultats académiques par l’administration Organisation des résultats par : matière année universitaire Validation avant publication Publication sécurisée des résultats Accès confidentiel via l’espace personnel de l’étudiant Rapidité de diffusion et respect de la confidentialité des données. - ### Consultation des Notes Suivez vos résultats académiques et visualisez votre progression tout au long de l'année |
| 4 | À quoi sert l'espace Diplômes de UC@Student ? | L'espace Diplômes permet de gérer et suivre le processus de délivrance des diplômes universitaires. Il couvre les étapes depuis le traitement du dossier jusqu'à la délivrance finale, avec traçabilité, historique, bordereaux et consultation de la situation du diplôme par l'étudiant, y compris sur mobile. | Oui - UC@Student | D'apres les informations retrouvees dans les documents UCA : - La plateforme est complétée par une application mobile Android dédiée aux étudiants, qui leur permet d’accéder, depuis leur smartphone, aux principales fonctionnalités de UC@Student (emploi du temps, résultats, demandes administratives, notifications, suivi des diplômes, etc.), offrant ainsi une expérience mobile moderne, continue et accessible à tout moment. - 🔹 Diplômes Suivi administratif des diplômes délivrés Gestion des différentes étapes du processus : En traitement Prêt Retiré Mise à jour en temps réel de l’état d’avancement Consultation du statut par l’étudiant Facilitation du travail des services administratifs Ce module garantit transparence, fiabilité et organisation dans la gestion des diplômes. - ### Demandes Administratives Effectuez vos demandes de diplômes, relevés de notes et certificats en ligne |
| 5 | Quelles sont les principales étapes de suivi d'un diplôme dans l'espace Diplômes ? | Le workflow peut inclure le traitement du fond du dossier, l'édition du diplôme, la signature du Doyen, l'envoi et l'arrivée à la Présidence, les corrections éventuelles, la signature par le Président, le retour à l'établissement, le statut diplôme prêt puis diplôme délivré. | Oui - Espace Diplômes | D'apres les informations retrouvees dans les documents UCA : - ✔ Centraliser le suivi institutionnel des diplômes pour tous les établissements - 🔸 Au niveau des établissements Consultation des diplômés par filière et par année Suivi de l’état d’avancement des diplômes Génération automatique des bordereaux par établissement Envoi des diplômes à la Présidence avec référence du bordereau - Objectifs de l’espace Diplômes |
| 6 | À quoi sert la plateforme PEDOC ? | PEDOC est le portail doctoral de l'UCA. Il sert à gérer et suivre les études doctorales : inscription et réinscription, dossier doctoral, encadrement, formations doctorales, soutenances, évaluations, notifications et administration des structures doctorales. | Oui - PEDOC | D'apres les informations retrouvees dans les documents UCA : - La plateforme PEDOC facilite la communication entre les doctorants, les encadrants, les formations doctorales, les centres d’études doctorales et l’administration universitaire. - Espace d’administration pour la gestion du système La plateforme PEDOC dispose d’un back-office administratif permettant aux responsables institutionnels de : • Gérer les comptes des doctorants, encadrants et responsables académiques. - Objectifs de la plateforme |
| 7 | Quelles fonctionnalités PEDOC propose-t-elle pour les doctorants ? | PEDOC permet l'inscription ou la réinscription en ligne, le suivi du dossier doctoral, la consultation de l'état d'avancement du parcours, le suivi scientifique avec les encadrants, les demandes de soutenance et les notifications liées aux validations ou échéances. | Oui - PEDOC | D'apres les informations retrouvees dans les documents UCA : - La plateforme PEDOC facilite la communication entre les doctorants, les encadrants, les formations doctorales, les centres d’études doctorales et l’administration universitaire. - Espace d’administration pour la gestion du système La plateforme PEDOC dispose d’un back-office administratif permettant aux responsables institutionnels de : • Gérer les comptes des doctorants, encadrants et responsables académiques. - ✔ Faciliter la communication entre doctorants, encadrants, st ... |
| 8 | Quel est le rôle de la plateforme CIP ? | La plateforme CIP, Centre d'Innovation Pédagogique, informe et communique autour des activités pédagogiques de l'UCA. Elle présente les missions du centre, les actualités, événements, formations, ressources pédagogiques, modules, capsules et supports d'accompagnement aux outils numériques. | Oui - CIP | D'apres les informations retrouvees dans les documents UCA : - Objectifs de la plateforme - ✔ Offrir un espace numérique d’information sur les formations, ressources pédagogiques et événements structurés par le CIP. - La plateforme sert à informer, communiquer et gérer les activités pédagogiques du centre, tout en offrant des ressources, des annonces et un espace pour consulter les événements et les formations liés à l’innovation pédagogique. |
| 9 | Quelles ressources trouve-t-on sur la plateforme CIP ? | On peut y trouver des modules de formation, des capsules pédagogiques, des documents de méthodologie, des supports d'accompagnement à l'usage des outils numériques, ainsi que des annonces, actualités et événements liés à l'innovation pédagogique. | Oui - CIP | D'apres les informations retrouvees dans les documents UCA : - ✔ Offrir un espace numérique d’information sur les formations, ressources pédagogiques et événements structurés par le CIP. - La plateforme sert à informer, communiquer et gérer les activités pédagogiques du centre, tout en offrant des ressources, des annonces et un espace pour consulter les événements et les formations liés à l’innovation pédagogique. - Objectifs de la plateforme |
| 10 | À quoi sert UCAPLAT ? | UCAPLAT est une plateforme pédagogique numérique de l'UCA destinée à l'enseignement en ligne. Elle permet la gestion des cours, ressources, activités pédagogiques, devoirs et interactions entre enseignants et étudiants dans un cadre d'apprentissage numérique. | Oui - UCAPLAT | UCAPLAT est une plateforme destinée à centraliser et structurer les demandes d’analyses scientifiques, optimiser l’utilisation des équipements scientifiques mutualisés et offrir un suivi en temps réel de l’état de la demande. Elle permet aux Enseignant-Chercheurs, Doctor _(test LM Studio, 199.8s, g?n?ration LLM) |
| 11 | Que permet la plateforme des Clubs des étudiants ? | Elle centralise les informations sur les clubs étudiants de l'UCA, valorise leurs activités, affiche les événements et permet aux clubs de gérer leur profil, proposer des activités et suivre leur validation par la Présidence avant publication officielle. | Non - Clubs des étudiants | D'apres les informations retrouvees dans les documents UCA : - Fiche de présentation – Plateforme du Club de l’Université Cadi Ayyad (Club UCA) 1. - La plateforme Club UCA a pour objectif de digitaliser la gestion du club universitaire, qui propose des activités sportives, des infrastructures de loisirs, des piscines, des terrains de sport et des services d’hébergement, réservés au personnel de l’université disposant d’un abonnement annuel. - Objectifs de la plateforme Présenter le Club de l’Université et ses services Offrir aux adhérents un accès en ligne aux réservations Centraliser la gestion des créneaux, des infrastructures et de l’hébergement Réduire la charge administrative et améliorer la traçabilité Améliorer l’expérience utilisateur des adhérents et des gestionnaires du club 3. |
| 12 | Comment les activités des clubs étudiants sont-elles validées ? | Les événements et activités ajoutés par les clubs passent par un workflow de validation institutionnelle par la Présidence. Ils peuvent être validés, rejetés ou renvoyés pour modification avant publication officielle. | Non - Clubs des étudiants | D'apres les informations retrouvees dans les documents UCA : - La plateforme Club UCA a pour objectif de digitaliser la gestion du club universitaire, qui propose des activités sportives, des infrastructures de loisirs, des piscines, des terrains de sport et des services d’hébergement, réservés au personnel de l’université disposant d’un abonnement annuel. |
| 13 | À quoi sert PUCA Staff ? | PUCA Staff est une plateforme interne de gestion administrative et RH du personnel de la Présidence de l'UCA. Elle digitalise la présence, les congés, absences, autorisations d'absence, heures supplémentaires, pointage, validation hiérarchique et suivi des demandes. | Oui - PUCAStaff | D'apres les informations retrouvees dans les documents UCA : - Fiche descriptive – Plateforme PUCA Staff Nom de la plateforme PUCA Staff – Plateforme de gestion administrative du personnel URL https://pucastaff.uca.ma/ Description générale PUCA Staff est une plateforme numérique interne destinée à la gestion administrative et RH des fonctionnaires de la Présidence de l’Université Cadi Ayyad. |
| 14 | Quel est le circuit de validation d'une demande sur PUCA Staff ? | Une demande suit deux niveaux de validation : d'abord par le Chef de division ou Chef de service, puis par la Direction des Ressources Humaines. Ce circuit assure la conformité administrative et le respect de la hiérarchie. | Oui - PUCAStaff | D'apres les informations retrouvees dans les documents UCA : - Elle permet de digitaliser l’ensemble des processus liés à la présence, aux congés, aux absences et aux heures supplémentaires, tout en assurant un circuit de validation hiérarchique structuré et une traçabilité complète des demandes. - Fiche descriptive – Plateforme PUCA Staff Nom de la plateforme PUCA Staff – Plateforme de gestion administrative du personnel URL https://pucastaff.uca.ma/ Description générale PUCA Staff est une plateforme numérique interne destinée à la gestion administrative et RH des fonctionnaires de la Présidence de l’Université Cadi Ayyad. - La plateforme est accessible aussi bien via une application web que via une application mobile Android, facilitant son usage au quotidien par les fonctionnaires. |
| 15 | À quoi sert la plateforme Mobilité Internationale ? | La plateforme Mobilité Internationale centralise les informations et démarches liées à la mobilité académique internationale pour étudiants, enseignants-chercheurs et personnel administratif : programmes Erasmus+, échanges, doubles diplômes, bourses, appels à candidatures et suivi des demandes. | Oui - Mobilité internationale | D'apres les informations retrouvees dans les documents UCA : - Cette plateforme constitue un guichet numérique unique pour l’ensemble des démarches de mobilité internationale, permettant aux utilisateurs d’accéder aux différentes opportunités de mobilité (ex : Erasmus+, doubles diplômes, accords d’échange) ainsi qu’à des services de suivi et de traitement des demandes. - 🔹 Espace d’administration pour la gestion des demandes La plateforme intègre un espace réservé aux gestionnaires administratifs des services en charge de la mobilité internationale : Consultation et tri des demandes reçues par type de programme ou statut ; Validation, rejet ou demande de complément d’information pour chaque candidature ; Communication interne avec les services concernés ou avec les candidats ; Archivage et suivi statistique des cand ... - ✔ Simplifier l’accès aux opportunités de mobilité internationale pour les membres de l’université. |
| 16 | Comment suivre une candidature de mobilité internationale ? | Après soumission, l'utilisateur dispose d'un espace personnalisé pour suivre le statut de sa demande, les documents requis ou manquants, les étapes ultérieures et les notifications envoyées par les services administratifs responsables. | Oui - Mobilité internationale, PEDOC | D'apres les informations retrouvees dans les documents UCA : - 🔹 Espace d’administration pour la gestion des demandes La plateforme intègre un espace réservé aux gestionnaires administratifs des services en charge de la mobilité internationale : Consultation et tri des demandes reçues par type de programme ou statut ; Validation, rejet ou demande de complément d’information pour chaque candidature ; Communication interne avec les services concernés ou avec les candidats ; Archivage et suivi statistique des cand ... - 🔹 Suivi des demandes Après soumission, l’utilisateur dispose d’un espace personnalisé pour suivre l’état de traitement de ses demandes de mobilité. - ✔ Centraliser et structurer les démarches de candidature et de gestion des mobilités. _(test LM Studio, 161.0s, fallback) |
| 17 | À quoi sert la plateforme Soutien à la Recherche ? | Elle permet aux chercheurs, doctorants et enseignants-chercheurs de déposer et suivre des demandes de soutien liées à la recherche : soutenance de thèse, diffusion et valorisation, organisation d'événements scientifiques, co-supervision internationale, mobilité scientifique, dépôt et valorisation de brevet. | Oui - Soutien-Recherche | D'apres les informations retrouvees dans les documents UCA : - La plateforme regroupe les demandes de soutien en cohérence avec les missions de la Vice-Présidence chargée de la Recherche et de la Coopération, et s’intègre au sein de l’écosystème numérique de la recherche à l’UCA en offrant un accès simple, structuré et multilingue aux services de support disponibles. - Elle a pour objectif de proposer un ensemble de formulaires en ligne permettant aux chercheurs, doctorants et enseignants-chercheurs de déposer des demandes de soutien institutionnel et administratif pour diverses étapes et activités de leurs projets de recherche. - ✅ Accès centralisé à l’ensemble des formulaires de soutien pour la recherche. |
| 18 | Quels avantages apporte la plateforme Soutien à la Recherche ? | Elle centralise les formulaires de soutien, réduit les démarches papier, standardise les demandes, facilite le traitement administratif et assure la traçabilité ainsi que le suivi numérique des demandes. | Oui - Soutien-Recherche | D'apres les informations retrouvees dans les documents UCA : - La plateforme regroupe les demandes de soutien en cohérence avec les missions de la Vice-Présidence chargée de la Recherche et de la Coopération, et s’intègre au sein de l’écosystème numérique de la recherche à l’UCA en offrant un accès simple, structuré et multilingue aux services de support disponibles. - Elle a pour objectif de proposer un ensemble de formulaires en ligne permettant aux chercheurs, doctorants et enseignants-chercheurs de déposer des demandes de soutien institutionnel et administratif pour diverses étapes et activités de leurs projets de recherche. - ✅ Accès centralisé à l’ensemble des formulaires de soutien pour la recherche. |
| 19 | À quoi sert la plateforme HPC UCA ? | HPC UCA fournit un accès centralisé et sécurisé aux ressources de calcul haute performance de l'université. Elle permet aux doctorants et enseignants-chercheurs de demander un accès, soumettre des jobs, suivre leur état et télécharger les résultats. | Oui - HPC UCA | D'apres les informations retrouvees dans les documents UCA : - Fiche de présentation – Plateforme HPC UCA 1. - HPC@UCA est conçue pour fournir un accès centralisé et sécurisé aux clusters de calcul et autres ressources informatiques avancées pour les besoins scientifiques et techniques des chercheurs de l’université. - Validation des droits pour accéder aux clusters HPC. |
| 20 | Quelles informations la plateforme Centre de Conférences UCA met-elle à disposition ? | Elle présente le Centre de Conférences, ses services et atouts, les événements passés ou à venir, une galerie photo, des pages À propos et Nos Atouts, ainsi qu'une page de contact pour les demandes d'information ou d'organisation d'événements. | Oui - Centre de conf?rences | D'apres les informations retrouvees dans les documents UCA : - Fiche de présentation – Plateforme Centre-de-Conférences UCA 1. - Objectifs de la plateforme La plateforme vise à : Présenter le Centre de Conférences, ses atouts et ses services aux visiteurs et organisateurs potentiels. - La plateforme sert de vitrine numérique pour le Centre de Conférences de l’Université Cadi Ayyad, un espace dédié à l’organisation de conférences, séminaires, colloques et événements académiques ou institutionnels au sein de l’université. |

## Comparaison et ?valuation

Bar?me utilis? :

- `3/3` : r?ponse correcte et suffisamment compl?te.
- `2/3` : r?ponse globalement correcte, mais incompl?te ou trop bruit?e.
- `1/3` : r?ponse partiellement li?e au sujet, mais insuffisante ou contenant du contexte hors sujet.
- `0/3` : r?ponse non disponible ou hors sujet.

| # | score | comparaison | evaluation |
|---|---:|---|---|
| 1 | 2/3 | La r?ponse retrouve UC@Student et plusieurs fonctionnalit?s, mais elle reste trop orient?e mobile/administration et ne synth?tise pas tous les services ?tudiants. | Correcte mais trop extractive. |
| 2 | 1/3 | La r?f?rence attend les attestations, relev?s de notes et autres documents administratifs. Le module r?pond surtout avec un extrait sur les dipl?mes et une phrase g?n?rale sur le portail. | Partielle, manque les ?l?ments cl?s. |
| 3 | 3/3 | La r?ponse r?cup?re les ?l?ments essentiels : importation, organisation des r?sultats, validation avant publication, publication s?curis?e et acc?s confidentiel. | Bonne r?ponse. |
| 4 | 2/3 | Le module mentionne l'espace dipl?mes, les statuts et l'acc?s mobile, mais ne couvre pas clairement tout le cycle institutionnel et les bordereaux. | Correcte mais incompl?te. |
| 5 | 1/3 | La r?ponse parle du suivi institutionnel et des bordereaux, mais ne liste pas les ?tapes pr?cises attendues du workflow. | Trop g?n?rale. |
| 6 | 2/3 | La r?ponse identifie PEDOC et son r?le de communication/administration doctorale, mais ne cite pas explicitement toutes les fonctionnalit?s attendues. | Correcte mais incompl?te. |
| 7 | 1/3 | Le module reste sur une description g?n?rale de PEDOC et ne d?taille pas assez les fonctionnalit?s c?t? doctorant. | Faible couverture. |
| 8 | 3/3 | La r?ponse indique que le CIP informe, communique et g?re les activit?s p?dagogiques, formations, ressources et ?v?nements. | Bonne r?ponse. |
| 9 | 2/3 | La r?ponse mentionne les ressources, formations et ?v?nements du CIP, mais ne donne pas tous les exemples attendus comme capsules, modules ou documents m?thodologiques. | Pertinente mais incompl?te. |
| 10 | 1/3 | La correction ?vite l'abstention et retrouve un chunk UCAPLAT, mais le contenu r?cup?r? parle surtout de demandes d'analyses scientifiques, pas de la plateforme p?dagogique attendue. | Am?lioration du retrieval, r?ponse encore mal align?e. |
| 11 | 1/3 | La r?ponse ne part plus vers Mobilit?/UC@Student, mais elle r?cup?re surtout la fiche Club UCA r?serv?e au personnel, pas la plateforme des clubs ?tudiants. | Moins hors sujet qu'avant, mais encore incorrect. |
| 12 | 0/3 | La r?ponse ne d?crit pas le workflow de validation par la Pr?sidence ; elle parle du Club UCA et de services d'h?bergement/sport. | Hors sujet pour cette question. |
| 13 | 2/3 | La r?ponse identifie correctement PUCA Staff et la gestion administrative/RH, avec beaucoup moins de bruit qu'avant. | Correcte mais concise. |
| 14 | 2/3 | Le module mentionne le circuit de validation hi?rarchique et la tra?abilit?, mais ne pr?cise pas clairement les deux niveaux chef de division/service puis RH. | Correcte mais impr?cise. |
| 15 | 3/3 | La r?ponse couvre le guichet mobilit?, les opportunit?s internationales, Erasmus+, doubles dipl?mes, accords d'?change et suivi des demandes. | Bonne r?ponse. |
| 16 | 3/3 | La r?ponse n'est plus en abstention et mentionne l'espace personnalis? de suivi des demandes de mobilit? apr?s soumission. | Correction r?ussie. |
| 17 | 3/3 | La r?ponse d?crit la finalit? de Soutien ? la Recherche, les demandes de soutien, les formulaires et l'acc?s centralis?. | Bonne r?ponse. |
| 18 | 2/3 | La r?ponse cite l'acc?s centralis? et les formulaires, mais ne couvre pas tous les avantages attendus comme r?duction du papier, standardisation et tra?abilit?. | Correcte mais incompl?te. |
| 19 | 3/3 | La r?ponse mentionne HPC UCA, l'acc?s centralis? et s?curis? aux clusters et la validation des droits. | Bonne r?ponse. |
| 20 | 3/3 | Le module identifie correctement le Centre de Conf?rences et cite pr?sentation, atouts, services, ?v?nements et r?le de vitrine. | Bonne r?ponse. |

### Synth?se chiffr?e

- Score total : `40/60`, soit `66,7 %`.
- Bonnes r?ponses (`3/3`) : 7/20, soit `35 %`.
- R?ponses acceptables (`2/3` ou plus) : 14/20, soit `70 %`.
- R?ponses faibles ou incorrectes (`0/3` ou `1/3`) : 6/20, soit `30 %`.
- Abstentions incorrectes : aucune sur les questions test?es apr?s correction.
- R?ponses encore hors sujet : surtout la question 12, et partiellement les questions 10 et 11.
- Documents pertinents retrouv?s par le retrieval : `18/20`, soit `90 %`.
- Documents non pertinents ou confondus au retrieval : questions 11 et 12, li?es ? la confusion entre `Club UCA` et `Clubs des ?tudiants`.
- Cas particulier : question 10 retrouve un document `UCAPLAT`, mais le chunk s?lectionn? ne correspond pas bien ? la r?ponse p?dagogique attendue.

### Analyse qualitative

La modification l?g?re am?liore le comportement du retrieval sans changer l'architecture du module. Les alias suppl?mentaires permettent au syst?me d'identifier davantage de services explicites, et l'assouplissement cibl? des garde-fous ?vite de rejeter un chunk lorsqu'il correspond clairement au service demand?.

Les am?liorations les plus visibles concernent UCAPLAT et Mobilit? Internationale. UCAPLAT ne tombe plus en abstention et a pu ?tre test? avec une g?n?ration LM Studio, mais le contenu r?cup?r? reste mal align? avec la r?ponse de r?f?rence. Mobilit? Internationale retrouve maintenant l'information de suivi des demandes ; lors du retest LM Studio, la g?n?ration a expir? et le module a conserv? son fallback extractif.

Il reste toutefois des limites : certaines r?ponses sont encore trop extractives, et le cas Clubs des ?tudiants reste fragile car le retrieval r?cup?re une fiche proche mais diff?rente, li?e au Club UCA. Cela indique qu'une prochaine am?lioration devrait porter sur la qualit? des m?tadonn?es `service_name`/`file_name` et sur la s?paration entre `Club UCA` et `Clubs des ?tudiants`.

### Conclusion

La correction a un impact limit? sur le code, mais am?liore le score global de `33/60` ? `40/60`. Le module devient plus robuste pour les questions contenant un service explicite, sans modification lourde du pipeline RAG. Pour aller plus loin, il faudra surtout nettoyer ou enrichir les m?tadonn?es des documents Drive et stabiliser la g?n?ration LLM afin de remplacer les r?ponses fallback extractives par des r?ponses synth?tiques.

## Test final du 15/05/2026

Objectif : verifier le comportement final apres les corrections legeres sur les alias, les metadonnees et les garde-fous de retrieval.

### Benchmark Drive final

Commande utilisee :

```bash
python -m rag_module.evaluation.evaluate_rag --benchmark drive --top-k 5 --skip-generation
```

Le test a ete lance sans generation LLM afin d'evaluer principalement le retrieval et d'eviter la latence de LM Studio.

Rapport genere :

- JSON : `data_storage/reports/rag_eval_drive_20260515_161849.json`
- TXT : `data_storage/reports/rag_eval_drive_20260515_161849.txt`

Resultats :

| Metrique | Resultat |
|---|---:|
| Questions testees | 13 |
| Service top-1 accuracy | 92,31 % |
| Hit@k rate | 61,54 % |
| Precision@k moyenne | 48,72 % |
| Coverage@k moyenne | 56,28 % |
| Dense hit@k | 76,92 % |
| BM25 hit@k | 84,62 % |
| Abstention rate | 0 % |
| Latence retrieval moyenne | 1709,44 ms |

Lecture du resultat :

- Le module retrouve le bon service en premiere position dans `12/13` cas.
- Les cas UCAPLAT restent bien rattaches au service UCAPLAT apres correction.
- Le cas encore fragile est `Espace Diplomes`, ou la source retrouvee est correcte mais le service top-1 reste confondu avec `UC@Student`.
- Le test confirme que la regle stricte sur les services explicites reduit les confusions sans casser le benchmark global.

### Test de contexte conversationnel avec 5 questions

Scenario teste :

| # | Question | Requete reecrite | Service top-1 | Resultat |
|---|---|---|---|---|
| 1 | A quoi sert UCAPLAT ? | A quoi sert UCAPLAT ? | UCAPLAT | OK |
| 2 | Comment deposer des devoirs ? | Comment deposer des devoirs ? Concernant devoirs sur UCAPLAT. | UCAPLAT | OK |
| 3 | Et pour les cours ? | Et pour les cours ? Concernant cours sur UCAPLAT. | UCAPLAT | OK |
| 4 | Comment candidater sur PEDOC ? | Comment candidater sur PEDOC ? | PEDOC | OK |
| 5 | Et les documents necessaires ? | Quels documents sont necessaires pour candidature sur PEDOC ? | PEDOC | OK |

Resultat : `5/5` questions contextuelles correctes.

Observation importante : le premier passage du test a revele que la question `Et pour les cours ?` gardait l'ancien sujet `devoirs`. La correction appliquee donne maintenant priorite a l'intention detectee dans la nouvelle question, ce qui permet de reecrire correctement vers `cours sur UCAPLAT`.

### Evaluation finale apres test

Le module est maintenant plus stable pour la demonstration :

- les questions avec un service explicite sont mieux controlees ;
- les confusions UCAPLAT/PEDOC/HPC sont reduites ;
- le contexte conversationnel fonctionne sur un scenario court et realiste ;
- les limites restantes sont surtout liees a la qualite du corpus et a certaines metadonnees historiques.

Conclusion pour la presentation : le projet peut etre presente comme un prototype RAG avance, avec un retrieval robuste sur les services explicites et une gestion contextuelle fonctionnelle, tout en gardant comme perspective l'amelioration du corpus et de la generation LLM.

## Sources Drive analysées

- `Fiche de présentation UC@Student.docx`
- `V2 - Fiche Espace de suivi des diplômes.docx`
- `Fiche Plateforme PEDoc.docx`
- `Fiche Plateforme CIP Exemple.docx`
- `V2 - Fiche de la Plateforme CIP.docx`
- `V2 - Fiche de la Plateforme UCAPLAT (2).docx`
- `V2 - Fiche de présentation Clubs des étudiants.docx`
- `V2 - Fiche descriptive PUCAStaff.docx`
- `V2 - Fiche Plateforme mobilite-internationale (1).docx`
- `V2 - Fiche Plateforme Soutien-Recherche (1).docx`
- `Fiche de présentation HPC UCA.docx`
- `Fiche de présentation Centre de conférences.docx`
- `Fiche de présentation Club UCA.docx`
- `V2 - Fiche Plateforme Appels à Projets.docx`
