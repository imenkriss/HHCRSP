# Modèle d'optimisation HHCRSP

L'optimiseur NSGA-II minimise quatre objectifs distincts : le coût total attendu,
le nombre de patients non affectés, l'opposé du score de satisfaction et la
variance attendue de la charge de travail. Le coût total additionne les coûts de
service, d'attente, d'heures supplémentaires, de déplacement, fixes et les
pénalités de retard.

## Champs patient facultatifs

- `time_windows`: liste de couples `[début, fin]`, dans l'unité de temps du
	benchmark. Une visite doit tenir entièrement dans au moins une fenêtre. En
	l'absence de ce champ, `ready_time` et `due_date` forment l'unique fenêtre.
- `service_hours_min` et `service_hours_max`: bornes de durée en heures. Sans
	ces champs, `service_hours` est traité comme une durée fixe.
- `travel_time_min` et `travel_time_max`: bornes du temps/déplacement dans la
	même unité que le champ historique `travel_time`. Sans ces champs, sa valeur
	actuelle est fixe.

Les durées et déplacements compris entre leurs bornes sont simulés par une loi
uniforme avec un nombre fixe de scénarios (32 par défaut). La graine garantit la
reproductibilité. Ces scénarios calculent des moyennes d'objectifs; ils ne
constituent pas une garantie probabiliste de faisabilité.

## Heures supplémentaires

Le soignant peut dépasser `max_work_hours` jusqu'à `max_overtime_hours` (2 heures
par défaut). Le dépassement reste pénalisé au tarif `overtime_cost_rate` (30 par
défaut). Pour interdire les heures supplémentaires, configurer
`max_overtime_hours` à `0`.

## Limites du modèle actuel

Les fenêtres vérifient qu'une visite, considérée individuellement, peut tenir
dans une fenêtre; le projet ne construit pas encore l'ordre des visites ni les
temps de trajet entre patients. Les données benchmark existantes ne fournissent
qu'une fenêtre par patient et aucune distribution empirique des durées : les
fenêtres multiples et plages d'incertitude doivent donc être ajoutées dans les
données avant de pouvoir évaluer ces cas réels.
