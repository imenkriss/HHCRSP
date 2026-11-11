# HHCRSP-MAS — RAG + LLM + MAS + optimisation multi-objectifs

Pipeline de planification de soins à domicile (HHCOP / HHCRSP) :

```
Streamlit UI (interface/)
  -> RAG retriever (rag/retriever.py, base de connaissances connaissance/)
  -> LLM reasoner (llm/raisonner.py, Ollama — fallback déterministe hors ligne)
  -> agents (agents/agentOrchestrateur.py, agentsPatients.py, agentsSoignants.py)
  -> optimisation multi-objectifs (optimisation/nsga2.py, optimisation/mtwmo/)
  -> moteur de décision (décision/decision.py)
  transversal : métriques temps/mémoire et qualité de Pareto (complexity/metrics.py)
```

## 1. Jeu de données — contrat et intégrité

Le fichier source des patients est **`data/dementia/new_dataset_with_predictions.csv`**
(1000 lignes, 26 colonnes, features normalisées min-max, `XCOORD.`/`YCOORD.`,
`Predictions` = type de soin).

| Fichier | Rôle | Lignes | Écrit par le code ? |
|---|---|---:|---|
| `data/dementia/new_dataset_with_predictions.csv` | patients (source de vérité) | 1000 | **jamais** (lecture seule) |
| `data/admin_caregivers.csv` | soignants (facultatif) | 0 (vide) | **jamais** |
| `data/dementia_patients_health_data.csv` | ancien placeholder 1 ligne | 1 | **jamais** (conservé pour traçabilité) |

- Le fichier `data/dementia_patients_health_data.csv` (81 octets, 1 enregistrement)
  et le CSV soignants vide sont d'anciens placeholders. Ils ne sont **plus** lus
  comme source : c'est ce qui faisait apparaître un « dataset d'une seule ligne ».
- Le code ne crée plus aucun fichier sous `data/`. Les artefacts générés vont dans
  `data/generated/` (cache d'embeddings) ou `outputs/` (exports), jamais dans la
  racine des données.
- Chaque exécution prend un **empreinte sha256 + nombre de lignes + colonnes**
  avant et après le pipeline et lève `DatasetIntegrityError` si un fichier source
  a changé.
- `patient_limit` (défaut **100**) limite seulement le **jeu de travail** d'une
  exécution ; `patient_limit=None` charge les 1000 lignes. Le fichier reste
  intact dans tous les cas.

Coordonnées : `XCOORD.`/`YCOORD.` sont normalisées dans `[0, 1]` et remises à
l'échelle d'une zone de 50 × 50 (défaut), 100 × 100 ou 200 × 200 avant le calcul
des distances.

## 2. Installation (Windows, Python 3.11, CPU uniquement)

```powershell
conda activate base
python -m pip install -r requirements.txt
```

Dépendances principales : `pandas`, `numpy`, `scikit-learn`, `streamlit`,
`pydantic`, `psutil`, `ollama` (client), `pytest`. `pymoo` est optionnel
(fallback Python pour l'hypervolume). Aucune dépendance CUDA/GPU.

## 3. Exécution

### Interface Streamlit

```powershell
python -m streamlit run interface/app.py
# ou en mode headless pour un accès réseau local :
python -m streamlit run interface/app.py --server.headless true --server.port 8501
```

L'interface permet de choisir la source (teacher CSV ou instance générée A–F),
la taille du jeu de travail, la population, les générations, l'algorithme, la
règle de décision, le budget d'incertitude Γ et la comparaison mesurée
NSGA-II vs MTCMOLite. Chaque tableau est filtrable, triable, paginé et
exportable en CSV/JSON (`outputs/`).

### Ligne de commande

```powershell
python main.py "Get the info for patient P1 with high priority"
```

### Benchmarks

```powershell
# P10 : NSGA-II vs MTCMOLite, 5 répétitions, instances A/B
python experiments/run_benchmark.py --datasets A B --runs 5

# D3 : SVM + K-Means + Tabu vs K-Means + Tabu (écrit ml/results/ml_benchmark.csv)
python -m ml.evaluation
```

### Ollama (optionnel)

Le raisonneur n'appelle Ollama que si `USE_OLLAMA=1` :

```powershell
$env:USE_OLLAMA = "1"
ollama pull qwen3:4b
python main.py "Patient P4 is urgent and caregiver S2 is delayed"
```

Sans Ollama, le fallback déterministe et validé par le schéma Pydantic est
utilisé (`reasoning_source == "fallback"`) : les tests et l'interface
fonctionnent hors ligne.

## 4. Comparaisons affichées

Aucune valeur n'est codée en dur. L'onglet « Comparaison (mesurée) » contient :

1. **Solutions du run** — une ligne par solution non dominée (`item_id` S01…),
   composantes Z1, Z2, Z3, patients servis, soignants utilisés.
2. **Configurations exécutées** — une ligne par exécution de la session
   (jeu de données, tailles, algorithme, modèle, objectifs, temps, mémoire).
3. **NSGA-II vs MTCMOLite** — les deux algorithmes exécutés sur la même instance
   générée et les mêmes graines, agrégés comme en P10. `MTCMOLite` est une
   ré-implémentation du principe de Qiao et al. (2022), jamais présentée comme
   l'algorithme original.
4. **Baseline ML (D3)** — lue depuis `ml/results/ml_benchmark.csv` si elle
   existe ; sinon un message indique comment la produire.

Une source absente est signalée explicitement : aucune ligne factice n'est
ajoutée pour « remplir » un tableau.

## 5. Tests

```powershell
python -m pytest -q
```

État vérifié sur cette machine : **258 tests passent** (≈ 16 s), dont :

| Test | Fichier | Vérifie |
|---|---|---|
| A — intégrité | `tests/test_dataset_integrity.py` | le fichier source est inchangé (sha256, 1000 lignes) après un run |
| B — table multi-lignes | `tests/test_comparison_tables.py` | 3 enregistrements ⇒ 3 lignes, ids stables, exports CSV/JSON |
| C — exécutions répétées | `tests/test_dataset_integrity.py` | pas de perte, pas de doublons, cache d'embeddings réutilisé |
| D — RAG | `tests/test_rag_cache.py` | ingestion, chunks, persistance des embeddings, ids/métadonnées |
| E — entrées invalides | `tests/test_dataset_integrity.py` | fichiers absents/vides/malformés ⇒ erreurs explicites |
| F — performance | `tests/test_performance.py` | temps et pic mémoire mesurés sur un jeu représentatif |

## 6. Performances mesurées (Intel i5-3320M, 4 threads, 10 Go, CPU seul)

| Configuration | Patients | Soignants | Front | Temps | Pic Python | RSS |
|---|---:|---:|---:|---:|---:|---:|
| UI par défaut (pop 12 / gén 5) | 50 | 13 | 9 | 0,88 s | 2,2 Mo | 95 Mo |
| CLI par défaut (pop 20 / gén 10) | 100 | 25 | 14 | 5,56 s | 2,3 Mo | 98 Mo |
| Chargement complet | 1000 | 250 | — | 0,24 s | 1,4 Mo | 94 Mo |
| Comparaison A_2_1 (2 algo × 2 runs) | 10 | 2 | 65 / 80 | 1,17 s | — | — |

Le coût dominant est l'évaluation NSGA-II (`O(G × N × S × (P + C))`), pas le RAG
ni les agents. Les embeddings sont mis en cache sur disque : une deuxième
exécution ne ré-embed rien (`hits == nombre de chunks`).

## 7. Limites connues

- Le CSV enseignant n'a ni durée de service ni niveau de compétence requis :
  `service_hours = 0.5` et `required_skill_level = 1` sont des hypothèses
  documentées (`docs/ASSUMPTIONS.md`).
- Les 700 lignes sans `Predictions` reçoivent `care_type = "General"` ; la
  complétion SVM reste disponible via `data.loader.load_predictions_dataset`
  (D1/D3) mais n'est pas exécutée à chaque run.
- `data/admin_caregivers.csv` est vide : les soignants par défaut sont un
  échafaudage déterministe (compétences 1–3, 8 h/jour), pas une donnée réelle.
- L'algorithme principal de l'UI est l'échantillonneur d'affectations
  `optimisation/nsga2.py` ; `MTCMOLite` est disponible pour les instances
  générées. Aucun des deux n'est présenté comme le NSGA-II canonique de Deb.
- Le `MTCMO` original de Qiao et al. (2022) n'est pas reproduit à l'identique.

Détail complet des réparations, mesures et décisions : `docs/REPAIR_REPORT.md`.
