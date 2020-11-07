# HistoricalDictionaryExpansion

## Review notebook: harvest senses with provenance

Notebook for reviewing functions

- `get_provenance_by_semantic_class`
- `extend_from_saved_lemma_query`

all saved in `utils.dataset_download`.

These functions assume:
    - a pickled dataframe with information harvested from the OED word endpoint for a given lemma id

What these functions should do:
    - for a given lemma id (e.g. `machine_nn01` saved in pickled data)
    - get all senses
    - for each of the senses get synonyms
    - for each of the senses + synonyms, get all branches (siblings and descedants
    - keep track of the relation between the initial lemma and sense harvested (this is saved in provenance and provenance_type column
    - for more documentation please refer to the code and this notebook