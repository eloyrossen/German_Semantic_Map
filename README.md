# German Semantic Map

A Python project that builds a semantic map of the German vocabulary I have been learning and collecting in my personal German–Spanish dictionary. Using the word vectors of spaCy's German model, it measures how similar the words in the dictionary are to each other and, given a word, returns its closest neighbours.

## How it works

1. **Reads the dictionary** (`diccionario_aleman.txt`), where each line follows the format `German word = Spanish definition`.
2. **Parses each line with a regular expression** that separates the article (*der/die/das*), the reflexive *sich*, the lemma, the plural and the definition, so that only the bare word is used for lookup.
3. **Looks up a static word vector** (300 dimensions) for each lemma in spaCy's `de_core_news_lg` model and drops the words that have no vector.
4. **Computes the cosine similarity** between every pair of words by normalizing the vectors and multiplying the resulting matrix by its transpose.
5. **Returns the nearest neighbours** of any word in the dictionary through the `vecinos()` function.

The approach relies on the distributional hypothesis: words that occur in similar contexts end up with similar vectors.

## Repository contents

| File | Description |
|---|---|
| `mapa_semantico_aleman.py` | Main script |
| `mapa_semantico_aleman.md` | Source code with English comments |
| `diccionario_aleman.txt` | Personal German–Spanish dictionary (input data) |

## Requirements

- Python 3.9+
- spaCy and its German model `de_core_news_lg` (a large download, around 0.5 GB)
- pandas
- NumPy (installed together with spaCy and pandas)

## Installation

```bash
pip install spacy pandas
python -m spacy download de_core_news_lg
```

## Usage

Run the script in interactive mode (`-i`), so that Python stays open after loading the data. It can be run from any folder, since it locates the dictionary file relative to its own path:

```bash
python -i mapa_semantico_aleman.py
```

Then query the neighbours of a word:

```python
vecinos('Hund')       # 10 nearest neighbours
vecinos('Hund', 5)    # 5 nearest neighbours
```

Each result is a table with the neighbour's lemma, its Spanish definition and the cosine similarity.

## Limitations

- **Similarity is not synonymy.** The vectors capture shared contexts, so antonyms (*heiß* / *kalt*) or thematically related words can appear as close neighbours.
- **One word form, one vector.** Static vectors do not distinguish senses: homographs such as *Steuer* (tax / steering wheel) share a single vector.
- **Case-sensitive lookup.** *Hund* and *hund* are different entries for the model.
- **Coverage.** Rare words and long compounds may have no vector in the model and are left out of the analysis.
