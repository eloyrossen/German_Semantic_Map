# German Semantic Map — Source Code

```python
# German semantic map: the goal of this Python project
# is to build a semantic map from the German words
# I have been learning and saving in my personal dictionary.

# For the terminal:
# !pip install spacy
# !python -m spacy download de_core_news_lg
# !pip install pandas
# !python -m spacy download de_core_news_lg

import re
import pandas
from pathlib import Path

# We look for the .txt file in the same folder as this script,
# regardless of where it is run from.
carpeta = Path(__file__).parent
texto = (carpeta / 'diccionario_aleman.txt').read_text(encoding='utf-8-sig')

# My German dictionary stores words in the following format:
# "(German word) = (Spanish definition)".
# However, there are some things to take into account, such as
# articles, plurals, multiple definitions, etc.

patron = re.compile(r'''
    ^(?:(?P<articulo>der|die|das)\s+)?
    (?:(?P<sich>sich)\s+)?
    (?P<lema>[^\s,=]+)
    (?:,\s*(?:der|die|das)?\s*(?P<plural>[^\s,=]+))?
    \s*=\s*
    (?P<definicion>.+?)\s*$
''', re.VERBOSE)

# The code splits the text into lines at each line break
# and then checks whether the pattern matches the line.
# If there is a match, it is stored in the "filas" list;
# otherwise, it is stored in the "descartadas" list. This makes
# it possible to check later whether the code worked correctly.

descartadas = []
filas = []
for linea in texto.splitlines():
        m = patron.match(linea)
        if m:
            filas.append(m.groupdict())
        else:
            descartadas.append(linea)

df = pandas.DataFrame(filas)

# Next, we load a spaCy model that provides natural language
# processing (NLP) tools for German. It contains a table of
# 500,000 German words with a 300-dimensional vector each, and
# it was trained on the TIGER corpus, made up of German newspaper
# texts annotated by hand, and on WikiNER for named entity
# recognition (NER).

import spacy
nlp = spacy.load('de_core_news_lg')

# For each word, we check whether the model has a vector.
# We use the model's vocabulary (nlp.vocab) instead of nlp(),
# because these are isolated words: there is no context to analyze,
# and running the whole pipeline on each one would be slow and useless.

df['tiene_vector'] = df['lema'].apply(nlp.vocab.has_vector)
df = df[df['tiene_vector']].copy()
df['vector'] = df['lema'].apply(nlp.vocab.get_vector)
df = df.reset_index(drop=True)

# We stack all the vectors into a matrix: each row is a
# word and each column is one of the 300 dimensions.

import numpy as np
matriz = np.vstack(df['vector'].to_numpy())

# We normalize each vector (divide it by its length) so that
# all of them have length 1. This way, multiplying two vectors
# directly gives their cosine similarity: 1 = same direction, 0 = unrelated.

normas = np.linalg.norm(matriz, axis=1, keepdims=True)
matriz_normalizada = matriz / normas

# We compute the similarity of each word with all the others.

similitudes = matriz_normalizada @ matriz_normalizada.T

# We store the lemmas in a list to look up the position of
# each word in the matrix.

lemas = df['lema'].tolist()

# The "vecinos" function takes a word and returns the n words in my
# dictionary most similar to it, with their definition and similarity.

def vecinos(palabra, n=10):
    if palabra not in lemas:
        print(f'"{palabra}" is not in the dictionary or has no vector.')
        return None
    i = lemas.index(palabra)
    orden = similitudes[i].argsort()[::-1]
    orden = [j for j in orden if j != i][:n]
    resultado = df.loc[orden, ['lema', 'definicion']].copy()
    resultado['similitud'] = similitudes[i, orden].round(3)
    return resultado

# The "graficar" function takes a list of words and places them on a plane
# according to their vectors. If n_vecinos is greater than 0, it also adds
# the nearest neighbours of each word, in the same colour.

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def graficar(palabras, n_vecinos=0):
    seleccion = []
    grupos = []
    for palabra in palabras:
        if palabra not in lemas:
            print(f'"{palabra}" is not in the dictionary or has no vector.')
            continue
        if palabra not in seleccion:
            seleccion.append(palabra)
            grupos.append(palabra)
        if n_vecinos > 0:
            for vecino in vecinos(palabra, n_vecinos)['lema']:
                if vecino not in seleccion:
                    seleccion.append(vecino)
                    grupos.append(palabra)

    if len(seleccion) < 3:
        print('You need at least 3 words to make a graph.')
        return None

    # We retrieve the vectors of the selected words and
    # reduce them from 300 dimensions to 2 with PCA.

    posiciones = [lemas.index(p) for p in seleccion]
    vectores = matriz_normalizada[posiciones]
    pca = PCA(n_components=2)
    coordenadas = pca.fit_transform(vectores)

    # We draw one point per word, with one colour per group,
    # and write each word next to its point.

    plt.figure(figsize=(10, 8))
    for grupo in dict.fromkeys(grupos):
        indices = [k for k, g in enumerate(grupos) if g == grupo]
        plt.scatter(coordenadas[indices, 0], coordenadas[indices, 1], label=grupo)
    for k, palabra in enumerate(seleccion):
        plt.annotate(palabra, (coordenadas[k, 0], coordenadas[k, 1]),
                     xytext=(5, 5), textcoords='offset points')

    explicado = pca.explained_variance_ratio_.sum()
    plt.title(f'Semantic map (the 2 axes retain {explicado:.0%} of the variance)')
    if n_vecinos > 0:
        plt.legend()
    plt.show()
    return coordenadas
```
