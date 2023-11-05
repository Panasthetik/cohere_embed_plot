import cohere
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import umap
#import umap-learn - this is required in pip install but not here for umap.UMAP.
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)

dataset = load_dataset("trec", split="train")

df = pd.DataFrame(dataset)[:1000]
df.head(10)

co = cohere.Client('<YOUR COHERE API KEY>')

embeds = co.embed(texts=list(df['text']),
                  model='embed-english-v2.0').embeddings

#annoy
search_index=AnnoyIndex(np.array(embeds).shape[1], 'angular')

#vectors add
for i in range(len(embeds)):
    search_index.add_item(i, embeds[i])
search_index.build(10)
search_index.save('testembed.ann')

# EXAMPLE 1
example_id = 92
similar_item_ids = search_index.get_nns_by_item(example_id, 10, include_distances=True)
#format and print sample
results = pd.DataFrame(data={'texts': df.iloc[similar_item_ids[0]]['text'],
                             'distance': similar_item_ids[1]}).drop(example_id)
print(f"Question: '{df.iloc[example_id]['text']}'\nNearest neighbors:")
print(results)

#EXAMPLE2
query = "What is the tallest mountain in the world?"
#query embed
query_embed = co.embed(texts=[query],
                  model='embed-english-v2.0').embeddings

similar_item_ids = search_index.get_nns_by_vector(query_embed[0], 10,
                                                  include_distances=True)

results = pd.DataFrame(data={'texts': df.iloc[similar_item_ids[0]]['text'],
                             'distance': similar_item_ids[1]})

print(f"Query:'{query}'\nNearest neighbors:")
print(results)

# Plot results ::
reducer = umap.UMAP(n_neighbors=20)
umap_embeds = reducer.fit_transform(embeds)
df_explore = pd.DataFrame(data={'text': df['text']})
df_explore['x'] = umap_embeds[:, 0]
df_explore['y'] = umap_embeds[:, 1]

chart = alt.Chart(df_explore).mark_circle(size=60).encode(
    x=#'x',
    alt.X('x',
          scale=alt.Scale(zero=False)
    ),
    y=
    alt.Y('y',
          scale=alt.Scale(zero=False)
    ),
    tooltip=['text']
).properties(
    width=700,
    height=400
)
chart.interactive()
chart.save('chart.html')