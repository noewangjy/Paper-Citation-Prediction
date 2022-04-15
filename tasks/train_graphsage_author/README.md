# Brainstorm

## Introduction:

clf:
- GraphSAGE on author network, feature = nn.Embedding(author) -> Nx32
- GraphSAGE on essay network, feature = WordFrequencyVector -> Nx32

## Features

`len(authors[u]) + len(authors[v])`❌

`distance(u, v)`❌

`max(2, distance(u,v)`✅

`len((authors[u] & authors[v]))`

`cos_sim(abstract[u], abstract[v])`

@yutong `cos_sim(deep_walk_embedding[u], deep_walk_embedding[v])`

@yutong `cos_sim(graphsage[u], graphsage[v])`

`mean(cos_sim([word in abstact[u]], [word in abstract[v]]))`

@yutong `knn(deep_walk(authors))` 在authors_network跑deepwalk, 对作者聚类

`hierarchical_neighbor(u) & hierarchical_neighbor(v)`🆗

`abs(sum(neighbor(u).degree) - sum(neighbor(v).degree))`

