# # make sure the batch is on device cpu
# def convert_data_to_sample(batch: Dict) -> List[BiEncoderSample]:
#     query = batch['query']
#     context = batch['context']
#     label = batch['label']
#
#
#
#
#
#     query_indices = batch["item"]
#     query_authors = batch["authors"]
#     query_abstracts = batch["abstracts"]
#     pos_indices = batch["pos_passages"].detach().cpu().numpy()
#     neg_indices = batch["neg_passages"].detach().cpu().numpy()
#     pos_authors = batch["pos_authors"]
#     pos_abstracts = batch["pos_abstracts"]
#     neg_authors = batch["neg_authors"]
#     neg_abstracts = batch["neg_abstracts"]
#     samples = []
#
#     for idx in range(len(query_indices)):
#         query_passage = BiEncoderPassage(query_authors[idx], query_abstracts[idx])
#         positive_passages = [BiEncoderPassage(pos_authors[i][idx], pos_abstracts[i][idx]) for i in
#                              range(pos_indices.shape[1]) if pos_indices[idx, i] != -1]
#         negative_passages = [BiEncoderPassage(neg_authors[i][idx], neg_abstracts[i][idx]) for i in
#                              range(neg_indices.shape[1])]
#         samples.append(BiEncoderSample(
#             query_passage,
#             positive_passages,
#             negative_passages,
#             [],
#         ))
#     return samples
