import json
import os
import random

import numpy as np
import torch
from transformers import BertModel, BertTokenizerFast


def bert_embedding(source="data/gai/author_documents_list.json",
                   destination="data/gai/features/bert/",
                   bert="bert-base-uncased",
                   seed=42):
    """
    Use pre-trained Bert to make feature vectors for all authors.
    """

    # Seed
    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    # Use bert to compute feature vectors

    # Some preparation
    if not os.path.isdir(destination):
        os.makedirs(destination)
        os.makedirs(destination + "cls/")
    with open(source) as file:
        data = json.load(file)
    order = []
    result_cls = {}
    n_authors = len(data)

    # Load model and go into evaluation mode
    model = BertModel.from_pretrained(bert)
    model.eval()
    tokenizer = BertTokenizerFast.from_pretrained(bert)

    # Dont store gradients to fasten things up
    with torch.no_grad():
        for i, (key, value) in enumerate(data.items()):
            order.append(int(key))
            author_result_cls = []
            for document in value:
                tokens = tokenizer([document],
                                   padding=True,
                                   truncation=True,
                                   return_tensors="pt",
                                   max_length=512)
                embs = model(**tokens)[0][0].detach()
                # CLS pooling
                X_cls = embs[0]
                author_result_cls.append(X_cls)
            author_result_cls = torch.mean(torch.stack(author_result_cls),
                                           0).numpy()
            result_cls[int(key)] = author_result_cls
            print(i+1, "/", n_authors)
    order.sort()
    print(order)
    result_cls = np.array([result_cls[i] for i in order])
    np.save(destination + "cls/embedding.npy", result_cls)


# Make text embeddings

if __name__ == "__main__":
    bert_embedding(source="data/gai/author_documents_list.json",
                   destination="data/gai/base_features/bert-768/",
                   bert="bert-base-uncased")
    bert_embedding(source="data/kdd/author_documents_list.json",
                   destination="data/kdd/base_features/bert-768/",
                   bert="bert-base-uncased")
