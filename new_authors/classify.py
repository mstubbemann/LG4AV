import json
import random
import os

from transformers import BertTokenizerFast, BertModel
import numpy as np
import networkx as nx
import torch
import pytorch_lightning as pl
import pandas as pd

from models.model import Net, DataModule


def experiment(modelpath="data/gai/model/tests/2/logs/",
               n_models=10,
               old_features="data/gai/base_features/bert-768/cls/embedding.npy",
               new_features="data/gai/new_authors/base_features/bert-768/cls/embedding.npy",
               old_graph="data/gai/graph.adjlist",
               new_graph="data/gai/new_authors/graph.adjlist",
               old_encoding="data/gai/encoding.json",
               new_encoding="data/gai/new_authors/study_encoding.json",
               new_test_data="data/gai/new_authors/test.csv",
               source_train="data/gai/train_per_author.csv",
               source_val="data/gai/val_per_author.csv",
               source_test="data/gai/test_per_author.csv",
               bert_string="bert-base-uncased",
               destination="data/gai/new_authors/",
               k=2):
    """
    Use LG4AV to classify for unseen authors.
    """

    # First load some stuff and preparation
    with open(old_encoding) as file:
        old_encoding = json.load(file)
    with open(new_encoding) as file:
        new_encoding = json.load(file)
    gpus = 1 if torch.cuda.is_available() else None

    # Prepare tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    old_authors = sorted(list(old_encoding.values()))
    old_authors_set = set(old_authors)
    n_old_authors = len(old_authors)
    new_authors = sorted(list(new_encoding.values()))
    n_new_authors = len(new_authors)
    authors = old_authors + new_authors
    special_tokens = ["[CLS-" + str(author) + "]"
                      for author in authors]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
    # Compute features and graphs
    old_features = np.load(old_features)
    old_G = nx.read_adjlist(old_graph, nodetype=int)
    G = nx.read_adjlist(new_graph, nodetype=int)
    new_features = np.load(new_features)
    features = np.concatenate((old_features, new_features))

    # Old and new results DataFrame
    old_results = pd.DataFrame()
    results = pd.DataFrame()

    # Go through all models
    for i in range(n_models):
        print("Start with model", i)

        # Reduce random sources
        seed = i+1
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        # Load data
        authors = sorted(list(G.nodes()))
        data = DataModule(authors=authors,
                          source_train=source_train,
                          source_val=source_val,
                          source_test=new_test_data,
                          features=features,
                          G=G,
                          k=k,
                          num_workers=0,
                          seed=seed)
        old_data = DataModule(source_train=source_train,
                              source_val=source_val,
                              source_test=source_test,
                              features=old_features,
                              G=old_G,
                              k=k,
                              num_workers=0,
                              seed=seed)
    
        # Load Bert model
        # Needs reshape for proper loading
        bert = BertModel.from_pretrained(bert_string)
        bert.resize_token_embeddings(len(old_data.train_data.tokenizer))
        print("Bert after resize:")
        cls_id = data.tokenizer.cls_token_id
        cls_embedding = bert.embeddings.word_embeddings.weight[cls_id].detach()
        with torch.no_grad():
            bert.embeddings.word_embeddings.weight[-n_old_authors:] = cls_embedding.repeat(
                n_old_authors, 1)

        # Start with old embeddings and make convience check
        print("Old BERT token ebmeddings convience check")
        print(bert.embeddings.word_embeddings.weight[-5:, :3])
        net = Net.load_from_checkpoint(modelpath + "version_" + str(i)
                                       + "/checkpoints/last.ckpt",
                                       bert=bert)
        old_embeddings = bert.embeddings.word_embeddings.weight[-n_old_authors:].detach()
        # Average embedding if no connections into the past
        average_embedding = torch.mean(old_embeddings, dim=0)

        print("Now compute new embeddings")
        # Now make embeddings for new tokens
        new_embeddings = []
        for author in new_authors:
            neighbors = list(set(G.neighbors(author)) & old_authors_set)
            if not neighbors:
                new_embeddings.append(average_embedding)
            else:
                embedding = torch.mean(old_embeddings[neighbors], dim=0)
                new_embeddings.append(embedding)
        new_embeddings = torch.stack(new_embeddings)
        net.bert.resize_token_embeddings(len(data.train_data.tokenizer))
        with torch.no_grad():
            net.bert.embeddings.word_embeddings.weight[-n_new_authors:] = new_embeddings

        # For convience print old test results
        print("For convenience: First test on old data")
        trainer = pl.Trainer(logger=False,
                             deterministic=True,
                             accumulate_grad_batches=4,
                             gpus=gpus)
        trainer.test(model=net, datamodule=old_data)
        old_results = old_results.append(net.test_result, ignore_index=True)

        # Now make new tests
        trainer = pl.Trainer(logger=False,
                             deterministic=True,
                             accumulate_grad_batches=4,
                             gpus=gpus)
        print("Lets do testing")
        trainer.test(model=net, datamodule=data)
        results = results.append(net.test_result, ignore_index=True)
        print("Finished run:", i)
        print("################")
        print("################")

    # Save finished results
    if destination:
        results.to_csv(destination + "lg4av.csv")
        old_results.to_csv(destination + "old_lg4av.csv")
