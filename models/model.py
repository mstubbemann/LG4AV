import os
import random
from math import ceil
import gc

import numpy as np
import pandas as pd
import networkx as nx
from transformers import BertModel, BertTokenizerFast
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


# ############################The network####################

class Net(pl.LightningModule):
    """
    The LG4AV architecture
    """

    def __init__(self,
                 feature_dim=768,
                 dir="data/model/",
                 bert=None,
                 bert_dim=768,
                 dropout=0.1,
                 k=2,
                 lr=2e-5,
                 train_steps=100,
                 warmup_steps=0,
                 weight_decay=0.01):
        # Save parameters
        super(Net, self).__init__()
        self.save_hyperparameters("feature_dim",
                                  "bert_dim",
                                  "dropout",
                                  "k",
                                  "lr",
                                  "train_steps",
                                  "warmup_steps",
                                  "weight_decay")

        # Some preperation
        self.feature_dim = feature_dim
        self.lr = lr
        self.train_steps = train_steps
        self.bert = bert
        self.bert_dim = bert_dim
        self.k = k

        # Check if warmup steps are fracs
        # or absolute numbers
        if type(warmup_steps) == int:
            self.warmup_steps = warmup_steps
        else:
            self.warmup_steps = int(warmup_steps * train_steps)

        self.weight_decay = weight_decay
        self.dir = dir
        if not os.path.isdir(dir):
            os.makedirs(dir)

        # Layers of the network
        if self.bert_dim != self.feature_dim:
            self.bert_reduction = nn.Linear(self.bert_dim,
                                            self.feature_dim)
        else:
            self.bert_reduction = nn.Identity()
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_layer = nn.Linear((k+1) * self.feature_dim, 1)
        print("Some Token embeddings:")
        print(self.bert.embeddings.word_embeddings.weight[-5:, :3])

    def forward(self,
                batch):
        ids, input_masks, segment_ids, Xs = batch

        # Feed doc through Bert
        B = self.bert(input_ids=ids,
                      attention_mask=input_masks,
                      token_type_ids=segment_ids)[0]

        # Only use cls tokens
        B = B[:, 0]

        # Multiply Bert output with features
        Xs = [X * B for X in Xs]

        # Feed through MLP
        X = torch.cat(Xs, axis=1)
        X = self.dropout(X)
        X = self.hidden_layer(X)

        # Return with dimension reduced:
        # (n X 1) -> (n)
        return X[:, 0]

    def training_step(self,
                      batch,
                      batch_idx):
        target = batch[-1]
        output = self(batch[:-1])
        loss = F.binary_cross_entropy_with_logits(output,
                                                  target)
        return loss

    def validation_step(self,
                        batch,
                        batch_idx):
        labels = batch[-1]
        output = self(batch[:-1])
        output = torch.sigmoid(output)
        return {"preds": output,
                "labels": labels}

    def test_step(self,
                  batch,
                  batch_idx):
        labels = batch[-1]
        output = self(batch[:-1])
        output = torch.sigmoid(output)
        return {"preds": output,
                "labels": labels}

    def validation_epoch_end(self, outs):
        scores = [out["preds"] for out in outs]
        scores = torch.cat(scores).detach().cpu().numpy()
        preds = np.array([1 if x >= 0.5 else 0 for x in scores])
        labels = torch.cat([out["labels"] for out in outs]).detach().cpu().numpy()
        f1 = f1_score(labels, preds)
        accuracy = accuracy_score(labels, preds)
        # Needed for validation sanity check
        n = len({x for x in labels})
        if n == 1:
            auc = 0
        else:
            auc = roc_auc_score(labels, scores)
        dic = {"val_f1": f1,
               "val_accuracy": accuracy,
               "val_auc": auc}

        # Print current validation results and store them
        print("\n")
        print(dic)
        self.log_dict(dic,
                      on_step=False,
                      on_epoch=True,
                      logger=True)
        self.val_result = dic

    def test_epoch_end(self, outs):
        print("\n")
        scores = [out["preds"] for out in outs]
        scores = torch.cat(scores).detach().cpu().numpy()
        preds = np.array([1 if x >= 0.5 else 0 for x in scores])
        labels = torch.cat([out["labels"] for out in outs]).detach().cpu().numpy()
        f1 = f1_score(labels, preds)
        accuracy = accuracy_score(labels, preds)
        n = len({x for x in labels})
        if n == 1:
            auc = 0
        else:
            auc = roc_auc_score(labels, scores)

        # Print test results and store them
        dic = {"test_f1": f1,
               "test_accuracy": accuracy,
               "test_auc": auc}
        print("\n")
        print(dic)
        self.log_dict(dic,
                      on_step=False,
                      on_epoch=True,
                      logger=True)
        self.test_result = dic

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        self.opt_params = [
            {'params': [p for n, p in self.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.named_parameters()
                        if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}]
        optimizer = AdamW(self.opt_params,
                          lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.train_steps)
        return {"optimizer": optimizer,
                "lr_scheduler": scheduler}

# ##############Data Preparation#############################


class Data(Dataset):

    def __init__(self,
                 source="data/gai/train_per_author.csv",
                 batch_size=4,
                 tokenizer=None,
                 features=None,
                 G=None,
                 k=2,
                 need_merge=True,
                 train_bert=True,
                 seed=42):
        """
        A dataset class needed for LG4AV.
        """
        # Load data and shuffle if wanted
        data = pd.read_csv(source,
                           quotechar="'",
                           index_col=0)

        if seed:
            data = data.sample(frac=1,
                               random_state=seed).reset_index(drop=True)
        # Preparte text inputs
        self.texts = list(data["text"])
        self.authors = torch.tensor(list(data["author"]), dtype=torch.long)

        # Add individual_cls tokens if wanted
        self.train_bert = train_bert
        if self.train_bert:
            self.texts = ["[CLS-" + str(author.item()) + "] " + text
                          for author, text in zip(self.authors,
                                                  self.texts)]
        self.tokenizer = tokenizer

        # Make neighborhood aggregation if needed
        if need_merge:
            G.add_edges_from([(i, i) for i in G.nodes])
            self.features = [torch.tensor(features, dtype=torch.float)]
            nodes = list(G.nodes)
            nodes.sort()
            A = nx.to_numpy_array(G, nodelist=nodes)
            inv_deg = np.sum(A, axis=1)**(-0.5)
            diag = np.diag(inv_deg)
            A = np.matmul(np.matmul(diag, A), diag)
            for _ in range(1, k+1):
                features = np.matmul(A, features)
                self.features.append(torch.tensor(features,
                                                  dtype=torch.float))
        else:
            self.features = features
        # Make final batches
        self.authors = torch.split(self.authors, batch_size)
        self.text_labels = torch.tensor(list(data["label"]),
                                        dtype=torch.float)
        self.text_labels = torch.split(self.text_labels,
                                       batch_size)
        self.inputs = [self.texts[i:i+batch_size]
                       for i in range(0, len(self.texts),
                                      batch_size)]
        self.batched_features = []
        for batch in self.authors:
            features = [X[batch] for X in self.features]
            self.batched_features.append(features)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        bert_input = self.tokenizer(input,
                                    truncation=True,
                                    padding=True,
                                    max_length=512)
        ids = torch.tensor(bert_input["input_ids"])
        input_masks = torch.tensor(bert_input["attention_mask"])
        segment_ids = torch.tensor(bert_input["token_type_ids"])
        # If we train Bert, we use  author specific cls token -> remove the classical
        # cls token!
        if self.train_bert:
            ids = ids[:, 1:]
            input_masks = input_masks[:, 1:]
            segment_ids = segment_ids[:, 1:]

        return (ids,
                input_masks,
                segment_ids,
                self.batched_features[idx],
                self.text_labels[idx])


class DataModule(pl.LightningDataModule):

    def __init__(self,
                 authors=None,
                 source_train="data/gai/train_per_author.csv",
                 source_val="data/gai/val_per_author.csv",
                 source_test="data/gai/test_per_author.csv",
                 batch_size=4,
                 features=None,
                 G=None,
                 tokenizer_string="bert-base-uncased",
                 k=2,
                 num_workers=0,
                 train_bert=True,
                 seed=42):
        """
        Pytorch Lighning Datamodule for LG4AV.
        """
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_string)
        if not authors:
            d_train = pd.read_csv(source_train,
                                  quotechar="'")
            self.authors = list(set(d_train["author"]))
            self.authors.sort()
        else:
            self.authors = authors

        # Add individual cls tokens to tokenizer
        if train_bert:
            special_tokens = ["[CLS-" + str(author) + "]"
                              for author in self.authors]
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": special_tokens})

        self.train_data = Data(source=source_train,
                               batch_size=batch_size,
                               tokenizer=self.tokenizer,
                               features=features,
                               G=G,
                               k=k,
                               need_merge=True,
                               seed=seed,
                               train_bert=train_bert)
        self.val_data = Data(source=source_val,
                             batch_size=batch_size,
                             tokenizer=self.tokenizer,
                             features=self.train_data.features,
                             G=G,
                             k=k,
                             need_merge=False,
                             seed=None,
                             train_bert=train_bert)
        self.test_data = Data(source=source_test,
                              batch_size=batch_size,
                              tokenizer=self.tokenizer,
                              features=self.train_data.features,
                              G=G,
                              k=k,
                              need_merge=False,
                              seed=None,
                              train_bert=train_bert)
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          num_workers=self.num_workers,
                          batch_size=None)

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          num_workers=self.num_workers,
                          batch_size=None)

    def test_dataloader(self):
        return DataLoader(self.test_data,
                          num_workers=self.num_workers,
                          batch_size=None)


def test_model(source_train="data/gai/train_per_author.csv",
               source_val="data/gai/val_per_author.csv",
               source_test="data/gai/test_per_author.csv",
               destination="data/gai/network-results.csv",
               bert_string="bert-base-uncased",
               tokenizer_string="bert-base-uncased",
               dir="data/gai/model/tests/",
               dropout=0.1,
               k=2,
               lr=2e-5,
               warmup_steps=0,
               weight_decay=0.01,
               train_bert=True,
               graph="data/gai/graph.adjlist",
               features=None,
               batch_size=4,
               accumulate_grad_batches=4,
               epochs=3,
               gpus=1,
               num_workers=0,
               iterations=10):
    """
    Run LG4AV on a specific dataset
    """
    if not os.path.isdir(dir):
        os.makedirs(dir)

    # Prepare training
    gpus = gpus if torch.cuda.is_available() else None
    # Read features and graph
    graph = nx.read_adjlist(graph, nodetype=int)
    features = np.load(features)

    results = pd.DataFrame()

    for i in range(1, iterations + 1):
        # Reduce random sources
        seed = i
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        # Some preparation
        data = DataModule(source_train=source_train,
                          source_val=source_val,
                          source_test=source_test,
                          batch_size=batch_size,
                          features=features,
                          G=graph,
                          tokenizer_string=tokenizer_string,
                          k=k,
                          num_workers=num_workers,
                          train_bert=train_bert,
                          seed=seed)
        # Compute train steps for optimizer with warump and cooldown
        train_instances = len(data.train_data)
        batches_per_epoch = ceil(train_instances/accumulate_grad_batches)
        train_steps = epochs * batches_per_epoch

        bert = BertModel.from_pretrained(bert_string)
        tokenizer = data.tokenizer
        n_authors = len(data.authors)

        # For individual cls token if Bert is trained:
        # Initalize embeddings for them at end
        # with current value of standard cls token
        if train_bert:
            bert.resize_token_embeddings(len(data.train_data.tokenizer))
            cls_id = tokenizer.cls_token_id
            cls_embedding = bert.embeddings.word_embeddings.weight[cls_id].detach()
            with torch.no_grad():
                bert.embeddings.word_embeddings.weight[-n_authors:] = cls_embedding.repeat(
                    n_authors, 1)
        else:
            for param in bert.parameters():
                param.requires_grad = False
        # Make the actual net
        net = Net(feature_dim=features.shape[1],
                  dir=dir,
                  bert=bert,
                  dropout=dropout,
                  k=k,
                  lr=lr,
                  train_steps=train_steps,
                  warmup_steps=warmup_steps,
                  weight_decay=weight_decay)

        # Some preparation
        logger = TensorBoardLogger(save_dir=dir,
                                   name="logs/")
        ckpt = ModelCheckpoint(save_last=True,
                               save_top_k=1,
                               monitor="val_auc",
                               mode="max")
        trainer = pl.Trainer(logger=logger,
                             deterministic=True,
                             min_epochs=1,
                             max_epochs=epochs,
                             accumulate_grad_batches=accumulate_grad_batches,
                             gpus=gpus,
                             callbacks=[ckpt])
        # Do the actual training
        trainer.fit(model=net, datamodule=data)
        result = {}
        result["val_f1"] = net.val_result["val_f1"]
        result["val_accuracy"] = net.val_result["val_accuracy"]
        result["val_auc"] = net.val_result["val_auc"]

        # Do testing
        trainer.test(model=None, datamodule=data,
                     ckpt_path=None)
        result["test_f1"] = net.test_result["test_f1"]
        result["test_accuracy"] = net.test_result["test_accuracy"]
        result["test_auc"] = net.test_result["test_auc"]
        result["i"] = i
        results = results.append(result,
                                 ignore_index=True)

        # Prevent GPU memory overflow
        del net, trainer, logger, ckpt, data, bert, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        if destination:
            results.to_csv(destination)
    return results
