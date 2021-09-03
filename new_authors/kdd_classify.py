from .classify import experiment

"""
Run LG4AV on new KDD authors not seen at training time.
"""

if __name__ == "__main__":
    experiment(modelpath="data/kdd/model/tests/2/logs/",
               old_features="data/kdd/base_features/bert-768/cls/embedding.npy",
               new_features="data/kdd/new_authors/base_features/bert-768/cls/embedding.npy",
               old_graph="data/kdd/graph.adjlist",
               new_graph="data/kdd/new_authors/graph.adjlist",
               old_encoding="data/kdd/encoding.json",
               new_encoding="data/kdd/new_authors/study_encoding.json",
               new_test_data="data/kdd/new_authors/test.csv",
               source_train="data/kdd/train_per_author.csv",
               source_val="data/kdd/val_per_author.csv",
               source_test="data/kdd/test_per_author.csv",
               bert_string="bert-base-uncased",
               destination="data/kdd/new_authors/")
