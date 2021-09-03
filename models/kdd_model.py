from .model import test_model

"""
Create and evaluate the LG4AV(-2), LG4AV-0 and LG4AV-F
models on the KDD data set.
"""

if __name__ == "__main__":
    print("##################")
    print("##################")
    print("##################") 
    print("Make KDD testing")
    print("Make freezed testing")
    result = test_model(source_train="data/kdd/train_per_author.csv",
                        source_val="data/kdd/val_per_author.csv",
                        source_test="data/kdd/test_per_author.csv",
                        destination="data/kdd/model/tests/freeze/network-results.csv",
                        bert_string="bert-base-uncased",
                        tokenizer_string="bert-base-uncased",
                        dir="data/kdd/model/tests/freeze/",
                        dropout=0.1,
                        k=2,
                        lr=2e-5,
                        weight_decay=0.01,
                        train_bert=False,
                        graph="data/kdd/graph.adjlist",
                        features="data/kdd/base_features/bert-768/cls/embedding.npy",
                        batch_size=4,
                        accumulate_grad_batches=4,
                        epochs=3,
                        gpus=1,
                        num_workers=0)
    print(result)
    for k in [0, 2]:
        print("##################")
        print("##################")
        print("Start for k=", k)
        result = test_model(source_train="data/kdd/train_per_author.csv",
                            source_val="data/kdd/val_per_author.csv",
                            source_test="data/kdd/test_per_author.csv",
                            destination="data/kdd/model/tests/" + str(k) + "/network-results.csv",
                            bert_string="bert-base-uncased",
                            tokenizer_string="bert-base-uncased",
                            dir="data/kdd/model/tests/" + str(k) + "/",
                            dropout=0.1,
                            k=k,
                            lr=2e-5,
                            weight_decay=0.01,
                            graph="data/kdd/graph.adjlist",
                            features="data/kdd/base_features/bert-768/cls/embedding.npy",
                            batch_size=4,
                            accumulate_grad_batches=4,
                            epochs=3,
                            gpus=1,
                            num_workers=0)
        print(result)
