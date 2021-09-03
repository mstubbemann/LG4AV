from preprocessing.features import bert_embedding

if __name__ == "__main__":
    print("Start with GAI features")
    bert_embedding(source="data/gai/new_authors/author_documents_list.json",
                   destination="data/gai/new_authors/base_features/bert-768/",
                   bert="bert-base-uncased")
    print("Start with KDD features")
    bert_embedding(source="data/kdd/new_authors/author_documents_list.json",
                   destination="data/kdd/new_authors/base_features/bert-768/",
                   bert="bert-base-uncased")
