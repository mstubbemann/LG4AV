import json
import random
from collections import defaultdict
import re

import langdetect
import networkx as nx
import pandas as pd
import os


def filter_english(source="data/german_ai_community_dataset.json",
                   destination="data/gai/en_data.json"):
    """
    Throw away all publications that are not in english.
    """
    # Make langauge detection reproducible
    langdetect.DetectorFactory.seed = 0
    with open(source) as file:
        with open(destination, "w") as out:
            for line in file:
                line = json.loads(line)
                if "abstract" in line and line["abstract"]:
                    document = line["abstract"]
                    try:
                        language = langdetect.detect(document)
                        if language != "en":
                            continue
                    except:
                        continue
                    json.dump(line, out)
                    out.write("\n")


def catch_authors_and_graph(source="data/gai/en_data.json",
                            author_keys="german_ai_authors_keys",
                            start_year=0,
                            year=2015,
                            biggest_component=True,
                            graph_destination="data/gai/graph.adjlist",
                            encoding_destination="data/gai/encoding.json",
                            venues=None,
                            seed=13):
    """
    Store all relevant authors and the corresponding co-authorgraph.
    """
    if seed:
        random.seed(seed)
    G = nx.Graph()
    with open(source) as file:
        for line in file:
            line = json.loads(line)
            # Check for venue if wanted
            if venues:
                if line["journal"]:
                    curr_venue = line["journal"][0]
                elif line["booktitle"]:
                    curr_venue = line["booktitle"][0]
                curr_venue = re.sub(" \\([0-9]+\\)$", "", curr_venue)
                if curr_venue not in venues:
                    continue
            if "year" in line and start_year <= int(line["year"]) <= year:
                keys = line[author_keys]
                G.add_nodes_from(keys)
                G.add_edges_from([(a, b) for a in keys for b in keys
                                  if a != b])
    # Check if just biggest_component wanted
    if biggest_component:
        gcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(gcc).copy()
    encoding = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, encoding)
    with open(encoding_destination, "w") as file:
        json.dump(encoding, file)
    nx.readwrite.adjlist.write_adjlist(G, graph_destination)


def author_vs_paper(source="data/gai/en_data.json",
                    source_author="data/gai/encoding.json",
                    destination="data/gai/",
                    author_keys="german_ai_authors_keys",
                    title=True,
                    start_year=0,
                    year=2015,
                    val_year=2016,
                    seed=42):
    """
    Generate the training, validation and testing data
    """
    # Some preparation
    if seed:
        random.seed(seed)
    with open(source_author) as file:
        encoding = json.load(file)
    if not os.path.isdir(destination):
        os.makedirs(destination)

    # Positive training examples
    true_data = pd.DataFrame(columns=["text",
                                      "author",
                                      "label"])

    # Catch all relevant papers and store abstracts
    paper_encoding = {}

    # Connections from authors to paper
    node_connections = defaultdict(list)

    # Connections from papers to authors
    paper_connections = defaultdict(list)

    # Iterate through publications and find relevant ones
    with open(source) as file:
        for line in file:
            line = json.loads(line)
            if "year" in line and start_year <= int(line["year"]) <= year:
                pre_keys = [encoding[x] for x in line[author_keys]
                            if x in encoding]
                # Fixup: Some authors doubled in Semantic SCholar
                key_set = set()
                keys = []
                for key in pre_keys:
                    if key in key_set:
                        continue
                    else:
                        key_set.add(key)
                        keys.append(key)

                if keys:
                    abstract = line["abstract"]
                    if title and "title" in line:
                        abstract = line["title"] + "\n" + abstract
                    # Replace quotechar
                    abstract = abstract.replace("'", "")
                    for key in keys:
                        true_data = true_data.append({"text": abstract,
                                                      "author": key,
                                                      "label": 1},
                                                     ignore_index=True)
                        node_connections[key].append(line["key"])
                        paper_connections[line["key"]].append(key)
                    paper_encoding[line["key"]] = abstract

    # Negative sampling
    false_author = pd.DataFrame(columns=["text",
                                         "author",
                                         "label"])
    papers = set(paper_connections.keys())

    # For each positive examples, find a negative one!
    for key in node_connections.keys():

        # Find equal amount of papers that are not from this author
        author_papers = set(node_connections[key])
        non_papers = papers - author_papers
        # For deterministic negative examples
        non_papers = sorted(list(non_papers))
        non_papers = random.sample(non_papers,
                                   len(author_papers))

        for paper in non_papers:
            false_author = false_author.append({"text": paper_encoding[paper],
                                                "author": key,
                                                "label": 0},
                                               ignore_index=True)

    # Concat positive and negative training examples
    # and save them.
    result = pd.concat([true_data,
                        false_author],
                       ignore_index=True)
    result.to_csv(destination + "train_per_author.csv",
                  quotechar="'")
    print("Finished training data")

    # Relevant authors to search for for
    # validation and testing
    relevant_nodes = set(result["author"])

    # Preparation to identify validation and test examples
    paper_encoding = {}
    val = pd.DataFrame(columns=["text",
                                "author",
                                "label"])
    author_val = defaultdict(list)
    paper_val = defaultdict(list)
    test = pd.DataFrame(columns=["text",
                                 "author",
                                 "label"])
    author_test = defaultdict(list)
    paper_test = defaultdict(list)

    # Iterate through data for Val and Test examples
    with open(source) as file:
        for line in file:
            line = json.loads(line)
            if "year" in line and year < int(line["year"]) <= val_year:

                # Identify relevant nodes of the current paper
                keys = [encoding[x] for x in line[author_keys]
                        if x in encoding]
                pre_keys = [x for x in keys if x in relevant_nodes]

                # Fixup: Some authors doubled in Semantic SCholar
                key_set = set()
                keys = []
                for key in pre_keys:
                    if key in key_set:
                        continue
                    else:
                        key_set.add(key)
                        keys.append(key)

                if keys:
                    abstract = line["abstract"]
                    if title and "title" in line:
                        abstract = line["title"] + "\n" + abstract
                    abstract = abstract.replace("'", "")
                    for key in keys:
                        val = val.append({"text": abstract,
                                          "author": key,
                                          "label": 1},
                                         ignore_index=True)
                        author_val[key].append(line["key"])
                        paper_val[line["key"]].append(key)
                    paper_encoding[line["key"]] = abstract
            elif "year" and int(line["year"]) > val_year:
                keys = [encoding[x] for x in line[author_keys]
                        if x in encoding]
                pre_keys = [x for x in keys if x in relevant_nodes]

                # Fixup: Some authors doubled in Semantic SCholar
                key_set = set()
                keys = []
                for key in pre_keys:
                    if key in key_set:
                        continue
                    else:
                        key_set.add(key)
                        keys.append(key)

                if keys:
                    abstract = line["abstract"]
                    if title and "title" in line:
                        abstract = line["title"] + "\n" + abstract
                    abstract = abstract.replace("'", "")
                    for key in keys:
                        test = test.append({"text": abstract,
                                            "author": key,
                                            "label": 1},
                                           ignore_index=True)
                        author_test[key].append(line["key"])
                        paper_test[line["key"]].append(key)
                    paper_encoding[line["key"]] = abstract
    print("Finished positive validation and test examples")

    # Negative sampling for validation
    false_author = pd.DataFrame(columns=["text",
                                         "author",
                                         "label"])
    papers = set(paper_val.keys())
    for key in author_val.keys():
        author_papers = set(author_val[key])
        non_papers = papers - author_papers
        # For deterministic negative examples
        non_papers = sorted(list(non_papers))
        non_papers = random.sample(non_papers,
                                   len(author_papers))
        for paper in non_papers:
            false_author = false_author.append({"text": paper_encoding[paper],
                                                "author": key,
                                                "label": 0},
                                               ignore_index=True)
    result = pd.concat([val,
                        false_author],
                       ignore_index=True)
    result.to_csv(destination + "val_per_author.csv",
                  quotechar="'")
    print("Finished validation data")

    # Negative sampling for testing
    false_author = pd.DataFrame(columns=["text",
                                         "author",
                                         "label"])
    papers = set(paper_test.keys())
    for key in author_test.keys():
        author_papers = set(author_test[key])
        non_papers = papers - author_papers
        # For deterministic negative examples
        non_papers = sorted(list(non_papers))
        non_papers = random.sample(non_papers,
                                   len(author_papers))
        for paper in non_papers:
            false_author = false_author.append({"text": paper_encoding[paper],
                                                "author": key,
                                                "label": 0},
                                               ignore_index=True)
    result = pd.concat([test,
                        false_author],
                       ignore_index=True)
    result.to_csv(destination + "test_per_author.csv",
                  quotechar="'")
    print("Finished test data")
    print("Finished data preprocessing")


def author_documents_list(source="data/gai/en_data.json",
                          source_author="data/gai/encoding.json",
                          destination="data/gai/author_documents_list.json",
                          author_keys="german_ai_authors_keys",
                          start_year=0,
                          year=2015,
                          title=True):
    """ 
    Identify for all authors their papers in training time.
    Used for feature computation.
    """
    with open(source_author) as file:
        encoding = json.load(file)
    result = defaultdict(list)
    with open(source) as file:
        for line in file:
            line = json.loads(line)
            if "year" in line and start_year <= int(line["year"]) <= year:
                abstract = line["abstract"]
                if title and "title" in line:
                    abstract = line["title"] + "\n" + abstract

                pre_keys = line[author_keys]
                # Fixup: Some authors doubled in Semantic SCholar
                key_set = set()
                keys = []
                for key in pre_keys:
                    if key in key_set:
                        continue
                    else:
                        key_set.add(key)
                        keys.append(key)

                for author in keys:
                    if author in encoding:
                        result[encoding[author]].append(abstract)
    with open(destination, "w") as file:
        json.dump(result, file)


# Run all preprocessing for experiments
if __name__ == "__main__":
    print("Preprocess data for GAI")
    if not os.path.isdir("data/gai/"):
        os.makedirs("data/gai/")
    print("Filter for english docs")
    filter_english()
    print("Catch relevant authors and co-author graph")
    catch_authors_and_graph()
    print("Catch av examples")
    author_vs_paper()
    print("Prepare data for author features")
    author_documents_list()
    print("Preprocess data for KDD")
    if not os.path.isdir("data/kdd/"):
        os.makedirs("data/kdd/")
    print("Filter for english docs")
    filter_english(source="data/ai_community_dataset.json",
                   destination="data/kdd/en_data.json")
    print("Catch relevant authors and co-author graph")
    catch_authors_and_graph(source="data/kdd/en_data.json",
                            author_keys="ai_authors_keys",
                            graph_destination="data/kdd/graph.adjlist",
                            encoding_destination="data/kdd/encoding.json",
                            venues={"KDD"})
    print("Catch av examples")
    author_vs_paper(source="data/kdd/en_data.json",
                    source_author="data/kdd/encoding.json",
                    destination="data/kdd/",
                    author_keys="ai_authors_keys")
    print("Prepare data for author features")
    author_documents_list(source="data/kdd/en_data.json",
                          source_author="data/kdd/encoding.json",
                          destination="data/kdd/author_documents_list.json",
                          author_keys="ai_authors_keys")
