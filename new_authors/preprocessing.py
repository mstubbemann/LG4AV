import random
import os
import json
from collections import defaultdict


import networkx as nx
import pandas as pd
import numpy as np


def new_authors(source="data/gai/en_data.json",
                author_source="data/gai/encoding.json",
                graph_source="data/gai/graph.adjlist",
                author_keys="german_ai_authors_keys",
                dest="data/gai/new_authors/",
                title=True,
                start_year=0,
                year=2015,
                val_year=2016,
                venues=None,
                seed=13):
    """
    Prepare data for experiment on authors not seen at training time.
    """
    # Set random seed
    if seed:
        random.seed(seed)
    # Setup stuff
    if not os.path.isdir(dest):
        os.makedirs(dest)
    val_dest = dest + "val.csv"
    test_dest = dest + "test.csv"
    encoding_study = dest + "study_encoding.json"
    graph_dest = dest + "graph.adjlist"
    adl_dest = dest + "author_documents_list.json"

    # Load data of graph
    G = nx.read_adjlist(graph_source, nodetype=int)
    with open(author_source) as file:
        encoding = json.load(file)
    connections_back = defaultdict(set)
    G_val = nx.Graph()

    # Catch all older authors
    authors = set()
    authors_in_gcc = set(encoding.keys())
    with open(source) as file:
        for line in file:
            line = json.loads(line)
            # Check for venue if wanted
            # If we want venues,
            # first catch all authors of this venue,
            # independent of time!
            if venues:
                if line["journal"]:
                    if line["journal"][0] not in venues:
                        continue
                    else:
                        keys = line[author_keys]
                        authors.update(keys)
                elif line["booktitle"]:
                    if line["booktitle"][0] not in venues:
                        continue
                    else:
                        keys = line[author_keys]
                        authors.update(keys)
                else:
                    continue
            elif "year" in line and start_year <= int(line["year"]) <= year:
                keys = line[author_keys]
                authors.update(keys)
    # If for venues, remove all old authors
    if venues:
        with open(source) as file:
            for line in file:
                line = json.loads(line)
                # Remove all old authors
                if "year" in line and start_year <= int(line["year"]) <= year:
                    keys = set(line[author_keys])
                    for key in keys:
                        authors.discard(key)
                        
    # Now get relevant info from window from
    # which all info is known
    with open(source) as file:
        for line in file:
            line = json.loads(line)
            if "year" in line and year < int(line["year"]) <= val_year:
                keys = set(line[author_keys])
                if venues:
                    new_keys = keys & authors
                else:
                    new_keys = keys - authors
                if new_keys:
                    old_authors = keys & authors_in_gcc
                    # Add edges between new authors
                    G_val.add_nodes_from(new_keys)
                    G_val.add_edges_from([(a, b)
                                          for a in new_keys for b in new_keys
                                          if a != b])
                    # Add new edges between old authors
                    G.add_edges_from([(encoding[a], encoding[b])
                                     for a in old_authors for b in old_authors
                                     if a != b])
                    # Store connections from new authors
                    # to old authors
                    for key in new_keys:
                        for old in old_authors:
                            connections_back[key].add(old)
    for node in G_val.nodes:
        if node not in connections_back:
            connections_back[node] = set()
    val_paper = {}
    val_author_paper = defaultdict(set)

    # Catch for all new authors
    # their papers in validation period
    with open(source) as file:
        for line in file:
            line = json.loads(line)
            if "year" in line and year < int(line["year"]) <= val_year:
                keys = line[author_keys]
                keys = [key for key in keys
                        if key in connections_back]

                # Fix Semantic Scholar doubled authors bug
                keys = sorted(list(set(keys)))

                if keys:
                    abstract = line["abstract"]
                    if title and "title" in line:
                        abstract = line["title"] + "\n" + abstract
                    abstract = abstract.replace("'", "")
                    val_paper[line["key"]] = abstract
                    for key in keys:
                        val_author_paper[key].add(line["key"])
    # Make test stuff
    test_paper = {}
    test_author_paper = defaultdict(set)
    with open(source) as file:
        for line in file:
            line = json.loads(line)
            if "year" in line and val_year < int(line["year"]):
                keys = line[author_keys]
                keys = [key for key in keys
                        if key in connections_back]

                # Fix Semantic Scholar doubled authors bug
                keys = sorted(list(set(keys)))

                if keys:
                    abstract = line["abstract"]
                    if title and "title" in line:
                        abstract = line["title"] + "\n" + abstract
                    test_paper[line["key"]] = abstract
                    for key in keys:
                        test_author_paper[key].add(line["key"])

    # Make processing of these extra data into new graph
    n_gcc = len(encoding)
    nodes = sorted(list(G_val.nodes))
    new_encoding = {node: i + n_gcc for i, node in enumerate(nodes)}
    G_val = nx.relabel_nodes(G_val, new_encoding)
    G_val = nx.union(G, G_val)
    connections_back = {new_encoding[key]: [encoding[x] for x in value]
                        for key, value in connections_back.items()}
    for key, value in connections_back.items():
        G_val.add_edges_from([(key, x) for x in value])

    val_author_paper = {new_encoding[key]: value
                        for key, value in val_author_paper.items()}
    test_author_paper = {new_encoding[key]: value
                         for key, value in test_author_paper.items()}
    # Make validation file
    result = pd.DataFrame(columns=["text",
                                   "author"])
    all_papers = set(val_paper.keys())

    # Make validation data
    # Go through in deterministic manner
    for key, value in val_author_paper.items():
        for x in value:
            result = result.append({"text": val_paper[x],
                                    "author": key,
                                    "label": 1},
                                   ignore_index=True)
        # Negative sampling
        author_papers = set(value)
        non_papers = sorted(list(all_papers - author_papers))
        negative_samples = random.sample(non_papers,
                                         len(author_papers))
        for value in negative_samples:
            result = result.append({"text": val_paper[value],
                                    "author": key,
                                    "label": 0},
                                   ignore_index=True)
    result.to_csv(val_dest,
                  quotechar="'")

    # Also make author document list for feature computing
    p_result = result[result["label"] == 1]
    adl = defaultdict(list)
    for _, line in p_result.iterrows():
        adl[line["author"]].append(line["text"])

    # Make Test file
    result = pd.DataFrame(columns=["text",
                                   "author",
                                   "label"])
    all_papers = set(test_paper.keys())

    # Positive examples
    for key, value in test_author_paper.items():
        for x in value:
            result = result.append({"text": test_paper[x],
                                    "author": key,
                                    "label": 1},
                                   ignore_index=True,)
        # Negative sampling
        author_papers = set(value)
        non_papers = sorted(list(all_papers - author_papers))
        negative_samples = random.sample(non_papers,
                                         len(author_papers))
        for value in negative_samples:
            result = result.append({"text": test_paper[value],
                                    "author": key,
                                    "label": 0},
                                   ignore_index=True,)
    result.to_csv(test_dest,
                  quotechar="'")

    # Dump other stuff
    with open(encoding_study, "w") as file:
        json.dump(new_encoding, file)
    with open(adl_dest, "w") as file:
        json.dump(adl, file)
    nx.write_adjlist(G_val, graph_dest)


if __name__ == "__main__":
    print("Start with GAI")
    new_authors()
    print("-----------------")
    print("Continue with KDD")
    new_authors(source="data/kdd/en_data.json",
                author_source="data/kdd/encoding.json",
                graph_source="data/kdd/graph.adjlist",
                author_keys="ai_authors_keys",
                dest="data/kdd/new_authors/",
                title=True,
                start_year=0,
                year=2015,
                val_year=2016,
                venues={"KDD"},
                seed=13)
