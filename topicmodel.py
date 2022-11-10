#download mallet: https://mallet.cs.umass.edu/download.php
#!pip install little_mallet_wrapper

import little_mallet_wrapper
from pathlib import Path
import random
import glob
import os

#import path to mallet and corpus
path_to_mallet = "./mallet-2.0.8/bin/mallet.bat"
path_to_corpus = "./txt/"

dir = os.path.join(path_to_corpus,"*.txt")

#open files and process text to training_data
training_data = []
def topic_model(dir):
    files = glob.glob(dir)
    for file in files:
        text = open(file, encoding='utf-8').read()
        processed_text = little_mallet_wrapper.process_string(text, numbers='remove')
        training_data.append(processed_text)

    #append original text for future reference
    """
    original_text = []
    for file in files:
        text = open(file, encoding='utf-8').read()
        original_text.append(text)
    """

    #create list of titles to label dataset
    titles = [Path(file).stem for file in files]

    #print general corpus stats
    little_mallet_wrapper.print_dataset_stats(training_data)

    #set # of topics, training output path
    num_topics = 3
    output_directory_path = 'topic-model-output/txt'

    Path(f"{output_directory_path}").mkdir(parents=True, exist_ok=True)

    path_to_training_data           = f"{output_directory_path}/training.txt"
    path_to_formatted_training_data = f"{output_directory_path}/mallet.training"
    path_to_model                   = f"{output_directory_path}/mallet.model.{str(num_topics)}"
    path_to_topic_keys              = f"{output_directory_path}/mallet.topic_keys.{str(num_topics)}"
    path_to_topic_distributions     = f"{output_directory_path}/mallet.topic_distributions.{str(num_topics)}"

    #train model
    little_mallet_wrapper.quick_train_topic_model(path_to_mallet,
                                                output_directory_path,
                                                num_topics,
                                                training_data)

    #render topic_distributions
    topics = little_mallet_wrapper.load_topic_keys(path_to_topic_keys)
    subject = []
    for topic_number, topic in enumerate(topics):
        print(f"topic {topic_number}\n\n{topic}\n")
    topic_distributions = little_mallet_wrapper.load_topic_distributions(path_to_topic_distributions)
    training_data_obit_titles = dict(zip(training_data, titles))

    #display top tiles per topic
    def top_titles_per_topic(topic_number=0, number_of_documents=5):
        print(f"topic {topic_number}\n\n{topics[topic_number]}\n")
        for probability, document in little_mallet_wrapper.get_top_docs(training_data, topic_distributions, topic_number, n=number_of_documents):
            print(round(probability, 4), training_data_obit_titles[document] + "\n")
        return

#results
topic_model(dir)
#topic_model.top_titles_per_topic(topic_number=0, number_of_documents=5)

#topic summaries by GPT-3:
#The fishing industry is a vital part of the economy and it relies on a number of resources, such as rights to fishing grounds, management of fisheries, and property rights. In order to manage these resources sustainably, it is important to account for bycatch, mortality, and economic incentives.
#The global fisheries industry is worth over $100 billion per year and employs over 1.2 million people. Overfishing, data management, and ecosystem conservation are important issues facing the fishing industry. The Eastern reefs are a major area for fishing and are home to many different types of fish, coral, and other marine life.
#The production and consumption of seafood is a global industry with significant economic and environmental impacts. Several factors, such as growing demand from developing countries, expanding aquaculture production, and rising fish stock levels, are contributing to growth in the seafood market.