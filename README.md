# Summary

This is the source code for the paper "A Case Study and Qualitative Analysis of Simple Cross-Lingual Opinion Mining", which was accepted as full paper with oral presentation at the 13th International [Conference on Knowledge Discovery and Information Retrieval (KDIR)](http://www.kdir.ic3k.org/).


# Documentation

- [XLING-simple-example.ipynb](https://github.com/ghagerer/cross-lingual-opinion-mining/blob/master/XLING-simple-example.ipynb): shows simple examples for converting English and German sentences into sentence embeddings and the cosine similarity between them.
- [article_based_topic_modeling_review.ipynb](https://github.com/ghagerer/cross-lingual-opinion-mining/blob/master/article_based_topic_modeling_review.ipynb): demonstrates how sentences in a single article are clustered into different topics and summarised the corresponding sentiment distribution.
- [cosine_similarity.ipynb](https://github.com/ghagerer/cross-lingual-opinion-mining/blob/master/cosine_similarity.ipynb): analyses the distribution of cosine similarity of sentences per topic.
- [create_article_wise_csv.ipynb](https://github.com/ghagerer/cross-lingual-opinion-mining/blob/master/create_article_wise_csv.ipynb): creates data file which contains sentiments and topic assignment for every sentence in a single article for further analysis.
- [radar_plot_final.ipynb](https://github.com/ghagerer/cross-lingual-opinion-mining/blob/master/radar_plot_final.ipynb): displays topic distribution per data source and document type.
- [radarfactory.py](https://github.com/ghagerer/cross-lingual-opinion-mining/blob/master/radarfactory.py): contains Python class for generating radar plot.
- [regenerate_sentences_metadata_json.ipynb](https://github.com/ghagerer/cross-lingual-opinion-mining/blob/master/regenerate_sentences_metadata_json.ipynb): helps to gather all related measurements and create an overall data file for analysis.
- [sankey_plot_final.ipynb](https://github.com/ghagerer/cross-lingual-opinion-mining/blob/master/sankey_plot_final.ipynb): generates the final sankey plot which shows the flow of topic distribution for increasing number of topics for a fixed among of input sentences.
- [sankey_plot_k_clusters.ipynb](https://github.com/ghagerer/cross-lingual-opinion-mining/blob/master/sankey_plot_k_clusters.ipynb): creates the first version of sankey plots.
- [sentence_posting_time.ipynb](https://github.com/ghagerer/cross-lingual-opinion-mining/blob/master/sentence_posting_time.ipynb): handles minor issues on sentences posting time.
- [senti_util.py](https://github.com/ghagerer/cross-lingual-opinion-mining/blob/master/senti_util.py): includes utility class for sentiment analysis.
- [sentiment.py](https://github.com/ghagerer/cross-lingual-opinion-mining/blob/master/sentiment.py): is a Python class for assigning SentiWordNet sentiment.
- [sentiwordnet_vs_textblob.ipynb](https://github.com/ghagerer/cross-lingual-opinion-mining/blob/master/sentiwordnet_vs_textblob.ipynb): shows distributions of Textblob sentiment and SentiWordNet sentiment.
- [simple_distribution_of_sentiment.ipynb](https://github.com/ghagerer/cross-lingual-opinion-mining/blob/master/simple_distribution_of_sentiment.ipynb): includes detail version of sentiment distribution.
- [time_related_distribution_of_sentiment.ipynb](https://github.com/ghagerer/cross-lingual-opinion-mining/blob/master/time_related_distribution_of_sentiment.ipynb): indicates the change of sentiment of news and that of the responses from readers.
- [top_sentences_per_clusters.ipynb](https://github.com/ghagerer/cross-lingual-opinion-mining/blob/master/top_sentences_per_clusters.ipynb): lists the top sentences per topic.
- [top_words.ipynb](https://github.com/ghagerer/cross-lingual-opinion-mining/blob/master/top_words.ipynb): records top words per topic.
- [util.py](https://github.com/ghagerer/cross-lingual-opinion-mining/blob/master/util.py): is a Python utility class with simple functions for analysis.
