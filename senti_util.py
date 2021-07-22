import json
import pandas as pd
import pprint
import numpy as np
from termcolor import colored, cprint

# !pip install arch
from arch.unitroot import DFGLS
from scipy.stats import pearsonr

import util
from util import DocType, Source, OptimalKClustersConfig

def get_sentence_cluster_sentiment_df(start_year = 2009, end_year = 2017, path = '/content/drive/My Drive/Colab Notebooks/media-agenda/data/sentence_cluster_sentiment_dict.json', verbose = True):

  with open(path, 'r') as f:
    assignments = json.load(f)

  if verbose:
    print('Loaded total {} cluster-sentiment assignments'.format(len(assignments)))
    print('Example of an assignment object')
    print()

  # Convert assignments into panda dataframe
  df = pd.DataFrame.from_dict(assignments)
  df['is_comment'] = df['comment_id'] != DocType.NON_COMMENT.value
  df['posting_time'] = pd.to_datetime(df['posting_time'])

  if verbose:
    print('Number of article from:')
    display(pd.DataFrame([('nytimes', len(df[df.source == 'nytimes'].article_id.unique())),
                          ('quora', len(df[df.source == 'quora'].article_id.unique())),
                          ('spiegel', len(df[df.source == 'spiegel'].article_id.unique())),
                          ('total', len(df[df.source == 'nytimes'].article_id.unique()) + len(df[df.source == 'quora'].article_id.unique()) + len(df[df.source == 'spiegel'].article_id.unique()))],
                        columns = ['source', 'num_of_article']))
    print()

  # Update the article posting time as its earliest comment posting time if there is any comment posting is earlier than the original article time
  article_posting_time_df = df[df.is_comment == False].groupby(['source', 'article_id']).posting_time.min().reset_index()
  comment_earliest_posting_time_df = df[df.is_comment == True].groupby(['source', 'article_id']).posting_time.min().reset_index()
  merged_df = pd.merge(article_posting_time_df, comment_earliest_posting_time_df, how = 'outer', on = ['source', 'article_id'], validate = 'one_to_one')

  incorrect_dated_article = []
  for index, row in merged_df.iterrows():
    if row['posting_time_x'].date() > row['posting_time_y'].date():
      incorrect_dated_article.append((row['source'], row['article_id'], row['posting_time_x'], row['posting_time_y'].date()))
      df.loc[(df.is_comment == False) & (df.source == row['source']) & (df.article_id == row['article_id']), 'posting_time'] = row['posting_time_y'].date()

  # create new columns
  df['posting_time'] = pd.to_datetime(df['posting_time'])
  df['date'] = df['posting_time'].dt.date
  df['month'] = df['posting_time'].dt.month
  df['year'] = df['posting_time'].dt.year

  # Filter out unnecessary the assignment records
  # 1. Records from garbage clusters
  # 2. Records not falls between 2009 and 2017
  # pprint.pprint(pd.unique(df['cluster']))
  df = df[~df['cluster'].isin(OptimalKClustersConfig.garbage_clusters)]

  if verbose:
    print('Unique cluster after filtering:')
    pprint.pprint(pd.unique(df['cluster']))
    print()

  start_datetime, end_datetime, start_datetime_str, end_datetime_str = util.get_start_end_datetime(start_year, end_year)

  df = df[(df['date'] >= start_datetime) & (df['date'] <= end_datetime)]
  df = df.reset_index()
  if verbose:
    print('Min year after filtering:', min(df['year']))
    print('Max yearafter filtering:', max(df['year']))
    print('Number of sentences after filtering:', df.shape[0])
    display(pd.DataFrame(columns = ['Source', 'Total sentences', 'Article sentences', 'Comment sentences'],
                data = [[Source.NYTIMES, df[df.source == Source.NYTIMES].shape[0], 
                          df[(df.source == Source.NYTIMES) & (df.is_comment == False)].shape[0],
                          df[(df.source == Source.NYTIMES) & (df.is_comment == True)].shape[0]], 
                        [Source.QUORA, df[df.source == Source.QUORA].shape[0], 
                          df[(df.source == Source.QUORA) & (df.is_comment == False)].shape[0],
                          df[(df.source == Source.QUORA) & (df.is_comment == True)].shape[0]],
                        [Source.SPIEGEL, df[df.source == Source.SPIEGEL].shape[0], 
                          df[(df.source == Source.SPIEGEL) & (df.is_comment == False)].shape[0],
                          df[(df.source == Source.SPIEGEL) & (df.is_comment == True)].shape[0]],]))
    print()
    print('Display top 5 row of the preprocessed dataframe:')
    display(df.head())
    print()

  if verbose:
    print('Reference:')
    print('Table showing the articles with posting time updated ')
    display(pd.DataFrame(incorrect_dated_article, columns = ['source', 'article_id', 'changed from', 'to']))
    print()
    print('Current values for the 12th article above:')
    print('Source:', incorrect_dated_article[12][0], ' Article_id:', incorrect_dated_article[12][1])
    display(df[(df.is_comment == False) & (df.source == incorrect_dated_article[12][0]) & (df.article_id == incorrect_dated_article[12][1])])

  return df

def get_df_with_time_interval_indexing(dataframe, start_year, end_year, interval = '364D', column_name = 'bin_of_every_364_days'):
  # start_datetime, end_datetime, _, _ = util.get_start_end_datetime(start_year = start_year, end_year = end_year)
  start_datetime, _, _, _ = util.get_start_end_datetime(start_year = start_year, end_year = start_year)
  end_datetime, _, _, _ = util.get_start_end_datetime(start_year = end_year + 1, end_year = end_year + 1)
  # date_list = pd.date_range(start = start_datetime, end = end_datetime, freq = interval, normalize = True)
  interval_list = pd.interval_range(start = start_datetime, end = end_datetime, freq = interval, closed = 'left')

  dataframe = dataframe.sort_values(by = 'date')
  dataframe[column_name] = -1

  for index, row in dataframe.iterrows():
    for i in range(interval_list.size):
      if pd.to_datetime(row['date']).date() in interval_list[i]:
        # print(row['date'], pd.to_datetime(row['date']).date(), interval_list[i])
        dataframe.at[index, column_name] = i
        break
    if dataframe.at[index, column_name] == -1:
      dataframe.at[index, column_name] = interval_list.size

  return dataframe, interval_list


def get_sentiment_stat(df, is_textblob, bin, is_by_cluster = False):
  if not is_by_cluster: # statistic for certain source
    if is_textblob: # with textblob (EN) or textblob-de (DE) for sentiment assignment
      stat_df = pd.DataFrame({'mean': df[df.textblob_subjectivity != 0].groupby(by = [bin]).textblob_polarity.mean(), 
                              'median': df[df.textblob_subjectivity != 0].groupby(by = [bin]).textblob_polarity.median(),
                              'first_quantile': df[df.textblob_subjectivity != 0].groupby(by = [bin]).textblob_polarity.quantile(0.25),
                              'third_quantile': df[df.textblob_subjectivity != 0].groupby(by = [bin]).textblob_polarity.quantile(0.75),
                              'groupby_count': df[df.textblob_subjectivity != 0].groupby(by = [bin]).textblob_polarity.count(),
                              'total_count': df[df.textblob_subjectivity != 0].shape[0]}).reset_index()
    else:  # with SentiWordNet (EN) or SentiWS (DE) for sentiment assignment
      stat_df = pd.DataFrame({'mean': df.groupby(by = [bin]).sentiw_sentiment.mean(), 
                              'median': df.groupby(by = [bin]).sentiw_sentiment.median(),
                              'first_quantile': df.groupby(by = [bin]).sentiw_sentiment.quantile(0.25),
                              'third_quantile': df.groupby(by = [bin]).sentiw_sentiment.quantile(0.75),
                              'groupby_count': df.groupby(by = [bin]).sentiw_sentiment.count(),
                              'total_count': df.shape[0]}).reset_index()

    stat_df['iqr'] = stat_df.third_quantile - stat_df.first_quantile
    stat_df['dominance'] = (stat_df.groupby_count / stat_df.total_count * 100)
    stat_df['hotness'] = stat_df.iqr * stat_df.dominance

  else: # statistic for certain source and by clusters
    if is_textblob:
      stat_df = pd.DataFrame({'mean': df[df.textblob_subjectivity != 0].groupby(by = ['cluster', bin]).textblob_polarity.mean(), 
                              'median': df[df.textblob_subjectivity != 0].groupby(by = ['cluster', bin]).textblob_polarity.median(),
                              'first_quantile': df[df.textblob_subjectivity != 0].groupby(by = ['cluster', bin]).textblob_polarity.quantile(0.25),
                              'third_quantile': df[df.textblob_subjectivity != 0].groupby(by = ['cluster', bin]).textblob_polarity.quantile(0.75),
                              'groupby_count': df[df.textblob_subjectivity != 0].groupby(by = ['cluster', bin]).textblob_polarity.count(),
                              'total_count': df[df.textblob_subjectivity != 0].shape[0]}).reset_index()

      tmp_df = pd.DataFrame({'cluster_total_count':  df[df.textblob_subjectivity != 0].groupby(by = ['cluster']).textblob_polarity.count()}).reset_index()
    else: 
      stat_df = pd.DataFrame({'mean': df.groupby(by = ['cluster', bin]).sentiw_sentiment.mean(), 
                              'median': df.groupby(by = ['cluster', bin]).sentiw_sentiment.median(),
                              'first_quantile': df.groupby(by = ['cluster', bin]).sentiw_sentiment.quantile(0.25),
                              'third_quantile': df.groupby(by = ['cluster', bin]).sentiw_sentiment.quantile(0.75),
                              'groupby_count': df.groupby(by = ['cluster', bin]).sentiw_sentiment.count(),
                              'total_count': df.shape[0]}).reset_index()
      

      
      tmp_df = pd.DataFrame({'cluster_total_count':  df[df.textblob_subjectivity != 0].groupby(by = ['cluster']).sentiw_sentiment.count()}).reset_index()

    stat_df = pd.merge(stat_df, tmp_df, how = 'left', on = 'cluster', suffixes=('', '_y'))
    stat_df['iqr'] = stat_df.third_quantile - stat_df.first_quantile
    stat_df['dominance'] = (stat_df.groupby_count / stat_df.cluster_total_count * 100)
    stat_df['hotness'] = stat_df.iqr * stat_df.dominance
    stat_df['overall_dominance'] = (stat_df.groupby_count / stat_df.total_count * 100)
    stat_df['overall_hotness'] = stat_df.iqr * stat_df.overall_dominance
    
  return stat_df


def get_pearson_correlation(x, y):
  # remove rows with any NaN, since cross-lagged correlation considers any arbitrary pair of A_x and C_y, does it make sense??
  x = np.array(x)
  y = np.array(y)
  nas = np.logical_or(np.isnan(x), np.isnan(y))
  if len(x[~nas]) <= 2 or len(y[~nas]) <= 2:
      cprint('Warn: Pearson correlation cannot be calculated with just a pair of data.', 'red')
      return np.nan, np.nan, np.nan
  r, p_val = pearsonr(x[~nas], y[~nas])

  valid_pairs = len(nas[~nas])
  return r, p_val, valid_pairs

def get_cross_lagged_correlation(x, y, shift = 1):
  return get_pearson_correlation(x[:-1], y.shift(-1)[:-1])


def is_stationary(timeseries):
  # print(DFGLS(timeseries).summary().as_text())
  return True if DFGLS(timeseries).pvalue < 0.05 else False


def get_date_list(start_datetime, end_datetime, freq = '14D'):
  return pd.date_range(start = start_datetime, end = end_datetime, freq = freq, normalize = True, closed = 'left')

def get_time_interval_list(start_datetime, end_datetime, freq = '14D'):
  return pd.interval_range(start = start_datetime, end = end_datetime, freq = freq, closed = 'left')