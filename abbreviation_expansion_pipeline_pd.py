# -*- coding: utf-8 -*-

# Checks for required Python packages and installs them if not already installed.

# pip3 install difflib
# pip3 install fuzzywuzzy
# pip3 install jellyfish
# pip3 install nltk
# pip3 install pandas
# pip3 install numpy
# pip3 install pandarallel


# Import installed packages.
import time
import gc
from typing import List,Pattern
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
import re
import string
from difflib import SequenceMatcher as difflib_sequencematcher
from fuzzywuzzy import fuzz
import jellyfish
import pandas as pd
import numpy as np
import nltk
nltk.download(['punkt', 'stopwords'])
from nltk import word_tokenize
from nltk.util import ngrams
import os
cpu_count = os.cpu_count()
cpu_count:int = cpu_count if cpu_count else 1
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True,nb_workers=cpu_count)

class AbbreviationExpansionPipelinePD():
  """
  A class for generating n-gram pairs from product descriptions in a DataFrame and suggesting mappings for text expansion.
  """
  def __init__(self,dataframe_object:pd.DataFrame,product_desc_column:str,ngram:int=2,output_file_name:str='Mined_Keyword_Mapping',cosine_threshold:float=0.73,min_text_match_threshold:float=85.0,):
    """
    Initialize the AbbreviationExpansionPipelinePD class.

    Args:
    - dataframe (pd.DataFrame): Input pandas DataFrame object containing data.
    - product_desc_column (str): Name of the column in 'dataframe' containing product descriptions.
    - ngram (int, optional), default = 2: Number of words in each n-gram.
    - output_file_name (str, optional), default = 'Mined_Keyword_Mapping': Name of the output file where results will be saved.
    - cosine_threshold (float, optional), default = 0.73: Threshold value for Cosine Similarity.
    - min_text_match_threshold (float, optional), default = 85.0: Minimum Threshold value for Text Similarity.
    """
    self.working_df:pd.DataFrame = dataframe_object
    self.product_desc_column:str = product_desc_column
    self.ngram_pair:int = ngram
    self.output_file_name:str = output_file_name
    self.cosine_threshold = cosine_threshold
    self.min_text_match_threshold = min_text_match_threshold

  def __repr__(self):
    """
    Returns a string representation of the class instance.
    """
    return f"AbbreviationExpansionPipelinePD()"

  def __str__(self):
    """
    Returns a description of the class.
    """
    return "A class for generating n-gram pairs from product descriptions in a DataFrame and suggesting mappings for text expansion."

  def get_fuzz_partial_ratio(self,sent_1:str,sent_2:str)->float:
    """
    Computes the partial fuzz ratio between two strings using partial matching.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: similarity score
    """
    return fuzz.partial_ratio(sent_1,sent_2)

  def get_jaro_winkler_similarity(self,sent_1:str,sent_2:str)->float:
    """
    Calculates the Jaro-Winkler similarity between two strings.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: similarity score (percentage).
    """
    return (jellyfish.jaro_winkler_similarity(sent_1,sent_2))*100

  def get_difflib_sequencematcher(self,sent_1:str,sent_2:str)->float:
    """
    Computes the similarity ratio between two strings using difflib's SequenceMatcher.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: Similarity ratio between the two strings, scaled to percentage (0 to 100).
      The ratio measures how similar the sequences are, where 100 means identical.
    """
    return difflib_sequencematcher(isjunk=None,autojunk=True,a=sent_1,b=sent_2).ratio()*100

  def get_fuzz_token_set_ratio(self,sent_1:str,sent_2:str)->float:
    """
    Computes the fuzz ratio between two strings using token sets.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: similarity score
    """
    return fuzz.token_set_ratio(sent_1,sent_2)

  def get_jaro_similarity(self,sent_1:str,sent_2:str)->float:
    """
    Computes the Jaro similarity between two strings.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: Jaro similarity score between the two strings.
      The score ranges from 0 (no similarity) to 1 (exact match),
      measuring the similarity between the strings based on character matching.
    """
    return (jellyfish.jaro_similarity(sent_1,sent_2))*100

  def trim_characters(self,stxt:str)->str:
    """
    Removes non-alphanumeric characters from a string.

    Args:
    - stxt (str): Input string.

    Returns:
    - str: String with non-alphanumeric characters removed.
    """
    stxt:str = str(stxt).lower().strip()
    stxt:str = stxt.replace('w/',' with ')
    stxt:str = re.compile(pattern=r'[^a-z]').sub(repl=r' ',string=str(stxt))
    stxt:str = re.compile(pattern=r'\s+').sub(repl=' ',string=stxt).strip()
    return stxt

  def generate_ngrams(self,txt:str='',ngram_count:int=2,is_lower:bool=True,remove_characters:bool=True)->List[str]:
    """
    Pre-processes text values by lowercasing, removing non-alphanumeric characters, and tokenizing.

    Args:
    - txt (str, optional): Input text.
    - ngram_count (int, optional), default = 2: Size of n-grams.
    - is_lower (bool, optional), default = True: Convert text to lowercase.
    - remove_characters (bool, optional), default = True: Remove non-alphanumeric characters.

    Returns:
    - List[str]: List of n-grams generated from the processed text.
    """
    if is_lower:
      txt:str=str(txt).lower().strip()
    else:
      txt:str=str(txt).strip()

    if remove_characters:
      txt:str=self.trim_characters(stxt=txt)
    else:
      pass

    return list(ngrams(sequence=[x for x in txt.split(' ') if ((x.isalnum()) and (len(x)>1))],n=ngram_count))

  def get_ngrams(self,row1:pd.Series,col_name:str,ngram:int=2,lower_case:bool=True,remove_punctuations:bool=True)->pd.Series:
    """
    Generates n-grams for a specific column in a DataFrame row.

    Args:
    - row1 (pd.Series): DataFrame row containing text data.
    - col_name (str): Name of the column containing text data.
    - ngram (int, optional), default = 2 "for bigrams": Number of words in each n-gram.
    - lower_case (bool, optional), default = True: Convert text to lowercase.
    - remove_punctuations (bool, optional), default = True: Remove punctuation characters.

    Returns:
    - pd.Series: Original DataFrame row with an added 'ngrams' list column containing generated n-grams.
    """
    row1['ngrams']:list = self.generate_ngrams(txt=row1[col_name],ngram_count=ngram,is_lower=lower_case,remove_characters=remove_punctuations)
    return row1

  def get_ngrams_summary(self,df:pd.DataFrame)->pd.DataFrame:
    """
    Generates a summary DataFrame showing the count and availability percentage of each n-gram in the input DataFrame.

    Args:
    - df (pd.DataFrame): Input DataFrame containing text data.

    Returns:
    - pd.DataFrame: Summary DataFrame with columns 'Context', 'ContextCount', 'CorpusCount', and 'PerAvailability'.
    """
    ngram_counter = Counter(df.explode().dropna())

    summary_df:pd.DataFrame = pd.DataFrame(data={'Context': list(ngram_counter.keys()),})

    # single core processing
    # summary_df['Context']:pd.Series = summary_df['Context'].apply(lambda x: ' '.join(x))

    # using pandarallel for multiprocessing
    summary_df['Context']:pd.Series = summary_df['Context'].parallel_apply(lambda x: ' '.join(x))

    return summary_df

  def replace_bounded_string(self,stxt:str,to_replace:str,replace_with:str)->str:
    """
    Replace 'to_replace' with 'replace_with' in 'stxt', matching whole words.

    Args:
    - stxt (str): The input string where replacements are to be made.
    - to_replace (str): The string to be replaced.
    - replace_with (str): The string to replace 'to_replace'.

    Returns:
    - str: The modified string after replacements.
    """
    return re.compile(pattern=r'\b'+re.escape(to_replace)+r'\b',flags=re.IGNORECASE|re.VERBOSE).sub(repl=replace_with,string=str(stxt))

  def get_replacement(self,item_1:str,item_2:str)->dict:
    """
    Generate replacements between item_1 and item_2, and returns the mappings.

    Args:
    - item_1 (str): The first string for replacement.
    - item_2 (str): The second string for replacement.

    Returns:
    - dict: A dictionary mapping the replacements.
    """
    item_1_list:list=item_1.split(' ')
    item_2_list:list=item_2.split(' ')
    common_keyword:set=set(item_1_list).intersection(set(item_2_list))
    item_1_list:list=[x for x in item_1_list if x not in common_keyword]
    item_2_list:list=[x for x in item_2_list if x not in common_keyword]
    mapped_results:dict={}
    for i1 in item_1_list:
      result1:str=''
      best_score:float=self.min_text_match_threshold
      for i2 in item_2_list:
        sc1:float = self.get_fuzz_partial_ratio(sent_1=i1,sent_2=i2)
        sc2:float = self.get_jaro_winkler_similarity(sent_1=i1,sent_2=i2)
        current_score:float = min(sc1,sc2)
        if best_score >= current_score:
          continue
        else:
          best_score:float = current_score
          result1:str = i2
      if result1:
        if len(str(i1))>len(str(result1)):
          mapped_results[result1]=i1
        else:
          mapped_results[i1]=result1
    return mapped_results

  def main(self)->None:
    """
    Orchestrates the process of generating n-gram pairs, computing cosine similarity,
    and suggesting mappings for text expansion based on product descriptions in the input DataFrame.
    """
    start_time:float = time.time()

    # Dropping Duplicates
    self.working_df:pd.DataFrame = self.working_df[[self.product_desc_column]].drop_duplicates().dropna()

    # Generate ngrams ======

    # single core processing
    # ngrams_df:pd.DataFrame = self.working_df.apply(lambda x: self.get_ngrams(row1=x,col_name=self.product_desc_column,ngram=self.ngram_pair,lower_case=True,remove_punctuations=True),axis=1)

    # using pandarallel for multiprocessing
    ngrams_df:pd.DataFrame = self.working_df.parallel_apply(lambda x: self.get_ngrams(row1=x,col_name=self.product_desc_column,ngram=self.ngram_pair,lower_case=True,remove_punctuations=True),axis=1)

    del self.working_df
    gc.collect()

    ngrams_df_status:pd.DataFrame = self.get_ngrams_summary(df=ngrams_df['ngrams'])
    ngrams_df_status:pd.DataFrame = ngrams_df_status[['Context']].drop_duplicates().dropna()

    print('\u2501'*35)
    print('\u2503','{:^31}'.format(f'Content Found: {ngrams_df_status.shape[0]}'),'\u2503')
    print('\u2501'*35)

    # get Common Elements between the two documents ======
    ngrams_df_status['tokens']:pd.Series = ngrams_df_status['Context'].str.split(' ')
    ngrams_df_status:pd.DataFrame = ngrams_df_status.explode(column='tokens')
    working_result_df:pd.DataFrame = pd.merge(left=ngrams_df_status,right=ngrams_df_status,on=['tokens'],suffixes=('_1', '_2'),how='inner')
    working_result_df:pd.DataFrame = working_result_df[['Context_1','Context_2']]
    working_result_df['Score']:pd.Series = 100
    del ngrams_df_status
    gc.collect()

    print('\u2501'*35)
    print('\u2503','{:^31}'.format(f'Mapped Content Found: {working_result_df.shape[0]}'),'\u2503')
    print('\u2501'*35)

    working_result_df:pd.DataFrame = working_result_df.rename(columns={'Context_1':'doc1_elements','Context_2':'doc2_elements',})
    working_result_df:pd.DataFrame = working_result_df[working_result_df['doc1_elements'] != working_result_df['doc2_elements']]
    working_result_df[['doc1_elements','doc2_elements']] = pd.DataFrame(np.sort(working_result_df[['doc1_elements','doc2_elements']],axis=1),index=working_result_df.index)
    working_result_df:pd.DataFrame = working_result_df.drop_duplicates()

    # Processing - Similarity Match ======
    working_result_df['high_score_difflib_sequencematcher']:pd.Series = working_result_df[['doc1_elements','doc2_elements']].parallel_apply(lambda x: self.get_difflib_sequencematcher(sent_1=x[0],sent_2=x[1]),axis=1)
    working_result_df:pd.DataFrame = working_result_df[working_result_df['high_score_difflib_sequencematcher']>29.99]
    print('\u2501'*35)
    print('\u2503','{:^31}'.format(f'After Filter Operation 1: {working_result_df.shape[0]}'),'\u2503')
    print('\u2501'*35)

    working_result_df['high_score_fuzz_token_set_ratio']:pd.Series = working_result_df[['doc1_elements','doc2_elements']].parallel_apply(lambda x: self.get_fuzz_token_set_ratio(sent_1=x[0],sent_2=x[1]),axis=1)
    working_result_df:pd.DataFrame = working_result_df[working_result_df['high_score_fuzz_token_set_ratio']>29.99]
    print('\u2501'*35)
    print('\u2503','{:^31}'.format(f'After Filter Operation 2: {working_result_df.shape[0]}'),'\u2503')
    print('\u2501'*35)

    working_result_df['high_score_jaro_similarity']:pd.Series = working_result_df[['doc1_elements','doc2_elements']].parallel_apply(lambda x: self.get_jaro_similarity(sent_1=x[0],sent_2=x[1]),axis=1)
    working_result_df:pd.DataFrame = working_result_df[working_result_df['high_score_jaro_similarity']>29.99]
    print('\u2501'*35)
    print('\u2503','{:^31}'.format(f'After Filter Operation 3: {working_result_df.shape[0]}'),'\u2503')
    print('\u2501'*35)

    working_result_df['high_score_fuzz_partial_ratio']:pd.Series = working_result_df[['doc1_elements','doc2_elements']].parallel_apply(lambda x: self.get_fuzz_partial_ratio(sent_1=x[0],sent_2=x[1]),axis=1)
    working_result_df:pd.DataFrame = working_result_df[working_result_df['high_score_fuzz_partial_ratio']>29.99]

    # Processing - Filtering Similarity Match Output ======
    print('\u2501'*35)
    print('\u2503','{:^31}'.format(f'Potential Matches Found: {working_result_df.shape[0]}'),'\u2503')
    print('\u2501'*35)

    for c_name in working_result_df.columns:
      if c_name not in ['doc1_elements','doc2_elements',]:
        working_result_df[c_name]:pd.Series = working_result_df[c_name].astype(float)
      else:
        continue
    del c_name

    # Processing - Text Replacement ======

    # single core processing
    # working_result_df['SUGGESTED_MAPPING']:pd.Series = working_result_df[['doc1_elements','doc2_elements']].apply(lambda x: self.get_replacement(item_1=x[0],item_2=x[1]),axis=1)
    # working_result_df['replacement_flag']:pd.Series = working_result_df['SUGGESTED_MAPPING'].apply(lambda x: True if x else False)

    # using pandarallel for multiprocessing
    working_result_df['SUGGESTED_MAPPING']:pd.Series = working_result_df[['doc1_elements','doc2_elements']].parallel_apply(lambda x: self.get_replacement(item_1=x[0],item_2=x[1]),axis=1)
    working_result_df['replacement_flag']:pd.Series = working_result_df['SUGGESTED_MAPPING'].parallel_apply(lambda x: True if x else False)

    working_result_df:pd.DataFrame = working_result_df[working_result_df['replacement_flag']==True]
    working_result_df:pd.DataFrame = working_result_df[['doc1_elements','doc2_elements','Score','SUGGESTED_MAPPING']]
    working_result_df.columns = ['PROD_DESC1','PROD_DESC1','SIMILARITY_SCORE','SUGGESTED_MAPPING']
    working_result_df['SIMILARITY_SCORE']:pd.Series = working_result_df['SIMILARITY_SCORE'].astype(float).round(0)
    working_result_df['SIMILARITY_SCORE']:pd.Series = working_result_df['SIMILARITY_SCORE'].astype(int)

    # writing file with abbreviation expansion suggestion with examples
    working_result_df.to_csv(path_or_buf=self.output_file_name+'_Examples.csv',index=False,encoding='latin-1',mode='w',header=True,)
    print('\u2501'*43)
    print('\u2503','{:^31}'.format(f'Abbreviation Expansion with Examples: {working_result_df.shape[0]}'),'\u2503')
    print('\u2501'*43)

    # single core processing
    # keyword_mapping_df:pd.DataFrame = pd.DataFrame(data={
    #   'TO_REPLACE':working_result_df['SUGGESTED_MAPPING'].apply(lambda x: list(x.keys())).explode().reset_index(drop=True),
    #   'REPALCE_WITH':working_result_df['SUGGESTED_MAPPING'].apply(lambda x: list(x.values())).explode().reset_index(drop=True),
    #   }).dropna().drop_duplicates().reset_index(drop=True)

    # using pandarallel for multiprocessing
    keyword_mapping_df:pd.DataFrame = pd.DataFrame(data={
      'TO_REPLACE':working_result_df['SUGGESTED_MAPPING'].parallel_apply(lambda x: list(x.keys())).explode().reset_index(drop=True),
      'REPALCE_WITH':working_result_df['SUGGESTED_MAPPING'].parallel_apply(lambda x: list(x.values())).explode().reset_index(drop=True),
      }).dropna().drop_duplicates().reset_index(drop=True)

    keyword_mapping_df['TO_REPLACE_LENGTH']:pd.Series = keyword_mapping_df['TO_REPLACE'].str.len()
    keyword_mapping_df:pd.DataFrame = keyword_mapping_df.sort_values(by=['TO_REPLACE','TO_REPLACE_LENGTH',],ascending=[True,True,])
    keyword_mapping_df:pd.DataFrame = keyword_mapping_df[['TO_REPLACE','REPALCE_WITH']]

    # single core processing
    # keyword_mapping_df['SIMILARITY_SCORE_1']:pd.Series = keyword_mapping_df[['TO_REPLACE','REPALCE_WITH']].apply(lambda x: self.get_fuzz_partial_ratio(sent_1=x[0],sent_2=x[1]),axis=1)
    # keyword_mapping_df['SIMILARITY_SCORE_2']:pd.Series = keyword_mapping_df[['TO_REPLACE','REPALCE_WITH']].apply(lambda x: self.get_jaro_winkler_similarity(sent_1=x[0],sent_2=x[1]),axis=1)

    # using pandarallel for multiprocessing
    keyword_mapping_df['SIMILARITY_SCORE_1']:pd.Series = keyword_mapping_df[['TO_REPLACE','REPALCE_WITH']].parallel_apply(lambda x: self.get_fuzz_partial_ratio(sent_1=x[0],sent_2=x[1]),axis=1)
    keyword_mapping_df['SIMILARITY_SCORE_2']:pd.Series = keyword_mapping_df[['TO_REPLACE','REPALCE_WITH']].parallel_apply(lambda x: self.get_jaro_winkler_similarity(sent_1=x[0],sent_2=x[1]),axis=1)

    keyword_mapping_df['SIMILARITY_SCORE_1']:pd.Series = keyword_mapping_df['SIMILARITY_SCORE_1'].round(0)
    keyword_mapping_df['SIMILARITY_SCORE_2']:pd.Series = keyword_mapping_df['SIMILARITY_SCORE_2'].round(0)
    keyword_mapping_df['SIMILARITY_SCORE_1']:pd.Series = keyword_mapping_df['SIMILARITY_SCORE_1'].astype(int)
    keyword_mapping_df['SIMILARITY_SCORE_2']:pd.Series = keyword_mapping_df['SIMILARITY_SCORE_2'].astype(int)

    # writing file with abbreviation expansion suggestion without examples
    keyword_mapping_df.to_csv(path_or_buf=self.output_file_name+'_Mapping.csv',index=False,encoding='latin-1',mode='w',header=True,)
    print('\u2501'*42)
    print('\u2503','{:^31}'.format(f'Unique Abbreviation Expansion Found: {keyword_mapping_df.shape[0]}'),'\u2503')
    print('\u2501'*42)

    del working_result_df,keyword_mapping_df
    gc.collect()

    print(f"Elapsed time: {((time.time() - start_time) / 60):.2f} minutes")
    return None

def custom_ram_cleanup_func()->None:
  """
  Clean up global variables except for specific exclusions and system modules.

  This function deletes all global variables except those specified in
  `exclude_vars` and variables starting with underscore ('_').

  Excluded variables:
  - Modules imported into the system (except 'sys' and 'os')
  - 'sys', 'os', and 'custom_ram_cleanup_func' itself

  Returns:
  None
  """
  import sys
  all_vars = list(globals().keys())
  exclude_vars = list(sys.modules.keys())
  exclude_vars.extend(['In','Out','_','__','___','__builtin__','__builtins__','__doc__','__loader__','__name__','__package__','__spec__','_dh','_i','_i1','_ih','_ii','_iii','_oh','exit','get_ipython','quit','sys','os','custom_ram_cleanup_func',])
  for var in all_vars:
      if var not in exclude_vars and not var.startswith('_'):
          del globals()[var]
  del sys
  return None

# Example usage:
if __name__ == "__main__":

  import pandas as pd

  # Sample DataFrame
  df:pd.DataFrame = pd.DataFrame(data={
      'PROD_DESC': [
        'drink - mix frsh',
        'drink_mix fresh',
        'wine white sparkling brut',
        'wine wht sparkling brut',
        'coffee grnd decf kcup',
        'coffee ground decaf kcup',
      ],
    }
  )

  # Create an instance of AbbreviationExpansionPipelinePD
  AbbreviationExpansionPipelinePD(
      dataframe_object=df,
      product_desc_column='PROD_DESC',
      ngram=2,
      output_file_name='BI_GRAM_KEYWORDS_MINING',
      cosine_threshold=0.75,
      min_text_match_threshold=85.0,
    ).main()

  custom_ram_cleanup_func()
  del custom_ram_cleanup_func

