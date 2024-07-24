# -*- coding: utf-8 -*-

# # Checks for required Python packages and installs them if not already installed.

# import subprocess
# import importlib

# req_packages:list = ['ast','collections','concurrent','gc','nltk','numpy','os','pandarallel','pandas','re','shutil','string','time','transformers','typing','warnings',]

# for package_name in req_packages:
#   try:
#     importlib.import_module(package_name)
#   except:
#     try:
#       # !pip install --quiet {package_name}
#       subprocess.check_call(['pip', 'install', '--quiet', package_name])
#     except Exception as e:
#       print(f"Required package {package_name} was not installed!: {str(e)}")
# del importlib
# print("All required packages are installed.")


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
import concurrent.futures
import re
import ast
import string
import pandas as pd
import numpy as np
import nltk
nltk.download(['punkt', 'stopwords'])
from nltk import word_tokenize
from nltk.util import ngrams
from transformers import AutoTokenizer,AutoModel
import os
cpu_count = os.cpu_count()
cpu_count:int = cpu_count if cpu_count else 1
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True,nb_workers=cpu_count)

import shutil

subprocess.run(['git', 'clone', '--quiet', 'https://github.com/sherozshaikh/text_to_vector_embedding_pipeline.git'])
shutil.move('./text_to_vector_embedding_pipeline/embedding.py', './')
shutil.rmtree('text_to_vector_embedding_pipeline')

subprocess.run(['git', 'clone', '--quiet', 'https://github.com/sherozshaikh/pdf_chunk_alignment.git'])
shutil.move('./pdf_chunk_alignment/doc_mapper.py', './')
shutil.rmtree('pdf_chunk_alignment')

subprocess.run(['git', 'clone', '--quiet', 'https://github.com/sherozshaikh/text_similarity_metrics.git'])
shutil.move('./text_similarity_metrics/text_scoring.py', './')
shutil.rmtree('text_similarity_metrics')

from embedding import TextEmbedding
from doc_mapper import DocMapper
from text_scoring import TextScoring
print("All required packages are imported.")

class AbbreviationExpansionPipeline():
  """
  A class for generating n-gram pairs from product descriptions in a DataFrame and suggesting mappings for text expansion.
  """
  def __init__(self,dataframe_object:pd.DataFrame,product_desc_column:str,ngram:int=2,output_file_name:str='Mined_Keyword_Mapping',hugging_face_model_name:str='google-bert/bert-base-uncased',max_text_length:int=256,cosine_threshold:float=0.73,min_text_match_threshold:float=85.0,):
    """
    Initialize the AbbreviationExpansionPipeline class.

    Args:
    - dataframe (pd.DataFrame): Input pandas DataFrame object containing data.
    - product_desc_column (str): Name of the column in 'dataframe' containing product descriptions.
    - ngram (int, optional), default = 2: Number of words in each n-gram.
    - output_file_name (str, optional), default = 'Mined_Keyword_Mapping': Name of the output file where results will be saved.
    - hugging_face_model_name (str, optional), default = 'google-bert/bert-base-uncased': Name of the Hugging Face transformer model.
    - max_text_length (int, optional), default = 256: Maximum length of text processed by the model.
    - cosine_threshold (float, optional), default = 0.73: Threshold value for Cosine Similarity.
    - min_text_match_threshold (float, optional), default = 85.0: Minimum Threshold value for Text Similarity.
    """
    self.working_df:pd.DataFrame = dataframe_object
    self.product_desc_column:str = product_desc_column
    self.ngram_pair:int = ngram
    self.filter_min_count = self.ngram_pair - 1
    self.output_file_name:str = output_file_name
    self.hf_model_name:str = hugging_face_model_name
    self.hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
    self.hf_model = AutoModel.from_pretrained(self.hf_model_name)
    self.text_max_length:int = max_text_length
    self.cosine_threshold = cosine_threshold
    self.min_text_match_threshold = min_text_match_threshold
    self.text_scoring = TextScoring(
        dataframe_object=pd.DataFrame(),
        output_folder='HF_MODEL',
        col_name_1='doc1_elements',
        col_name_2='doc2_elements',
        metrics_list=['fuzz_partial_ratio'],
      )
    self.doc1_embeddings:list = []

  def __repr__(self):
    """
    Returns a string representation of the class instance.
    """
    return f"AbbreviationExpansionPipeline()"

  def __str__(self):
    """
    Returns a description of the class.
    """
    return "A class for generating n-gram pairs from product descriptions in a DataFrame and suggesting mappings for text expansion."

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
    summary_df:pd.DataFrame = pd.DataFrame(data={'Context': list(ngram_counter.keys()),'ContextCount': list(ngram_counter.values()),'CorpusCount': len(df),})
    summary_df['PerAvailalibility']:pd.Series = summary_df['ContextCount'].div(summary_df['CorpusCount'])

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
        sc1:float = self.text_scoring.get_fuzz_partial_ratio(sent_1=i1,sent_2=i2)
        sc2:float = self.text_scoring.get_jaro_winkler_similarity(sent_1=i1,sent_2=i2)
        current_score:float = sc1 if sc1 >= sc2 else sc2
        # current_score:float = min(sc1,sc2)
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

  def get_embeddings_chunk(self,doc1_chunk:list)->np.ndarray:
    """
    Compute embeddings for a chunk of text elements using a pretrained model.

    Args:
    - doc1_chunk (list): A chunk of text elements to compute embeddings for.

    Returns:
    - np.ndarray: Array of embeddings for the input text chunk.
    """
    return TextEmbedding().get_pre_trained_models_embedding(texts=doc1_chunk,model_name=self.hf_model,model_tokenizer=self.hf_tokenizer,custom_max_length=self.text_max_length)

  def get_chunked_process_list(self,elements:list,chunk_size:int):
    """
    Generator function to yield chunks of elements from a given list.

    Args:
    - elements (list): List of elements to be chunked.
    - chunk_size (int): Size of each chunk.

    Yields:
    - list: Chunks of elements as lists.
    """
    for i in range(0,len(elements),chunk_size):
      yield elements[i:i + chunk_size]

  def process_chunks_in_parallel(self,elements:list,chunk_size:int):
    """
    Process chunks of elements in parallel using a ProcessPoolExecutor.

    Args:
    - elements (list): List of elements to process in chunks.
    - chunk_size (int): Size of each chunk.

    Actions:
    - Modifies self.doc1_embeddings by extending it with embeddings from each chunk.
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
      futures:list = []

      for chunk in self.get_chunked_process_list(elements,chunk_size):
        future = executor.submit(self.get_embeddings_chunk,chunk)
        futures.append(future)

      for future in concurrent.futures.as_completed(futures):
        embeddings_chunk = future.result()
        self.doc1_embeddings.extend(embeddings_chunk)

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
    doc1_elements:list = ngrams_df_status['Context'].drop_duplicates().dropna().tolist()
    del ngrams_df_status
    gc.collect()

    # Processing - Embedding ======
    # doc1_embeddings:np.ndarray = TextEmbedding().get_pre_trained_models_embedding(texts=doc1_elements,model_name=self.hf_model,model_tokenizer=self.hf_tokenizer,custom_max_length=self.text_max_length)

    self.process_chunks_in_parallel(elements=doc1_elements,chunk_size=2_000)
    self.doc1_embeddings:np.ndarray = np.array(self.doc1_embeddings)

    # generate nearest mapping ======
    DocMapper(
      doc1_elements_list=doc1_elements,
      doc2_elements_list=doc1_elements,
      doc1_elements_embedding=self.doc1_embeddings,
      doc2_elements_embedding=self.doc1_embeddings,
      threshold_=self.cosine_threshold,
      output_folder='HF_MODEL',
      same_flag=True,
      create_folder=False,
      create_zip=False,
      ).main()

    del doc1_elements,self.doc1_embeddings
    gc.collect()

    # Processing - Filtering Embedding Output ======
    working_result_df:pd.DataFrame = pd.read_csv(filepath_or_buffer='HF_MODEL_Mapping.csv',dtype='str',encoding='latin-1')
    print('\u2501'*35)
    print('\u2503','{:^31}'.format(f'Nearest Matches Found: {working_result_df.shape[0]}'),'\u2503')
    print('\u2501'*35)

    # get Common Elements between the two documents ======
    working_result_df['CommonElementsCount']:pd.Series = working_result_df[['doc1_elements','doc2_elements']].parallel_apply(lambda x: len(set(str(x[0]).split(' ')).intersection(set(str(x[1]).split(' ')))),axis=1)
    working_result_df:pd.DataFrame = working_result_df[working_result_df['CommonElementsCount']>self.filter_min_count]
    print('\u2501'*35)
    print('\u2503','{:^31}'.format(f'Potential Matches Found: {working_result_df.shape[0]}'),'\u2503')
    print('\u2501'*35)

    # Processing - Similarity Match ======
    TextScoring(
        dataframe_object=working_result_df,
        output_folder='HF_MODEL',
        col_name_1='doc1_elements',
        col_name_2='doc2_elements',
        metrics_list=[
            'difflib_sequencematcher',
            'fuzz_token_set_ratio',
            'jaro_similarity',
            'fuzz_partial_ratio',
            ],
      ).main()

    del working_result_df
    gc.collect()

    # Processing - Filtering Similarity Match Output ======
    working_result_score_df:pd.DataFrame = pd.read_csv(filepath_or_buffer='HF_MODEL_Similarity_Scores.csv',dtype='str',encoding='latin-1')

    os.remove('HF_MODEL_Mapping.csv')
    os.remove('HF_MODEL_Similarity_Scores.csv')

    for c_name in working_result_score_df.columns:
      if c_name not in ['doc1_elements','doc2_elements',]:
        working_result_score_df[c_name]:pd.Series = working_result_score_df[c_name].astype(float)
      else:
        continue
    del c_name

    # Processing - Match Filtering ======
    working_result_score_df:pd.DataFrame = working_result_score_df[working_result_score_df['high_score_difflib_sequencematcher']>29.99]
    print('\u2501'*35)
    print('\u2503','{:^31}'.format(f'After Filter Operation 1: {working_result_score_df.shape[0]}'),'\u2503')
    print('\u2501'*35)

    working_result_score_df:pd.DataFrame = working_result_score_df[working_result_score_df['high_score_fuzz_token_set_ratio']>29.99]
    print('\u2501'*35)
    print('\u2503','{:^31}'.format(f'After Filter Operation 2: {working_result_score_df.shape[0]}'),'\u2503')
    print('\u2501'*35)

    working_result_score_df:pd.DataFrame = working_result_score_df[working_result_score_df['high_score_jaro_similarity']>29.99]
    print('\u2501'*35)
    print('\u2503','{:^31}'.format(f'After Filter Operation 3: {working_result_score_df.shape[0]}'),'\u2503')
    print('\u2501'*35)

    working_result_score_df:pd.DataFrame = working_result_score_df[working_result_score_df['high_score_fuzz_partial_ratio']>29.99]
    print('\u2501'*35)
    print('\u2503','{:^31}'.format(f'After Filter Operation 4: {working_result_score_df.shape[0]}'),'\u2503')
    print('\u2501'*35)

    working_result_score_df:pd.DataFrame = working_result_score_df.sort_values(by=['high_score_difflib_sequencematcher','high_score_fuzz_token_set_ratio','high_score_jaro_similarity','high_score_fuzz_partial_ratio',],ascending=[False,False,False,False])

    # Processing - Text Replacement ======

    # single core processing
    # working_result_score_df['SUGGESTED_MAPPING']:pd.Series = working_result_score_df[['doc1_elements','doc2_elements']].apply(lambda x: self.get_replacement(item_1=x[0],item_2=x[1]),axis=1)
    # working_result_score_df['replacement_flag']:pd.Series = working_result_score_df['SUGGESTED_MAPPING'].apply(lambda x: True if x else False)

    # using pandarallel for multiprocessing
    working_result_score_df['SUGGESTED_MAPPING']:pd.Series = working_result_score_df[['doc1_elements','doc2_elements']].parallel_apply(lambda x: self.get_replacement(item_1=x[0],item_2=x[1]),axis=1)
    working_result_score_df['replacement_flag']:pd.Series = working_result_score_df['SUGGESTED_MAPPING'].parallel_apply(lambda x: True if x else False)

    working_result_score_df:pd.DataFrame = working_result_score_df[working_result_score_df['replacement_flag']==True]
    working_result_score_df:pd.DataFrame = working_result_score_df[['doc1_elements','doc2_elements','Score','SUGGESTED_MAPPING']]
    working_result_score_df.columns = ['PROD_DESC1','PROD_DESC2','SIMILARITY_SCORE','SUGGESTED_MAPPING']
    working_result_score_df['SIMILARITY_SCORE']:pd.Series = working_result_score_df['SIMILARITY_SCORE'].astype(float).round(0)
    working_result_score_df['SIMILARITY_SCORE']:pd.Series = working_result_score_df['SIMILARITY_SCORE'].astype(int)

    suggested_map_df:pd.DataFrame = working_result_df['SUGGESTED_MAPPING'].value_counts().reset_index(name='REPLACEMENT_COUNT')
    suggested_map_df['map']:pd.Series = suggested_map_df['SUGGESTED_MAPPING'].apply(lambda x: [str(k)+'!'+str(v) for k,v in ast.literal_eval(node_or_string=str(x)).items()])
    suggested_map_df:pd.DataFrame = suggested_map_df.explode(column='map')
    suggested_map_df[['TO_REPLACE','REPALCE_WITH']]:pd.DataFrame = suggested_map_df['map'].str.split(pat='!',expand=True,regex=False)
    suggested_map_df:pd.DataFrame = suggested_map_df[['SUGGESTED_MAPPING','REPLACEMENT_COUNT','TO_REPLACE','REPALCE_WITH']]

    working_result_df['LEN1']:pd.Series = working_result_df['PROD_DESC1'].str.len()
    working_result_df['LEN2']:pd.Series = working_result_df['PROD_DESC2'].str.len()
    working_result_df:pd.DataFrame = working_result_df.sort_values(by=['SUGGESTED_MAPPING','LEN2','LEN1'],ascending=[True,False,False]).groupby(by=['SUGGESTED_MAPPING']).head(1)[['PROD_DESC1','PROD_DESC2','SUGGESTED_MAPPING']]

    working_result_df:pd.DataFrame = pd.merge(left=working_result_df,right=suggested_map_df,on=['SUGGESTED_MAPPING'],how='inner')
    del suggested_map_df
    gc.collect()

    working_result_df:pd.DataFrame = working_result_df[['PROD_DESC1','PROD_DESC2','REPLACEMENT_COUNT','TO_REPLACE','REPALCE_WITH']]
    working_result_df:pd.DataFrame = working_result_df.sort_values(by=['REPLACEMENT_COUNT','REPALCE_WITH','TO_REPLACE'],ascending=[False,True,True])

    # single core processing
    # working_result_df['SIMILARITY_SCORE_1']:pd.Series = working_result_df[['TO_REPLACE','REPALCE_WITH']].apply(lambda x: self.text_scoring.get_fuzz_partial_ratio(sent_1=x[0],sent_2=x[1]),axis=1)
    # working_result_df['SIMILARITY_SCORE_2']:pd.Series = working_result_df[['TO_REPLACE','REPALCE_WITH']].apply(lambda x: self.text_scoring.get_jaro_winkler_similarity(sent_1=x[0],sent_2=x[1]),axis=1)

    # using pandarallel for multiprocessing
    working_result_df['SIMILARITY_SCORE_1']:pd.Series = working_result_df[['TO_REPLACE','REPALCE_WITH']].parallel_apply(lambda x: self.text_scoring.get_fuzz_partial_ratio(sent_1=x[0],sent_2=x[1]),axis=1)
    working_result_df['SIMILARITY_SCORE_2']:pd.Series = working_result_df[['TO_REPLACE','REPALCE_WITH']].parallel_apply(lambda x: self.text_scoring.get_jaro_winkler_similarity(sent_1=x[0],sent_2=x[1]),axis=1)

    working_result_df['SIMILARITY_SCORE_1']:pd.Series = working_result_df['SIMILARITY_SCORE_1'].round(0)
    working_result_df['SIMILARITY_SCORE_2']:pd.Series = working_result_df['SIMILARITY_SCORE_2'].round(0)
    working_result_df['SIMILARITY_SCORE_1']:pd.Series = working_result_df['SIMILARITY_SCORE_1'].astype(int)
    working_result_df['SIMILARITY_SCORE_2']:pd.Series = working_result_df['SIMILARITY_SCORE_2'].astype(int)

    # writing file with abbreviation expansion suggestion with examples
    working_result_df.to_csv(path_or_buf=self.output_file_name+'_Examples.csv',index=False,encoding='latin-1',mode='w',header=True,)
    print('\u2501'*43)
    print('\u2503','{:^31}'.format(f'Abbreviation Expansion with Examples: {working_result_df.shape[0]}'),'\u2503')
    print('\u2501'*43)

    del working_result_df
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

  # Create an instance of AbbreviationExpansionPipeline
  AbbreviationExpansionPipeline(
      dataframe_object=df,
      product_desc_column='PROD_DESC',
      ngram=2,
      output_file_name='BI_GRAM_KEYWORDS_MINING',
      hugging_face_model_name='google-bert/bert-base-uncased',
      max_text_length=256,
      cosine_threshold=0.75,
      min_text_match_threshold=85.0,
    ).main()

  custom_ram_cleanup_func()
  del custom_ram_cleanup_func

