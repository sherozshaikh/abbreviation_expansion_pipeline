## 📝 Abbreviation Expansion Pipeline

Abbreviation Expansion Pipeline is a Python class designed for generating n-gram pairs from product descriptions in a DataFrame and suggesting mappings for text expansion. It utilizes Natural Language Processing (NLP) techniques to preprocess text, compute similarity scores, and suggest replacements based on similarity metrics.

Explore the capabilities of the Abbreviation Expansion Pipeline to streamline text analysis tasks and improve the accuracy of abbreviation expansions in your datasets.

### Overview

Abbreviation Expansion Pipeline provides a comprehensive toolkit for processing textual data, identifying n-grams, computing similarity scores between textual elements, and suggesting mappings for abbreviation expansion. It is designed to enhance text analysis workflows by facilitating efficient preprocessing, similarity assessment, and replacement suggestion tasks.

This pipeline supports various functionalities such as:

- Text preprocessing including character trimming and normalization.
- Generation of n-gram pairs from product descriptions.
- Utilization of pre-trained language models for text embedding.
- Calculation of similarity scores using metrics like fuzzy matching and sequence similarity.
- Identification and suggestion of mappings for text expansion based on similarity scores.

Explore the capabilities of the Abbreviation Expansion Pipeline to streamline text processing tasks and improve the accuracy of text expansion mappings.

---

### ✨ Key Features

- **N-gram Generation**: Extracts n-gram pairs from product descriptions to identify potential abbreviation expansions.
- **Text Preprocessing**: Normalizes text by trimming characters and replacing specified patterns.
- **Similarity Assessment**: Computes similarity scores using advanced NLP techniques such as cosine, fuzzy matching and sequence similarity.
- **Mapping Suggestions**: Suggests mappings for text expansion based on the highest similarity scores between textual elements.

---

### 📦 Requirements

- Python 3.x
- pandas
- transformers
- fuzzywuzzy
- nltk
- numpy
- pandarallel (optional for parallel processing)

---

### 🚀 Usage

#### Using `abbrev_expand.py`

```python

import pandas as pd
from abbreviation_expansion_pipeline import AbbreviationExpansionPipeline

# Example DataFrame with product descriptions
df:pd.DataFrame = pd.read_csv('product_descriptions.csv')

# Create an instance of AbbreviationExpansionPipeline
pipeline = AbbreviationExpansionPipeline(
    dataframe_object=df,
    product_desc_column='description',
    ngram=2,
    output_file_name='Mined_Keyword_Mapping',
    hugging_face_model_name='bert-base-uncased',
    max_text_length=256,
    cosine_threshold=0.73,
    min_text_match_threshold=85.0
  ).main()
```

---

```python
import pandas as pd
from abbreviation_expansion_pipeline import AbbreviationExpansionPipeline

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
```

---

#### Using `abbrev_expand.ipynb`

For a detailed demonstration and interactive usage, refer to the [Abbreviation-Expansion-Pipeline-Notebook](abbrev_expand.ipynb) Jupyter Notebook. It provides examples of how to use the Abbreviation Expansion Pipeline with example.
