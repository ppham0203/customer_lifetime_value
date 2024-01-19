# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Python code recipe
# This is regular python code. Python coders can continue to do what they have already been doing within Dataiku.
#this is an edit from my code studio
# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Read recipe input dataset
# Dataiku manages data connections to datasources, enforcing security defined on the connections.
# The `dataiku` API lets you read data from the managed connections as pandas dataframes.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
customer_data_clean = dataiku.Dataset("customer_data_clean")
df = customer_data_clean.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Drop duplicates
# This is what this code recipe does.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#Drop Duplicates
df.drop_duplicates(subset="customer_id", keep=False, inplace=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Write recipe output dataset
# Use the `dataiku` API to write the resulting pandas dataframe to the Dataiku-managed data connection.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
customer_data_clean_python = dataiku.Dataset("customer_data_clean_python")
customer_data_clean_python.write_with_schema(df)