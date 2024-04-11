+++
title = 'dRAG 🐲'
summary = 'A streamlit deployed chatbot that knows about your specific database! Dynamic corpus creation and Q&A with text2sql Huggingface LLMs+RAG'
languageCode = 'en-us'
date = 2024-04-04
draft = false
tags = ['notes', 'reflections']
showRecent = true
showTableOfContents = false
+++

Demo:
https://youtu.be/EdrIJUNsmQs

dRAG dynamically generates a corpus about your Snowflake/Databrickcs schema and uses retrieval augmented generation (RAG) to answer specific questions about your data.

Additionally, dRAG can convert your question into a SQL statement and run the SQL statement in your data warehouse to find the answer for you.

[Github Repo](https://github.com/ubitquitin/dbrag/tree/main) | [Try it out](https://datarag.streamlit.app/)

<img width="500" height="500" src="featured.PNG">