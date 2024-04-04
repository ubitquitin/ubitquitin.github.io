+++
title = 'Recommendations Model'
summary = 'Collaborative Filtering vs LLM based recommendations.'
languageCode = 'en-us'
date = 2023-12-21
draft = false
tags = ['notes', 'reflections']
showRecent = true
showTableOfContents = false
+++

Anime recommendation service that returns top n most similar shows to the user's queried show. Neural Collaborative Filtering was implemented and embeddings deployed onto a RESTful endpoint. Serverless framework was used to spin up API Gateway + Lambda stack using infrastructure as code development. Additional endpoint deployed that utilizes LLM embeddings to and compare a user based recommendation approach with a text similarity one.

[Try it out]({{< ref "animerecco/animerecco.md" >}}) | [Github Repo](https://github.com/ubitquitin/mal_reccos)

<img width="500" height="500" src="featured.png">