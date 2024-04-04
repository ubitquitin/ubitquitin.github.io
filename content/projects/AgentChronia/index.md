
+++
title = 'AgentChronia'
summary = 'Using Reinforcement Learning to build an agent to traverse a virtual obstacle course with Proximal Policy Optimization (PPO).'
languageCode = 'en-us'
date = 2023-06-06
draft = false
tags = ['notes', 'reflections']
showRecent = true
showTableOfContents = false
+++

This is a WORK IN PROGRESS.
Latest progress video here: https://youtu.be/2S2oA0Il4a4

{{< video src="clip.mp4" type="video/mp4" preload="auto" autoplay="true">}}

The goal is to build an agent to optimally traverse a ~5 minute obstacle course in the online game Runescape. By building an agent that can take the same actions as a player can for moving around a grid based map, the goal is to try and train an agent using proximal policy optimization (PPO) to complete the course in the fastest time. 

I want to see if the agent can come up with new strategies that the human playerbase did not think about, and if it can even break the current record for fastest course completion. 

This project uses the mlagents package in Unity.

[Github Repo](https://github.com/ubitquitin/agentchronia/tree/master)

