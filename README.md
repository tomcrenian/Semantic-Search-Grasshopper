# Semantic-Search-Grasshopper
A tool which uses OpenAI Assistants API to search databases of reclaimed components for matches in Grasshopper, Rhino3D

You will have to install the following prerequisites either in the Grasshopper Python Environment or on Python Desktop:
- OpenAI
- Pandas
- Scikit-Learn
- Numpy

Use the main script 'GPTFinal.gh' to interface with the OpenAI API and perform searching OR repurpose the python module to use the assistants API for another function

Use the provided embeddings database 'embeddings.csv' for cosine similarity search 
OR
Use the 'generateembeddings.py' script to either generate your own embeddings from Tixier's list of words.

An example database is provided to search from 'exampledatabase.json' which you can upload to your OpenAI API Assistant

Larger files provided below:
https://photonserver.eichi.net:5003/d/s/119o6cgLlH48icaCV3PV1iyEuq6vD8Dg/72VY_wYM3DzM3SG9gpIxVeSoLkQVJCxW-Kr5AuRV-3Qs

Testing and data validation files are provided as references
