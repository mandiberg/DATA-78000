# Data Clustering
CUNY GC: DATA 78000 - Special Topics
"AI and Machine Learning for Artists and Humanists"
Prof Michael Mandiberg

## How this fits in the course
Having built a topic model by turning words into numbers, and finding the relationships between groups of those tokens, we will now turn to grouping sets of numbers and categories. The core idea is the same — finding patterns in the data — but the details and tools are different. Instead of working from the same shared dataset, you will be preparing your own data.

## Assignment Description
In this assignment, you will select/query a data set, and then analyze it using data clustering techniques. Clustering hyper dimensional data is one of the key techniques in machine learning data analysis. In this exercise, we will learn the basics of k-means and k-modes clustering. These are techniques used for grouping data based on common patterns present in their features. K-means is for numerical data, such as measured sizes (ie inches, meters), points in a coordinate system, color as numerical value (HSV, CMYK), speed, mass, etc. K-modes is for categorical data, like size categories (S/M/L/XL), color terms (red, green, blue), gender, occupation, nationality, etc. 

For the first phase of this assignment, we will source one of our datasets from Wikidata. This will be an opportunity to understand what Wikidata is, why it matters, and how to query it. The second data set, you can source from anywhere you like (including Wikidata)

One of the datasets needs to be categorical data, and the other needs to be numerical, because you will complete two different analyses: one using k-means and the other using k-mode. Ideally, they would be related in some way, but this is not a requirement. 

## What You Will Learn (AKA Learning Goals)
By the end of this assignment, you will be able to:

- Query Wikidata using SPARQL 
- Critically assess the strengths and limitations of Wikidata
- Use k-means clustering and k-modes clustering to group your data with appropriate cluster sizes
- Critically reflect on the limits of the tool

## Order of Operations

I have prepared a [Colab notebook with an example k-means code pattern](https://colab.research.google.com/drive/1vP3jNZgd-VmTyajFImPdClnJDH4nd5-u#scrollTo=3ae3bc93)

### Week one:
Introduction to Wikidata, including linked open data, visual query builder, and using a model to generate SPARQL for Wikdata Query Service.

### Week two
Bring two datasets to class, with research questions
Discuss research questions
k-means and k-modes demo
Get it running on your data
Discussion of Elbow Method for cluster size selection
Discussion of preparing data for clustering 

### Week three
Lightning presentations and workshopping

### Week four 
Quick check in

### Week five 
Project due


## Produce an Interpretation/Visualization
- Produce a document/work that answers the research question you asked. It can be a written text that analyzes the themes, presents examples, and interprets them textually. If you are producing a written text, you should include figures/data visualizations. these could analyze your model (coherence, k count, patterns in the clusters) and they also likely will be visualizations that will answer your research question. It can also be a open form and/or creative response that makes use of the same tools to produce meaning. Either way, you need to also include a reflection on your process, describing the strengths and limitations of the tool, and the assumptions baked in to it. 
- You should include/link to your complete Python notebook or script that runs end-to-end

## Good Practices (AKA You Should Always...)
- Keep your code readable and reproducible
- Explain each major step in comments: Do this for me, but also for future you (I promise you that you will forget why you did what you did, and you might need to come back to it in the future)
- Be careful with your assumptions (about what is numerical and what is scalar, how you normalize/weight the data)
- Do not treat any model output as an absolute truth: any output is a probabilistic summary, based on the shape of the data it was trained on.
- When your results are weak or ambiguous, say so explicitly. If you know why this is the case, present that information. 