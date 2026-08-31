# Topic Modeling the Graduate Center's Intellectual History
CUNY GC: DATA 78000 - Special Topics
"AI and Machine Learning for Artists and Humanists"
Prof Michael Mandiberg

## How this fits in the course
As the first step in learning how to use AI and ML tools and to think critically about their use, we will be textual analyzing the CUNY Graduate Center’s archive of Dissertations & Theses. This assignment will introduce what it means to "train" a "model" and explore ways to integrate that model into a larger workflow or project. This assignment will call upon your existing Python skills and your experience preparing data. As you explore this new tool, we will focus on framing questions that the tool and data can answer.  

## Assignment Description
In this assignment, you will build a topic model describing the 19,000 dissertations and theses created at the GC. A topic model is a ML tool for understanding large collections of documents by grouping them into clusters based on the statistical relationships between the words present in these documents. The end result are a set of thematic topics which you can use to group individual documents. By modeling the abstracts, you will be able to see the thematic patterns present, and how they have changed over time.

This assignment shows you a standard text analysis workflow, used in DH, archival research, and exploratory data analysis. Topic Modelling is used when a corpus is too large to actually read, but has meaningful thematic information that you can extract and interpret. We will be using Gensim, which is a pretty standard package. You will be doing real research: this is an unexplored dataset. You will start with the raw CSV, prepare the corpus, train the model, interpret/visualize/explore the results, and reflect on the possibilities and limitations of the tool. Beyond simply training a model, our goal is to understand what questions the tool can answer, what it can't capture, and how choices you make about preprocessing, stop-word removal, and the number of topics affect the outcome--making this as much an art as a science.

Because of the nature of the work at hand, we could choose to work in small groups (3-4 people?). If we do so, each group will coallesce around a research question during the second session. And each group would be responsible for one aspect of the corpus preparation (some of the data needs to be normalized). Different research questions will require different preparations of the data. In theory, we can build groups where each team member brings a different set of skills to the collaboration. We will discuss whether we do or do not want to work in groups during the second session. 

## What You Will Learn (AKA Learning Goals)
By the end of this assignment, you will be able to:

- Assess and prepare a real-world dataset for analysis
- Understand the role of preprocessing text data for analysis, expecially the role of stopwords
- Build a dictionary for a corpus, and understand the way words become numbers (aka tokens)
- Train, evaluate, and interpret a topic model
- Apply the model to existing documents, mapping them to specific topics for further analysis
- Ask questions the tool can answer, and identify questions that it cannot answer
- Integrate the model into your own intepretation/visualization/exploration 


## The Dataset and its Ethical Caretaking
We will be using the CUNY Graduate Center’s archive of Dissertations & Theses. This dataset is searchable in parts on CUNY Academic Works, but you need to be logged in. Because the data is not on the openweb, we need to ensure we handle the data responsibly. This means it should not ever be posted to your Github. You can ensure that happens, by immediately adding the file to your gitignore, or by copying my gitignore which already has it. If you don't know what a .gitignore file is, ask a peer or speak up. 


---

## Order of Operations

I have prepared a [Colab notebook with an example Gensim code pattern](https://colab.research.google.com/drive/1UznWi2NEYZtGSdgQPFqdL77Q5rQQaIXt?usp=sharing)

### 1. Load and Explore the dataset
- Read the dataset and understand the shape of the data
- Articulate a clear Research Question that you will answer through your analysis

### 2. Preprocess the text
- Do all the regular conversion (stemming, lemmatize, filtering), and see what that produces
- Make conscious decisions about what additional words you add to your stop words
- Document your preprocessing decisions

### 3. Build and evaluate Topic Model
- Explore a range of topic counts
- Measure coherence scores for different model settings
- Compare model quality and interpretability -- how meanigful is the coherence score?
- Do you need to ditch the coherence model and go on your human interpretation?
- Explain why one topic count may be better than another

### 4. Produce an Interpretation/Visualization
- Produce a document/work that answers the research question you asked. 
- It can be a written text that analyzes the themes, presents examples, and interprets them textually
 - If you are producing a written text, you should include figures/data visualizations. these could analyze your model (coherence scores, top words per topic, document distribution summary) and they also likely will be visualizations that will answer your research question
- It can also be a open form and/or creative response that makes use of the same tools to produce meaning
- Either way, you need to also include a reflection on your process, describing the strengths and limitations of the tool, and the assumptions baked in to it. 
- You should include your complete Python notebook or script that runs end-to-end

## Good Practices (AKA You Should Always...)
- Keep your code readable and reproducible
- Explain each major step in comments: in particularly your decisions re: preprocessing and topic count. Do this for me, but also for future you (I promise you that you will forget why you did what you did, and you might need to come back to it in the future)
- Be careful with your assumptions (about stop words, stemming, and term filtering)
- Do not treat any model output as an absolute truth: any output is a probablistic summary, based on the shape of the data it was trained on.
- When your results are weak or ambiguous, say so explicitly. If you know why this is the case, present that information. 