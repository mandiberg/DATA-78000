# Topic Modeling Assignment
CUNY GC: DATA 78000 - Special Topics
"AI and Machine Learning for Artists and Humanists"
Prof Michael Mandiberg

## Course Context
As the first step in learning how to use AI and ML tools and to think critically about their use, will be starting with textual analysis of the CUNY Graduate Center’s archive of Dissertations & Theses. This assignment will introduce what it means to "train" a "model" and explore ways to integrate that model into a larger workflow or project. This assignment will call upon your existing Python skills, and your experience preparing data. As you explore this new tool, we will focus on framing questions that the tool and data can answer.  

## Assignment Description
In this assignment, you will build a topic model describing the 19,000 dissertations and theses created at the GC. A topic model is a ML tool for understanding large collections of documents by grouping them into clusters based on the statistical relationships between the words present in these documents. The end result are a set of thematic topics which you can use to group individual documents. By modeling the abstracts, you will be able to see the thematic patterns present, and how they have changed over time.

This assignment shows you a standard text analysis workflow, used in DH, archival research, and exploratory data analysis. Topic Modelling is used when a corpus is too large to actually read, but has meaningful thematic information that you can extract and interpret. We will be using Gensim, which is a pretty standard package. You will be doing real research: this is an unexplored dataset. You will start with the raw CSV, prepare the corpus, train the model, interpret/visualize/explore the results, and reflect on the possibilities and limitations of the tool. Beyond simply training a model, our goal is to understand what questions the tool can answer, what it can't capture, and how choices you make about preprocessing, stop-word removal, and the number of topics affect the outcome, making this as much an art as a science.

Because of the nature of the work at hand, we could choose to work in small groups (3-4 people?). If we do so, each group will coallesce around a research question during the second session. And each group would be responsible for one aspect of the corpus preparation (some of the data needs to be normalized). In theory, we can build groups where each team member brings a different set of skills to the collaboration. We will discuss whether we do or do not want to work in groups during the second session. 

## What You Will Learn (AKA Learning Objectives)
By the end of this assignment, you will be able to:

- Assess and prepare a real-world dataset for analysis
- Understand the role of preprocessing text data for analysis, expecially including the role of stopwords
- Build dictionary for a corpus, and understand the way words become numbers (aka tokens)
- Train, evaluate, and interpret a Gensim Latent Dirichlet Allocation (LDA) topic model
- Apply the model to existing documents, mapping them to specific topics for further analysis
- Ask questions the tool can answer, and identify questions that it cannot answer
- Integrate the model into your own intepretation/visualization/exploration 


## The Dataset and its Ethical Caretaking

We will be using the CUNY Graduate Center’s archive of Dissertations & Theses. This dataset is searchable in parts on CUNY Academic Works, but you need to be logged in. Because the data is not on the openweb, we need to ensure we handle the data responsibly. This means it should not ever be posted to your Github. You can ensure that happens, by immediately adding the file to your gitignore, or by copying my gitignore which already has it. If you don't know what a .gitignore file is, ask a peer or speak up. 


---

## Assignment Tasks
Complete the following steps in a Python notebook or script.

### 1. Data Loading and Exploration
- Read the dataset into a pandas DataFrame
- Identify the text column to analyze
- Inspect the first few rows and check for missing values
- Summarize the corpus size and basic structure
- Articulate a clear Research Question that you will answer through your analysis

### 2. Text Preprocessing
- Normalize text by converting to lowercase
- Remove punctuation and stop words
- Make conscious decisions about what additional words you add to your stop words
- Tokenize the text
- Lemmatize or stem words as appropriate
- Filter excessively short or uninformative terms
- Document your preprocessing decisions

### 3. Build the Topic Model
- Create a dictionary from the processed documents
- Filter extremely rare or overly common terms
- Construct a bag-of-words corpus
- Train an LDA model with a chosen number of topics
- Inspect top terms associated with each topic

### 4. Model Evaluation
- Explore a range of topic counts
- Measure coherence scores for different model settings
- Compare model quality and interpretability -- how meanigful is the coherence score?
- Explain why one topic count may be better than another

### 5. Interpretation/Visualization
- Produce a document or work that analyzes or interprets the themes and analyses
- It can be a written text that analyzes the themes, presents examples, and interprets them textually
- If you are producing a written text, you should include figures/data visualizations. 
 - these could analyze your model (coherence scores, top words per topic, document distribution summary)
 - and they also likely will be visualizations that will answer your research question
- It can also be a open form and/or creative response that makes use of the same tools to produce meaning
- Either way, you need to also include a reflection on your process, describing the strengths and limitations of the tool, and the assumptions baked in to it. 

---

## Deliverables
Submit the following:

1. A complete Python notebook or script that runs end-to-end
2. A brief written analysis describing the dataset, methods, and results
3. At least one visualization/interpretation of the model output
4. A short critical reflection on the limitations of topic modeling

### Potential final report structure (feel free to use, or invent your own)
- Introduction and research question
- Data description
- Data preprocessing decisions
- Modeling approach
- Results and interpretation
- Evaluation and limitations
- Conclusion

---

## Timeline
| Week | Focus |
| --- | --- |
| 1 | Intro to Topic Modeling |
| 2 | Dataset inspection and preprocessing |
| 3 | Troubleshooting process |
| 4 | Visualization/Interpretation and final write-up |

---

## Good Practices (AKA You Should Always...)
- Keep your code readable and reproducible
- Explain each major step in comments: in particularly your decisions re: preprocessing and topic count
- Be careful with your assumptions (about stop words, stemming, and term filtering)
- Do not treat any model output as an absolute truth: any output is an interpretive summary, based on the shape of the data it was trained on.
- When your results are weak or ambiguous, say so explicitly. If you feel confident, you can describe why they might have taken that form. 

