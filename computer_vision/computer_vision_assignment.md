# Computer Vision
CUNY GC: DATA 78000 - Special Topics
"AI and Machine Learning for Artists and Humanists"
Prof Michael Mandiberg

## How this fits in the course
We have now learned some ways to analyze text, numbers, and categorical data, we will now explore how we can analyze images. This is a low stakes assignment -- your goal is to complete it, and not worry about "quality" or "evaluation." Just get the code to work, and spends some time thinking about how you can incorporate it into a workflow.

## Assignment Description
For this module, we will use OpenCV as our base framework to complete four related exercises. These will cover some core computer vision skills and methods: 
1. Face Detection and Position using Mediapipe (is there a face, and where is it looking?)
2. Face Recognition (are two faces the same person?)
3. Object Detection with YOLOe (is there a Car, Horse, or Sportsball in this image?)
4. Building a custom object detection model with YOLO26 (is that ball a basketball or a soccerball?)
These cover some of the basic principles of computer vision, giving you a foundation for further exploration. For each exercise, I will give you a canonical code pattern, and ask you to take that canonical pattern and move it one step towards usefulness, by... building a pipeline that allows you to process a folder of images ands save the results, testing the model with a range of images to determine where the model breaks down, incorporating the data into another pipeline (either one we have used in class, or one you use in your own work), etc. 

Kaggle is a great place to find training datasets, which you can also use as test data for a pipeline, but it is also awesome if you have your own data you are genuinely interested in. Wikimedia Commons might be a useful place to collect some images from; I will share a simple scraping script I have previously used to scrape the Commons. 

## What You Will Learn (AKA Learning Goals)
By the end of this assignment, you will be able to:

- See faces and objects with your computer 
- Train an object detection model with a custom training set
- Critically reflect on the limits of these tools

## Order of Operations

I have prepared:
- [a Colab notebook with an example mediapipe code pattern](https://colab.research.google.com/drive/1rJjngmv-TjW9y8-PwfctGQe4YU1c9TpZ?usp=sharing)
- [a Colab notebook for the DeepFace face recognition package](https://colab.research.google.com/drive/1rRLg2GeZftc_hw8plpYLDcLbi6lh06Sc?usp=sharing)
- [a Colab notebook for the DeepFace where you can upload your own image](https://colab.research.google.com/drive/1HuQWUNOFkGq-qPI6aXmO8NowlpXxiuWX?usp=sharing)
- YOLOe Object Detection
- YOLO26 Training Run

### Week one:
Face Detection and Positioning 

### Week two
Face Recognition

### Week three
Object Detection

### Week four 
Custom Models for YOLO

### Week five 
All four exercises due. This assignment is low stakes. 


## What to turn in
Turn in four notebooks or .py scripts for each of your customizations of the canoncial examples. No writing necessary, though you are welcome to reflect on what you have learned. 

## Good Practices (AKA You Should Always...)
- Keep your code readable and reproducible
- Explain each major step in comments: Do this for me, but also for future you (I promise you that you will forget why you did what you did, and you might need to come back to it in the future)
- Be careful with your assumptions (about what is numerical and what is scalar, how you normalize/weight the data)
- Do not treat any model output as an absolute truth: any output is a probabilistic summary, based on the shape of the data it was trained on.
- When your results are weak or ambiguous, say so explicitly. If you know why this is the case, present that information. 

## If you want to dive deep, here are some references
- [High level overview of computer vision](https://www.ibm.com/think/topics/computer-vision)
- [Chapter from Foundations of Computer Vision (MIT)](https://visionbook.mit.edu/object_recognition_v3.html)
- [The original Harr Cascade Object Detection paper](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)
- [An explanation of a Convolutional Neural Network (CNN)](https://learnopencv.com/understanding-convolutional-neural-networks-cnn/)
- [Cosine Similarity explainer, and relationshop to Cartesian Distance](https://towardsdatascience.com/demystifying-cosine-similarity/)
- [A different Cosine Similarity explainer](https://www.ibm.com/think/topics/cosine-similarity)
- [A good visual explanation of what a "neural network" is](https://www.3blue1brown.com/lessons/neural-networks/)
