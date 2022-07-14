# Sentiment Analysis of the Comments to Optimize the Query Resolution.

<br>

Sentiment analysis is contextual mining of text which identifies and extracts subjective information in the source material and helps a business to understand the social sentiment of their brand, product or service while monitoring online conversations.
We are using NLTK for Natural Language Processing. NLTK stands for Natural Language Toolkit. It is a powerful tool complete with different Python modules and libraries to carry out simple to complex natural language processing (NLP). NLTK provides a VADER model to analyse the sentiment of the comments on the query.

<br>

## What is VADER?

<br>

VADER ( Valence Aware Dictionary for Sentiment Reasoning) is a model used for text sentiment analysis that is sensitive to both polarity (positive/negative) and intensity (strength) of emotion. It is available in the NLTK package and can be applied directly to unlabeled text data.

VADER sentimental analysis relies on a dictionary that maps lexical features to emotion intensities known as sentiment scores. The sentiment score of a text can be obtained by summing up the intensity of each word in the text.

For example- Words like ‘love’, ‘enjoy’, ‘happy’, ‘like’ all convey a positive sentiment. Also VADER is intelligent enough to understand the basic context of these words, such as “did not love” as a negative statement. It also understands the emphasis of capitalization and punctuation, such as “ENJOY”

<br>

## Advantages

<br>

Here are the advantages of using VADER which makes a lot of things easier:
* It does not require any training data.
* It can very well understand the sentiment of a text containing emoticons, slangs, conjunctions, capital words, punctuations and much more.
* It works excellent on social media text.
* VADER can work with multiple domains.

Let’s start analysing the sentiment using VADER

<br>

```python
import os
import json

# For Preprocessing
import re

# For Natural Language Processing
import nltk
nltk.download("stopwords")
nltk.downloader.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
```

<br>

## Loading the Data

<br>

Assume that queries on social media such as Twitter received an incremental comments of different types. That may contain possible solutions, scams, promotions or negative feedback about the company.

<br>

```python
# Reading Json Data
f = open("data.json")
data = json.load(f)
df = data
```

<br>

## Preprocessing the Comments

<br>

Here, we are preprocessing all the text to remove unnecessary numeric data and stop words to derive more accurate results for the sentiment analysis. 

```python
# Converting Tweet to Clean Text by removing Stop Words to improve Accuracy
def tweet_to_words(tweet):
    text = tweet.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    words = [PorterStemmer().stem(w) for w in words]
    return words
```

<br>

```python
# Cleaning the Comments Text by removing Stop Words and Mentions
for d in df["data"]:
    for c in d["comments"]:
        c["cleantext"] = " ".join(tweet_to_words(c["text"])[2:])

```

<br>

## VADER's Sentiment Intensity Analyzer

<br>

VADER’s `SentimentIntensityAnalyzer()` takes in a string and returns a dictionary of scores in each of four categories:
* Negative
* Neutral
* Positive
* Compound

The Compound score is the sum of positive, negative & neutral scores which is then normalized between -1(most extreme negative) and +1 (most extreme positive).

<br>

```python
# Computing the VADER (Valence Aware Dictionary for Sentiment Reasoning) Score
def compute_vader_scores(text):
    sid = SentimentIntensityAnalyzer()
    return {
        "vader_negative": sid.polarity_scores(text)["neg"],
        "vader_neutral": sid.polarity_scores(text)["neu"],
        "vader_positive": sid.polarity_scores(text)["pos"],
        "vader_compound": sid.polarity_scores(text)["compound"]
    }
```

<br>
    
```python
for d in df["data"]:
    for c in d["comments"]:
        score = compute_vader_scores(c["cleantext"])
        c["vader_negative"] = score.get("vader_negative")
        c["vader_positive"] = score.get("vader_positive")
        c["vader_compound"] = score.get("vader_compound")
```

<br>


## Analysing the Comments

<br>

We are analyzing the comments based on the compound score to get the best possible solution. Here, we are putting a constraint on the compound score to be greater than zero to get the positive sentiment comments. Comment with a big compound score may be the possible solution for a given query.

<br>

```python
analysis = []
for d in df["data"]:
    max_compond = 0
    a = { "query": d["text"], "solutions": []}
    for c in d["comments"]:
        if c["vader_compound"] >= max_compond:
            max_compond = c["vader_compound"]
    if max_compond > 0:
        for c in d["comments"]:
            if c["vader_compound"] == max_compond:
                a["solutions"].append(c["text"])
        analysis.append(a)
```

## Result

<br>

We can see the list of possible solutions for a given query. We can improve accuracy by adding advanced processing and intent checks to get the best results. Also, updating the existing knowledge base using this will further improve the accuracy, consistency and performance of the current system to enhance the query resolution of a customer.

<br>

```python
for a in analysis:
    print("❓ {query}".format(query = a["query"]))
    print()
    for s in a["solutions"]:
        print("🟢 {solution}".format(solution = s))
    print()
    print()
```

### Output

<br>

```
❓ @lg My LG refrigerator has sheets of ice build up on the bottom every week. Any idea how to fix it?

🟢 @lg @bill There should be a drain that allows the moisture that is generated in the freezer to drain into a pan under the refrigerator where the heat from the compressor will evaporate it. There is a RED button on the back that you can press to fix this. Generally that drain gets clogging and allows the water to accumulate in the freezer and...it freezes.
```
