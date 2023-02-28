#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar
import collections
from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    ans = dict()
    for word in x.split():
        if word not in ans:
            ans[word] = 1
        else: ans[word]+= 1
    return ans
    # END_YOUR_CODE


############################################################
# Problem 3b: stochastic gradient descent

T = TypeVar('T')


def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    '''
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    def prediction(x):
        theta_x = featureExtractor(x)
        if dotProduct(weights,theta_x) <0:return -1
        else: return 1
    for i in range(numEpochs):
        for (x,y) in trainExamples:
            theta_x = extractWordFeatures(x)
            margin = dotProduct(weights,theta_x)*y 
            if margin < 1:
                increment(weights,eta*y,theta_x)
        print(f'Epochs {i} Train set: {evaluatePredictor(trainExamples,prediction)} , Test set:\
              {evaluatePredictor(validationExamples,prediction)}')
    # END_YOUR_CODE
    return weights


############################################################
# Problem 3c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        phi={}
        for item in random.sample(list(weights),random.randint(1,len(weights))):
            phi[item]=random.randint(1,100)
        y=1 if dotProduct(weights,phi)>1 else -1
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        new_x = x.replace(' ','')
        ans = dict()
        for i in range(len(new_x)-n+1):
            sub_str = new_x[i:i+n]
            if sub_str not in ans:
                ans[sub_str] = 1
            else: ans[sub_str] += 1
        return ans


    return extract


############################################################
# Problem 3e:


def testValuesOfN(n: int):
    '''
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    '''
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples,
                             validationExamples,
                             featureExtractor,
                             numEpochs=20,
                             eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights,
                        'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(
        validationExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" %
           (trainError, validationError)))


############################################################
# Problem 5: k-means
############################################################

def kmeans(examples: List[Dict[str, float]], K: int,
           maxEpochs: int) -> Tuple[List, List, float]:
    '''
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 28 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    # we have k clusters
    centroids=[sample.copy() for sample in random.sample(examples,K)]
    bestmatch=[random.randint(0,K-1) for item in examples]
    distances=[0 for item in examples]
    pastmatches=None
    examples_squared=[]

    #compute square of each data point
    for item in examples:
        tempdict=dict()
        for k,v in item.items():
            tempdict[k]=v*v
        examples_squared.append(tempdict)

    for run_range in range(maxEpochs):

        #compute square of each centroid
        centroids_squared=[]
        for item in centroids:
            tempdict = dict()
            for k, v in item.items():
                tempdict[k] = v * v
            centroids_squared.append(tempdict)
    
        for index,item in enumerate(examples):
            min_distance=float('inf')
            for i,cluster in enumerate(centroids):

                #(a-b)^2 = a^2+b^2 - 2ab
                #use this formula to compute the real distance
                distance=sum(examples_squared[index].values())+sum(centroids_squared[i].values())
                for k in set(item.keys() & cluster.keys()):
                    distance+=-2*item[k]*cluster[k]
                
                if distance<min_distance:
                    min_distance=distance
                    bestmatch[index]=i
                    distances[index]=min_distance
        if pastmatches==bestmatch:
            break
        else:
            clustercounts=[0 for cluster in centroids]
            for i,cluster in enumerate(centroids):
                for k in cluster:
                    cluster[k]=0.0
            for index,item in enumerate(examples):
                clustercounts[bestmatch[index]]+=1
                cluster=centroids[bestmatch[index]]
                for k,v in item.items():
                    if k in cluster:
                        cluster[k]+=v
                    else:
                        cluster[k]=0.0+v
            for i, cluster in enumerate(centroids):
                for k in cluster:
                    cluster[k]/=clustercounts[i]
            pastmatches=bestmatch[:]
    return centroids,bestmatch,sum(distances)
    # END_YOUR_CODE
