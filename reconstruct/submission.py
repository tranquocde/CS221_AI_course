from typing import Callable, List, Set

import shell
import util
import wordsegUtil


############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query: str, unigramCost: Callable[[str], float]):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        #-> cursor
        return 0
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        return state == len(self.query)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        #-> list of (action,newState,cost)
        ans = []
        for i in range(state+1,len(self.query)+1):
            ans.append((i-state,i,self.unigramCost(self.query[state:i])))
        return ans
        # END_YOUR_CODE


def segmentWords(query: str, unigramCost: Callable[[str], float]) -> str:
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    state = 0
    words = []
    for i in ucs.actions:
        word = query[state:state+i]
        state += i
        words.append(word)
    return ' '.join(words)
    # END_YOUR_CODE


############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords: List[str], bigramCost: Callable[[str, str], float],
            possibleFills: Callable[[str], Set[str]]):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        #-> cursor,last_word
        return 0,wordsegUtil.SENTENCE_BEGIN
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        return state[0] == len(self.queryWords)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
        #-> (action,newState,cost)
        index,last_word = state
        temp = self.queryWords[index]
        words = self.possibleFills(temp)
        ans = []
        if len(words) == 0:
            words = set([temp])
        for word in words:
            newState = index+1,word
            ans.append((word,newState,self.bigramCost(last_word,word)))
        return ans

        # END_YOUR_CODE


def insertVowels(queryWords: List[str], bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]]) -> str:
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords,bigramCost,possibleFills))
    actions = ucs.actions
    return ' '.join(actions)
    # END_YOUR_CODE


############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query: str, bigramCost: Callable[[str, str], float],
            possibleFills: Callable[[str], Set[str]]):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        # state = cursor,last_word
        return 0,wordsegUtil.SENTENCE_BEGIN        
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        cursor,last_word = state
        return cursor == len(self.query)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 14 lines of code, but don't worry if you deviate from this)
        cursor,word = state
        ans = []
        for index in range(cursor+1,len(self.query)+1):
            temp = self.query[cursor:index]#new string taking from old string
            words = self.possibleFills(temp)
            for newWord in words:
                cost = self.bigramCost(newWord,word) 
                action = newWord
                newState = (index,newWord)
                ans.append((newWord,newState,cost))
        return ans
    

def segmentAndInsert(query: str, bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]]) -> str:
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query,bigramCost,possibleFills))
    actions = ucs.actions
    return ' '.join(actions)
    # END_YOUR_CODE


############################################################

if __name__ == '__main__':
    shell.main()
