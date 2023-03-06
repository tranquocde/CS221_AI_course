"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]

class Node:
    def __init__(self, state, pred, action, priority=0):
        self.state = state
        self.pred = pred
        self.action = action
        self.priority = priority
    def __repr__(self):
        return "State: {0}, Action: {1}".format(self.state, self.action)
    
def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    
    
    closed = dict()
    frontier = util.Stack()
    frontier.push(Node(problem.getStartState(), None, None))
    while frontier.isEmpty() is not True:
        node = frontier.pop()
        if problem.isGoalState(node.state) is True:
            actions = list()
            while node.action is not None:
                actions.append(node.action)
                node = node.pred
            actions.reverse()
            return actions
        if not closed.get(node.state):
            closed[node.state] = True
            for succ,action,cost in problem.getSuccessors(node.state):
                frontier.push(Node(succ, node, action))
    return list()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    closed = dict()
    frontier = util.Queue()
    frontier.push(Node(problem.getStartState(), None, None))
    while frontier.isEmpty() is not True:
        node = frontier.pop()
        if problem.isGoalState(node.state) is True:
            actions = list()
            while node.action is not None:
                actions.append(node.action)
                node = node.pred
            actions.reverse()
            return actions
        if not closed.get(node.state):
            closed[node.state] = True
            for succ,action,cost in problem.getSuccessors(node.state):
                frontier.push(Node(succ, node, action))
    return list()

    

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    closed = dict()
    frontier = util.PriorityQueue()
    frontier.push(Node(problem.getStartState(), None, None),0)
    while frontier.isEmpty() is not True:
        node = frontier.pop()
        node:Node
        if problem.isGoalState(node.state) is True:
            actions = list()
            while node.action is not None:
                actions.append(node.action)
                node = node.pred
            actions.reverse()
            return actions
        if not closed.get(node.state):
            closed[node.state] = True
            for succ,action,cost in problem.getSuccessors(node.state):
                frontier.push(Node(succ, node, action,cost + node.priority),priority=cost + node.priority)
    return list()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    closed = dict()
    frontier = util.PriorityQueue()
    frontier.push(Node(problem.getStartState(), None, None),0)
    while frontier.isEmpty() is not True:
        node = frontier.pop()
        node:Node
        if problem.isGoalState(node.state) is True:
            actions = list()
            while node.action is not None:
                actions.append(node.action)
                node = node.pred
            actions.reverse()
            return actions
        if not closed.get(node.state):
            closed[node.state] = True
            for succ,action,cost in problem.getSuccessors(node.state):
                frontier.push(Node(succ, node, action,cost + node.priority),priority=cost + node.priority + heuristic(succ,problem))
    return list()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
