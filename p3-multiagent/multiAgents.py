# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Distance minimale à la nourriture
        foodList = newFood.asList()
        if foodList:
            minFoodDist = min(manhattanDistance(newPos, foodPos) for foodPos in foodList)
        else:
            minFoodDist = 0  # Pas de nourriture restante

        # Distances aux fantômes
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]

        # Analyse fantômes proches
        minGhostDist = min(ghostDistances) if ghostDistances else float('inf')

        # On commence avec le score de base
        score = successorGameState.getScore()

        # Si un fantôme est trop proche et pas effrayé => pénalité sévère
        for i, dist in enumerate(ghostDistances):
            if newScaredTimes[i] == 0 and dist <= 1:
                return -float('inf')  # Mieux vaut éviter cette action à tout prix

        # Bonus si fantômes effrayés et proches (on peut les manger)
        for i, dist in enumerate(ghostDistances):
            if newScaredTimes[i] > 0:
                score += 200 / (dist + 1)  # Plus proche, plus le bonus est grand

        # Pénaliser la distance à la nourriture : plus proche c'est mieux
        if minFoodDist > 0:
            score += 10.0 / minFoodDist

        # Pénaliser l’action STOP pour éviter de rester sur place
        if action == Directions.STOP:
            score -= 5

        return score
        
def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(agentIndex, depth, state):
            # Cas terminal : fin de partie ou profondeur atteinte
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            # Agent Max (Pacman)
            if agentIndex == 0:
                bestValue = float('-inf')
                bestAction = None
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    val = minimax(1, depth, successor)  # passe au fantôme 1, même profondeur
                    if val > bestValue:
                        bestValue = val
                        bestAction = action
                # Si on est à la racine, retourne l'action, sinon retourne la valeur
                if depth == 0:
                    return bestAction
                else:
                    return bestValue

            # Agents Min (Fantômes)
            else:
                nextAgent = (agentIndex + 1) % numAgents
                nextDepth = depth + 1 if nextAgent == 0 else depth

                bestValue = float('inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    val = minimax(nextAgent, nextDepth, successor)
                    if val < bestValue:
                        bestValue = val
                return bestValue

        # Lance la recherche à partir du premier agent (Pacman) et profondeur 0
        return minimax(0, 0, gameState)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(agentIndex, depth, state, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            if agentIndex == 0:  # Pacman (Max)
                value = float('-inf')
                bestAction = None
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    val = alphabeta(1, depth, successor, alpha, beta)
                    if val > value:
                        value = val
                        bestAction = action
                    if value > beta:  # Beta cutoff
                        break
                    alpha = max(alpha, value)
                if depth == 0:
                    return bestAction
                else:
                    return value

            else:  # Ghosts (Min)
                value = float('inf')
                nextAgent = (agentIndex + 1) % numAgents
                nextDepth = depth + 1 if nextAgent == 0 else depth
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    val = alphabeta(nextAgent, nextDepth, successor, alpha, beta)
                    if val < value:
                        value = val
                    if value < alpha:  # Alpha cutoff
                        break
                    beta = min(beta, value)
                return value

        return alphabeta(0, 0, gameState, float('-inf'), float('inf'))

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agentIndex, depth, state):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            if agentIndex == 0:  # Pacman (Max node)
                value = float('-inf')
                bestAction = None
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    val = expectimax(1, depth, successor)
                    if val > value:
                        value = val
                        bestAction = action
                if depth == 0:
                    return bestAction
                else:
                    return value

            else:  # Ghosts (Chance node)
                nextAgent = (agentIndex + 1) % numAgents
                nextDepth = depth + 1 if nextAgent == 0 else depth

                actions = state.getLegalActions(agentIndex)
                if not actions:
                    return self.evaluationFunction(state)

                values = []
                prob = 1 / len(actions)  # Uniform probability
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    val = expectimax(nextAgent, nextDepth, successor)
                    values.append(val)
                return sum(values) * prob

        return expectimax(0, 0, gameState)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Evaluation basée sur :
    - Score du jeu (positif)
    - Distance inverse à la nourriture la plus proche (plus on est proche, mieux c’est)
    - Distance inverse aux fantômes non effrayés (plus on est loin, mieux c’est)
    - Bonus pour les fantômes effrayés (possibilité de les manger)
    - Pénalité si proche d’un fantôme actif

    Cette fonction combine plusieurs critères pondérés afin de mieux guider Pacman dans ses décisions.
    """
    "*** YOUR CODE HERE ***"
    from util import manhattanDistance

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    scaredTimes = [ghost.scaredTimer for ghost in ghosts]

    score = currentGameState.getScore()

    # Distance à la nourriture la plus proche
    foodDistances = [manhattanDistance(pos, f) for f in food] if food else [0]
    minFoodDist = min(foodDistances) if foodDistances else 0
    foodScore = 1.0 / (minFoodDist + 1)  # +1 pour éviter division par zéro

    # Distance aux fantômes actifs
    activeGhostDistances = [manhattanDistance(pos, ghost.getPosition()) for ghost in ghosts if ghost.scaredTimer == 0]
    ghostPenalty = 0
    if activeGhostDistances:
        closestGhostDist = min(activeGhostDistances)
        if closestGhostDist <= 1:
            ghostPenalty = -500  # Pénalité forte proche d’un fantôme
        else:
            ghostPenalty = -2.0 / closestGhostDist

    # Bonus fantômes effrayés
    scaredBonus = sum([5.0 / (dist + 1) for ghost, dist in zip(ghosts, [manhattanDistance(pos, g.getPosition()) for g in ghosts]) if ghost.scaredTimer > 0])

    return score + foodScore + ghostPenalty + scaredBonus

# Abbreviation
better = betterEvaluationFunction
