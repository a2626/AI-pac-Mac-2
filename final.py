# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

import sys

sys.path.append("teams/<COMPAI>/")


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAstar', second='DefensiveMontecarlo'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########
"""
Base Agent with utility and common functions for a* agent and MonteCarlo agent 
"""


class TeamAgent(CaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''

        self.start = gameState.getAgentPosition(self.index)
        # print 'START POSITION'
        # print  self.start
        self.lastAction = gameState.getLegalActions(self.index)
        self.goalPosition = None;
        self.homePositions = None;

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getMyTeamColor(self, gameState):
        # agentState = gameState.getAgentState(self.index);
        if gameState.isOnRedTeam(self.index):
            return 'R'
        else:
            return 'B'

    def isGhost(self, gameState, index):
        """
        Returns true ONLY if we can see the agent and it's definitely a ghost
        """
        position = gameState.getAgentPosition(index)
        if position is None:
            return False

        agentState = gameState.getAgentState(index);
        return not agentState.isPacman

    def isScared(self, gameState, index):
        """
        Says whether or not the given agent is scared
        """
        timer = gameState.data.agentStates[index].scaredTimer;
        # print 'scared timer ' + str(timer)
        isScared = bool(gameState.data.agentStates[index].scaredTimer)
        return isScared

    def getBorderHomePositions(self, gameState):
        if self.homePositions == None:
            if self.getMyTeamColor(gameState) == 'R':
                self.homePositions = self.halfGrid(gameState.data.layout, True);
            else:
                self.homePositions = self.halfGrid(gameState.data.layout, False);

    def getClosestHomePosition(self, gameState, pos):
        minPos = min(self.getMazeDistance(h, pos) for h in self.homePositions);
        return minPos;

    def getNoScaredGhostPositions(self, gameState):
        opponents = self.getOpponents(gameState)
        ghosts = [];
        indx = None;
        for o in opponents:

            ghostPos = gameState.getAgentPosition(o);

            if ghostPos != None:
                if self.isGhost(gameState, o) and not self.isScared(gameState, o):
                    ghosts.append((ghostPos, o));

        return ghosts;

    def getNoScaredClosestGhostDistance(self, pos, gameState):
        ghosts = self.getNoScaredGhostPositions(gameState);
        minGhostDistance = None;
        ghost = None
        ghostIndx = None;
        if len(ghosts) > 0:
            minG = 99999;
            for g in ghosts:
                gpos, gindx = g;
                # dist = util.manhattanDistance(pos, gpos)
                dist = self.getMazeDistance(pos, gpos);
                if dist < minG:
                    minG = dist;
                    ghost = gpos;
                    ghostIndx = gindx;

            minGhostDistance = minG

        return minGhostDistance, ghost, ghostIndx;

    def getGhostPositions(self, gameState):
        opponents = self.getOpponents(gameState)
        ghosts = [];
        indx = None;
        for o in opponents:

            ghostPos = gameState.getAgentPosition(o);

            if ghostPos != None:
                if self.isGhost(gameState, o):
                    ghosts.append((ghostPos, o));

        return ghosts;

    def getClosestGhostDistance(self, pos, gameState):
        ghosts = self.getNoScaredGhostPositions(gameState);
        minGhostDistance = None;
        ghost = None
        ghostIndx = None;
        if len(ghosts) > 0:
            minG = 99999;
            for g in ghosts:
                gpos, gindx = g;
                # dist = util.manhattanDistance(pos, gpos)
                dist = self.getMazeDistance(pos, gpos);
                if dist < minG:
                    minG = dist;
                    ghost = gpos;
                    ghostIndx = gindx;

            minGhostDistance = minG

        return minGhostDistance, ghost, ghostIndx;

    def getNumberGhostsOneStepAway(self, gameState, pos):
        walls = gameState.getWalls();
        ghosts = self.getNoScaredGhostPositions(gameState)
        gposL = [];
        for g in ghosts:
            gpos, gindx = g;
            gposL.append(gpos);

        numberGhostsOneStepAway = 0;
        if len(ghosts) > 0:
            numberGhostsOneStepAway = sum(pos in game.Actions.getLegalNeighbors(gh, walls) for gh in gposL)

        return numberGhostsOneStepAway;

    def closestSafeFood(self, gameState, pos, food):
        if len(food) > 0:
            minFood = 0;
            minDis = 9999999
            for f in food:
                if f in self.getCapsules(gameState):
                    minFood = f;
                    minDis = self.getMazeDistance(pos, f);
                    break;

                foodSafeOfGhosts = self.getNumberGhostsOneStepAway(gameState, f) == 0;
                if (self.getMazeDistance(pos, f) < minDis and foodSafeOfGhosts):
                    minFood = f;
                    minDis = self.getMazeDistance(pos, f)

            return minFood, minDis

        return None, None;

    def closestTotalFood(self, gameState, pos, food):
        if len(food) > 0:
            minFood = 0;
            minDis = 9999999
            for f in food:
                if f in self.getCapsules(gameState):
                    minFood = f;
                    minDis = self.getMazeDistance(pos, f);
                    break;

                if (self.getMazeDistance(pos, f) < minDis):
                    minFood = f;
                    minDis = self.getMazeDistance(pos, f)

            return minFood, minDis

        return None, None;

    def closestFood(self, pos, food):
        if len(food) > 0:
            minFood = 0;
            minDis = 9999999
            for f in food:
                if (self.getMazeDistance(pos, f) < minDis):
                    minFood = f;
                    minDis = self.getMazeDistance(pos, f)

            return minFood, minDis

        return None, None;

    def closestSafeCapsule(self, gameState, pos, capsules):
        if len(capsules) > 0:
            minFood = 0;
            minDis = 9999999
            for f in capsules:
                foodSafeOfGhosts = self.getNumberGhostsOneStepAway(gameState, f) == 0;
                if (self.getMazeDistance(pos, f) < minDis and foodSafeOfGhosts):
                    minFood = f;
                    minDis = self.getMazeDistance(pos, f)

            return minFood, minDis

        return None, None

    def closestCapsule(self, pos, capsules):
        if len(capsules) > 0:
            minFood = 0;
            minDis = 9999999
            for f in capsules:
                if (self.getMazeDistance(pos, f) < minDis):
                    minFood = f;
                    minDis = self.getMazeDistance(pos, f)

            return minFood, minDis

        return None, None


#####################
# Montecarlo Agents #
#####################
class OffensiveMontecarlo(TeamAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index);

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.DefenceStatus = DefensiveFn(self, self.index, gameState);
        self.OffenceStatus = OffensiveFn(self, self.index, gameState);

    def chooseAction(self, gameState):
        self.enemies = self.getOpponents(gameState)
        IamPacman = gameState.getAgentState(self.index).isPacman
        invaders = [a for a in self.enemies if gameState.getAgentState(a).isPacman]
        myPosition = gameState.getAgentPosition(self.index)
        distances = []
        for invader in invaders:
            enemyPosition = gameState.getAgentPosition(invader)

            if enemyPosition:
                distance = self.getMazeDistance(enemyPosition, myPosition)
                distances.append(distance)
        if distances:
            distToNearestPacman = min(distances)
        else:
            distToNearestPacman = 999

        if self.getScore(gameState) >= 10 or (not IamPacman and distToNearestPacman <= 4):
            return self.DefenceStatus.chooseAction(gameState)
        else:
            return self.OffenceStatus.chooseAction(gameState)


class DefensiveMontecarlo(TeamAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.DefenceStatus = DefensiveFn(self, self.index, gameState)
        self.OffenceStatus = OffensiveFn(self, self.index, gameState)

    def chooseAction(self, gameState):
        self.enemies = self.getOpponents(gameState)
        invaders = [a for a in self.enemies if gameState.getAgentState(a).isPacman]
        numInvaders = len(invaders)

        scaredTimes = [gameState.getAgentState(enemy).scaredTimer for enemy in self.enemies]

        return self.DefenceStatus.chooseAction(gameState)


##########################################
# ======= Classes used in MCTS =========
##########################################
class MontecarloFn():

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)

        evaluation = features * weights;

        return evaluation

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.agent.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}


class OffensiveFn(MontecarloFn):
    def __init__(self, agent, index, gameState):
        self.agent = agent
        self.index = index
        self.agent.distancer.getMazeDistances()

        if self.agent.getMyTeamColor(gameState) == 'R':
            limitZone = (gameState.data.layout.width - 2) / 2
        else:
            limitZone = ((gameState.data.layout.width - 2) / 2) + 1
        self.limitZone = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(limitZone, i):
                self.limitZone.append((limitZone, i))

    def getFeatures(self, gameState, action):

        features = util.Counter()

        successor = self.getSuccessor(gameState, action)

        # Score
        features['successorScore'] = self.agent.getScore(successor)

        # Current state and position of this agent...
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()

        # Closest boundary to return home

        features['closestBoundary'] = min(
            self.agent.getMazeDistance(myPos, self.limitZone[i]) for i in range(len(self.limitZone)));

        # Carrying food
        features['numCarryingFood'] = successor.getAgentState(self.index).numCarrying

        # Closest food
        foodList = self.agent.getFood(successor).asList()

        if len(foodList) > 0:
            minFood, minFoodDis = self.agent.closestFood(myPos, foodList);
            features['closestFood'] = minFoodDis;

        # Closest capsule
        capsuleList = self.agent.getCapsules(successor)

        features['closestCapsule'] = 0;
        if len(capsuleList) > 0:
            minCap, minCapDis = self.agent.closestCapsule(myPos, capsuleList);
            features['closestCapsule'] = minCapDis;

        # Closest Ghost
        visibleGhosts = 0;
        for i in self.agent.getOpponents(successor):
            ghostState = successor.getAgentState(i);
            if ghostState.getPosition() != None and not ghostState.isPacman:
                visibleGhosts += 1;

        if visibleGhosts > 0:
            closestGhostDist = self.agent.getClosestGhostDistance(myPos, successor);

            if closestGhostDist <= 5:
                features['closestGhost'] = closestGhostDist
        else:
            probabilityGhostDist = []
            for i in self.agent.getOpponents(successor):
                probabilityGhostDist.append(successor.getAgentDistances()[i])
            features['closestGhost'] = min(probabilityGhostDist)

        # Closest enemy (not ghost, opponent pacman)
        for i in self.agent.getOpponents(successor):
            ghostState = successor.getAgentState(i);
            enemiesNotGhostPacman = [];
            if ghostState.isPacman and ghostState.getPosition() != None:
                enemiesNotGhostPacman.append(ghostState.getPosition());

        if len(enemiesNotGhostPacman) > 0:
            closestEnemyNotGhosts = min(self.agent.getMazeDistance(myPos, p) for p in enemiesNotGhostPacman);
            if closestEnemyNotGhosts < 4:
                features['closestNoGhostEnemy'] = closestEnemyNotGhosts
        else:
            features['closestNoGhostEnemy'] = 0

        return features

    def getWeights(self, gameState, action):
        successor = self.getSuccessor(gameState, action)
        foodCarrying = successor.getAgentState(self.index).numCarrying

        visibleGhosts = [];
        for i in self.agent.getOpponents(successor):
            ghostState = successor.getAgentState(i);
            if not ghostState.isPacman and ghostState.getPosition() != None:
                visibleGhosts.append(ghostState);

        if len(visibleGhosts) > 0:
            for ghost in visibleGhosts:
                if ghost.scaredTimer > 0 and ghost.scaredTimer > 12:
                    return {'successorScore': 110, 'closestFood': -10, 'closestNoGhostEnemy': 0, 'closestGhost': -1,
                            'closestCapsule': 0, 'closestBoundary': 10 - 3 * foodCarrying, 'numCarryingFood': 350}
                elif ghost.scaredTimer > 0 and 6 < ghost.scaredTimer < 12:
                    return {'successorScore': 110 + 5 * foodCarrying, 'closestFood': -5, 'closestNoGhostEnemy': 0,
                            'closestGhost': -1,
                            'closestCapsule': -10, 'closestBoundary': -5 - 4 * foodCarrying, 'numCarryingFood': 100}
                # normal state of the ghost, not scared and visible...
                elif not ghost.scaredTimer > 0:
                    return {'successorScore': 110, 'closestFood': -10, 'closestNoGhostEnemy': 0,
                            'closestGhost': 20, 'closestCapsule': -15, 'closestBoundary': -15, 'numCarryingFood': 0}

        return {'successorScore': 1000 + foodCarrying * 3.5, 'closestFood': -7, 'closestGhost': 0,
                'closestNoGhostEnemy': 0,
                'closestCapsule': -5, 'closestBoundary': 5 - foodCarrying * 3, 'numCarryingFood': 350}

    def MTCS(self, depth, gameState, decay):
        new_state = gameState.deepCopy()
        value = self.evaluate(new_state, Directions.STOP)
        exponent = 1
        for i in range(0, depth):

            actions = new_state.getLegalActions(self.index)
            current_direction = new_state.getAgentState(self.agent.index).configuration.direction

            reversed_direction = Directions.REVERSE[current_direction]
            if reversed_direction in actions and len(actions) > 1:
                actions.remove(reversed_direction)

            a = random.choice(actions)
            new_state = new_state.generateSuccessor(self.agent.index, a)

            value += pow(decay, exponent) * self.evaluate(new_state, Directions.STOP)
            exponent += 1

        return value

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.agent.index)
        actions.remove(Directions.STOP)
        feasible = []
        for a in actions:
            value = self.MTCS(3, gameState.generateSuccessor(self.index, a),
                              0.7)  # decay value and depth affect the performance hugely
            feasible.append(value)

        highestValue = max(feasible)
        temp = zip(feasible, actions)
        possibleChoice = []
        for value, action in temp:
            if value == highestValue:
                possibleChoice.append((value, action))

        return random.choice(possibleChoice)[1]


class DefensiveFn(MontecarloFn):
    def __init__(self, agent, index, gameState):
        self.index = index
        self.agent = agent
        self.DefendList = {}
        teamColorRed = self.agent.red
        self.target = None
        self.previousFood = None

        if teamColorRed != True:
            gate = ((gameState.data.layout.width - 2) / 2) + 1
        elif teamColorRed == True:
            gate = (gameState.data.layout.width - 2) / 2

        self.notWallList = []

        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(gate, i):
                self.notWallList.append((gate, i))

        while (gameState.data.layout.height - 2) / 2 < len(self.notWallList):
            self.notWallList.pop(0)
            self.notWallList.pop(len(self.notWallList) - 1)

        self.defenceProbability(gameState)

    def defenceProbability(self, gameState):
        total = 0
        if len(self.notWallList) > 0:
            for position in self.notWallList:
                food = self.agent.getFoodYouAreDefending(gameState).asList()

                closestFoodDistance_tmp = []
                for f in food:
                    closestFoodDistance_tmp.append(self.agent.getMazeDistance(position, f))
                closestFoodDistance = min(closestFoodDistance_tmp)

                if closestFoodDistance == 0:
                    closestFoodDistance = 1
                self.DefendList[position] = 1.0 / float(closestFoodDistance)
                total = total + self.DefendList[position]

        for i, v in (self.DefendList).iteritems():
            self.DefendList[i] = float(v) / float(total)

    def selectPatrolSpot(self):
        maxProb = -1

        for i, v in (self.DefendList).iteritems():
            if v > maxProb:
                maxProb = v

        bestPatrolSpot = []

        for i, v in (self.DefendList).iteritems():
            if v == maxProb:
                bestPatrolSpot.append(i)

        return random.choice(bestPatrolSpot)

    def chooseAction(self, gameState):

        DefendingList = self.agent.getFoodYouAreDefending(gameState).asList()

        while self.previousFood and len(self.previousFood) != len(DefendingList):
            self.defenceProbability(gameState)
            break

        myPos = gameState.getAgentPosition(self.index)
        if myPos == self.target:
            self.target = None

        enemies = []
        for i in self.agent.getOpponents(gameState):
            x = gameState.getAgentState(i)
            enemies.append(x)

        inRange = []
        for x in enemies:
            if x.isPacman and x.getPosition() != None:
                inRange.append(x)

        if len(inRange) > 0:
            tem_dis = []
            pos = []
            mini = 9999999
            pos_index = -1

            for x in inRange:
                tem_dis.append(self.agent.getMazeDistance(myPos, x.getPosition()))
                pos.append(x)

            for x, v in enumerate(tem_dis):
                if v < mini:
                    mini = v
                    pos_index = x
            for i in pos:
                y = pos[pos_index]

            eneDis, enemyPac = (mini, y)

            self.target = enemyPac.getPosition()

        elif self.previousFood != None:
            eaten = set(self.previousFood) - set(self.agent.getFoodYouAreDefending(gameState).asList())
            if len(eaten) > 0:
                tem_dis = []
                pos = []
                mini = 999999

                for f in eaten:
                    tem_dis.append(self.agent.getMazeDistance(myPos, f))
                    pos.append(f)

                for i, x in enumerate(tem_dis):
                    if x < mini:
                        mini = x
                        pos_index = i
                for i in pos:
                    y = pos[pos_index]

                closestFood, self.target = (mini, y)

        self.previousFood = self.agent.getFoodYouAreDefending(gameState).asList()

        while self.target == None:
            if len(self.agent.getFoodYouAreDefending(gameState).asList()) <= 4:
                food = self.agent.getFoodYouAreDefending(gameState).asList() + self.agent.getCapsulesYouAreDefending(
                    gameState)
                self.target = random.choice(food)
            else:
                self.target = self.selectPatrolSpot()

        actions = gameState.getLegalActions(self.index)

        feasible = []
        fvalues = []
        feasible = [a for a in actions if
                    not gameState.generateSuccessor(self.index, a).getAgentState(self.index).isPacman
                    and not a == Directions.STOP]

        fvalues = [self.agent.getMazeDistance(gameState.generateSuccessor(self.index, a).getAgentPosition(self.index),
                                              self.target) for a in actions if
                   not gameState.generateSuccessor(self.index, a).getAgentState(self.index).isPacman
                   and not a == Directions.STOP]

        min = 999999
        for f in fvalues:
            if f < min:
                min = f
        best = min

        ties = []
        for x in range(len(fvalues)):
            if fvalues[x] == best:
                ties.append((fvalues[x], feasible[x]))
        return random.choice(ties)[1]


# ---------------------- ASTAR AGENT FOR ATTACK -------------------------------- #

#####################
# A* Attacker Agent #
#####################

# TODO:
# give priority to foods in walls when scared time...
# take priority to foods in walls when no scared time...
class OffensiveAstar(TeamAgent):

    def __init__(self, index):
        CaptureAgent.__init__(self, index)

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        legalActions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''
        foodLeft = len(self.getFood(gameState).asList())
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        if foodLeft <= 2:
            print
            "food left <= 2, let's go back home!"
            self.goalPosition = self.getClosestHomePosition(gameState, myPos);  # self.start

        if len(legalActions) != 0:
            food = self.getFood(gameState).asList() + self.getCapsules(gameState)
            goal = self.getGoal(gameState, myPos, food)
            action = self.chooseActionAstar(gameState, legalActions, goal, self.getHeuristicOfAction,
                                            self.getCostsOfActions)

            self.lastAction = action
            return action;

    def evaluateCost(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getCostFeatures(gameState, action)
        weights = self.getCostWeights(gameState, action)
        return features * weights

    def evaluateHeuristic(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getHeuristicFeatures(gameState, action)  # getHeuristicFeatures(self, gameState, action):
        weights = self.getHeuristicWeights(gameState, action)
        return features * weights

    def getHeuristicOfAction(self, gameState, pos, action, goal):
        h = self.evaluateHeuristic(gameState, action)
        h = h + self.getMazeDistance(pos, goal);
        return h;

    def getCostsOfActions(self, state, actions):
        cost = 0;
        for a in actions:
            cost += self.evaluateCost(state, a);
        return cost;

    def getCostFeatures(self, gameState, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = self.getFood(gameState).asList() + self.getCapsules(gameState)
        walls = gameState.getWalls()

        features = util.Counter()

        x, y = gameState.getAgentPosition(self.index)
        dx, dy = game.Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        ghost_1step = self.getNumberGhostsOneStepAway(gameState, (next_x, next_y));

        if ghost_1step:
            # print 'Ghosts are too close, we might find another food'
            raise Exception('Ghosts are too close, we might find another food')

        if features["#-of-ghosts-1-step-away"]:
            # maybe I can find another goal food!!
            # print 'Ghosts are too close, we might find another food'
            raise Exception('Ghosts are too close, we might find another food')

        ghostDistance, ghost, ghostIndx = self.getNoScaredClosestGhostDistance((x, y), gameState);
        if ghostDistance != None and not self.isScared(gameState, ghostIndx):
            features["closest-ghost"] = 1 / (ghostDistance + 0.001)  # / (walls.width * walls.height)

        # features.divideAll(10.0)

        return features

    def getCostWeights(self, gameState, action):
        return {'closest-ghost': -1}

    def getHeuristicFeatures(self, gameState, action):
        # extract the grid of food and wall locations and get the ghost locations
        capsules = self.getCapsules(gameState);
        totalFood = self.getFood(gameState).asList() + capsules;
        walls = gameState.getWalls()

        features = util.Counter()

        features['goal-count'] = len(totalFood)  # goal count

        # compute the location of pacman after he takes the action
        x, y = gameState.getAgentPosition(self.index);

        x, y = gameState.getAgentPosition(self.index)
        dx, dy = game.Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # if there is no danger of ghosts then add the food feature
        if (next_x, next_y) in totalFood:
            features["eats-food"] = 1.0

        minFood, dist = self.closestSafeFood(gameState, (x, y), totalFood)

        if minFood != None and minFood in capsules:
            features["closest-capsule"] = 1 / (float(dist) + 0.001)
        else:
            minCap, capDist = self.closestSafeCapsule(gameState, (x, y), capsules);

            if minCap != None:
                features["closest-capsule"] = 1 / (float(capDist) + 0.001)

        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = 1 / (float(dist) + 0.001)  # / (walls.width * walls.height)

        return features

    def getHeuristicWeights(self, gameState, action):
        return {'goal-count': 1, 'eats-food': -1, 'closest-capsule': 1, 'closest-food': 1}

    def getGoal(self, gameState, pos, food):

        if self.start == gameState.getAgentPosition(self.index):
            self.goalPosition = random.choice(food);
            return self.goalPosition

        if self.goalPosition == None:
            self.goalPosition, dis = self.closestSafeFood(gameState, pos, food);

        # print 'goal set to : '
        # print self.goalPosition

        return self.goalPosition;

    def astarSearch(self, gameState, goal, heuristicFn, costFn):
        startState = gameState;
        startPos = startState.getAgentPosition(self.index);

        # list of directions to return
        path = [];
        # all visited nodes
        visited = [];

        # to return path...  the same fringe type as the search algorithm
        actions = util.PriorityQueue();

        # fringe
        states = util.PriorityQueue();

        states.push(startState, 0);

        while not states.isEmpty():
            visitedState = states.pop();
            #
            # print 'visited state'
            # print visitedState

            visitedActions = self.getPath(visitedState, actions);

            if visitedState not in visited:
                visited.append(visitedState);

                if visitedState.getAgentPosition(self.index) == goal:
                    # print('I found the goal: ' + str(visitedState));
                    # the path...
                    path = visitedActions;
                    # print('Path to return : ' + str(path));
                    break;

                for action in visitedState.getLegalActions(self.index):
                    successorState = self.getSuccessor(visitedState, action);
                    # successorState = suc[0];
                    # action = suc[1];
                    successorActions = visitedActions;  # parent node actions (path to the parent node)
                    successorActions = successorActions + [
                        action];  # add successors action (to complete the path to the successor)

                    succesorPos = successorState.getAgentPosition(self.index)

                    # cost = len(successorActions); #see which cost can we pick
                    cost = costFn(successorState, successorActions);
                    cost = cost + heuristicFn(successorState, succesorPos, action,
                                              goal);  # def getHeuristicOfAction(self, gameState, pos, action, goal):

                    states.push(successorState, cost);
                    actions.push([successorState, successorActions], cost)

        return path;

        # to get the path

    def getPath(self, state, actions):
        while not actions.isEmpty():
            c = actions.pop();
            if state in c:
                return c[1];

        return [];

    def heuristicDefault(self, pos1, pos2):
        return self.getMazeDistance(pos1, pos2);

    def chooseActionAstar(self, gameState, legalActions, goal, heuristicFn, costFn):
        # print 'hi astar!!'
        try:
            pathAstar = self.astarSearch(gameState, goal, heuristicFn, costFn);
            if len(pathAstar) == 0:
                self.goalPosition = None;
                return random.choice(legalActions);
            else:
                return pathAstar[0];
        except Exception as e:
            # print e;
            # traceback.print_exc()

            self.goalPosition = None;

            if self.lastAction in legalActions:
                return self.lastAction;

            return random.choice(legalActions);  # for now, maybe it would be better to reverse? :O
            # self.chooseAction(state);