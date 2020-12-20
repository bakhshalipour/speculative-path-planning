#! /usr/bin/env python3

import sys
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np



def makeGroupedBarChart(xAxisLabels, allBars, chartTitle, yAxisTitle, annotate):

    barsLabels = list(allBars.keys())
    barsValues = list(allBars.values())

    x = np.arange(len(xAxisLabels))  # the label locations
    d = 0.8 * 1/len(barsLabels) * (x[1] - x[0])  # the width of the bars

    fig, ax = plt.subplots()
    rects = []
    for i in range(len(barsLabels)):
        rects.append(ax.bar(x - (len(barsLabels)/2-i)*d, barsValues[i], d, label=barsLabels[i]))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(yAxisTitle)
    ax.set_title(chartTitle)
    ax.set_xticks(x)
    ax.set_xticklabels(xAxisLabels)
    ax.legend()

    def autolabel(rects, normalizedNumbers=None):
        """Attach a text label above each bar in *rects*, displaying its height."""
        assert normalizedNumbers == None or len(normalizedNumbers) == len(rects)
        for i in range(len(rects)):
            height = rects[i].get_height()
            ax.annotate('{:.2f}'.format(height) if normalizedNumbers == None else '{:.2f}x'.format(normalizedNumbers[i]),
                    xy=(rects[i].get_x() + rects[i].get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90)

    if annotate:
        normalizedNumbers = []
        baselineNumbers = barsValues[0]
        for i in range(len(barsValues)):
            thisBar = barsValues[i]
            thisNormNumbers = []
            for j in range(len(thisBar)):
                thisNormNumbers.append(thisBar[j]/baselineNumbers[j])
            normalizedNumbers.append(thisNormNumbers)

        for i in range(len(rects)):
            autolabel(rects[i], normalizedNumbers[i])

    fig.tight_layout()
    fileName = chartTitle.replace(' ', '').replace('=', '').replace('(', '').replace(')', '') + '.png'
    plt.savefig(fileName);



def parseInput(line):
    pattern = "Results: map_file=inputset\/(.*\.txt), EPSILON=(\d*), " \
    "MAX_THREADS=(\d*), DO_SPECULATION=(\d), time=(.*?), "\
    "StraightExpansions=(.*?), SpeculativeUsefull=(.*?), "\
    "AverageSpecLoopIterations=(.*?), ActiveThreadsPerExpansion=(.*?), "\
    "NonSpeculativeActiveThreadsPerExpansion=(.*?), "\
    "SpeculativeActiveThreadsPerExpansion=(.*?)$"

    grps = re.search(pattern, line)
    mapName = grps.group(1)
    epsilon = int(grps.group(2))
    threadCount = int(grps.group(3))
    speculation = int(grps.group(4))
    execTime = float(grps.group(5))
    straightExpansions = float(grps.group(6))
    usefulSpeculations = float(grps.group(7))
    averageSpecLoopIterations = float(grps.group(8))
    averageThreadsPerExpansion = float(grps.group(9))
    averageNonSpecThreadsPerExpansion = float(grps.group(10))
    averageSpecThreadsPerExpansion = float(grps.group(11))

    return mapName, epsilon, threadCount, speculation, execTime, \
            straightExpansions, usefulSpeculations, averageSpecLoopIterations, \
            averageThreadsPerExpansion, averageNonSpecThreadsPerExpansion, \
            averageSpecThreadsPerExpansion

if __name__ == '__main__':

    if len(sys.argv[1:]) != 1:
        print('Usage: {} path/to/result_file'.format(sys.argv[0]))
        exit()

    resFile = open(sys.argv[1], 'r')
    lines = resFile.readlines()

    allResults = []
    allEpsilons = []
    for line in lines:
        if line.startswith('Results:'):
            mapName, epsilon, threadCount, speculation, execTime, \
            straightExpansions, usefulSpeculations, averageSpecLoopIterations, \
            averageThreadsPerExpansion, averageNonSpecThreadsPerExpansion, \
            averageSpecThreadsPerExpansion = parseInput(line)

            if epsilon not in allEpsilons: allEpsilons.append(epsilon)

            theDict = defaultdict(list)
            theDict['mapName'] = mapName.replace('.txt', '')
            theDict['epsilon'] = epsilon
            theDict['threadCount'] = threadCount
            theDict['speculation'] = speculation
            theDict['execTime'] = execTime
            theDict['straightExpansions'] = straightExpansions
            theDict['usefulSpeculations'] = usefulSpeculations
            theDict['averageSpecLoopIterations'] = averageSpecLoopIterations
            theDict['averageThreadsPerExpansion'] = averageThreadsPerExpansion
            theDict['averageNonSpecThreadsPerExpansion'] = averageNonSpecThreadsPerExpansion
            theDict['averageSpecThreadsPerExpansion'] = averageSpecThreadsPerExpansion
            allResults.append(theDict)



    '''
    Performance graphs for various epsilons:
        X-axis: map
        Y-axis: execution time
        bars: threadCount + speculation?
    '''
    for eps in allEpsilons:
        xAxisLabels = []
        allBars = defaultdict(list)
        for thisDict in allResults:
            if thisDict['epsilon'] != eps: continue
            mapName = thisDict['mapName']
            threadCount = thisDict['threadCount']
            speculation = thisDict['speculation']
            execTime = thisDict['execTime']
            thisBarLabel = 't' + str(threadCount) + ',s' + str(speculation)
            if mapName not in xAxisLabels:
                xAxisLabels.append(mapName)
            allBars[thisBarLabel].append(execTime)

        yAxisTitle = 'Execution Time (seconds)'
        chartTitle = 'Performance Comparison of Various Configurations (Epsilon={})'.format(eps)
        makeGroupedBarChart(xAxisLabels, allBars, chartTitle, yAxisTitle, True)



    '''
    Speculative usefulness graph:
        X-axis: map
        Y-axis: usefulness
        bars: threadCount
        *** epsilon == 1 in this graph, threadCount > 1
    '''
    xAxisLabels = []
    allBars = defaultdict(list)
    for thisDict in allResults:
        if thisDict['epsilon'] != 1 or thisDict['speculation'] == 0 or thisDict['threadCount'] == 1: continue
        mapName = thisDict['mapName']
        threadCount = thisDict['threadCount']
        usefulSpeculations = thisDict['usefulSpeculations']
        thisBarLabel = 't' + str(threadCount)
        if mapName not in xAxisLabels:
            xAxisLabels.append(mapName)
        allBars[thisBarLabel].append(usefulSpeculations)

    yAxisTitle = 'Useful Speculations (%)'
    chartTitle = 'The Effectiveness of Speculations (Epsilon=1)'
    makeGroupedBarChart(xAxisLabels, allBars, chartTitle, yAxisTitle, False)



    '''
    Average number of nodes evaluated in the speculative mode:
        X-axis: map
        Y-axis: average speculative nodes (== loops); not to be confused with
        the number of neighbours. Y-axis * number of motions == number of
        neighbours evaluated
        bars: threadCount
        *** epsilon == 1 in this graph
    '''
    xAxisLabels = []
    allBars = defaultdict(list)
    for thisDict in allResults:
        if thisDict['epsilon'] != 1 or thisDict['speculation'] == 0 or thisDict['threadCount'] == 1: continue
        mapName = thisDict['mapName']
        threadCount = thisDict['threadCount']
        averageNodes = thisDict['averageSpecLoopIterations']
        thisBarLabel = 't' + str(threadCount)
        if mapName not in xAxisLabels:
            xAxisLabels.append(mapName)
        allBars[thisBarLabel].append(averageNodes)

    yAxisTitle = 'Average Number of Nodes'
    chartTitle = 'Lookahead of Speculative Execution (Epsilon=1)'
    makeGroupedBarChart(xAxisLabels, allBars, chartTitle, yAxisTitle, False)



    '''
    Average number of active speculative and non-speculative threads per expansion:
        X-axis: map
        Y-axis1: activeNonSpecThreads, Y-axis: activeNonSpecThreads
        bars: threadCount
        *** epsilon == 1 in this graph
    '''
    xAxisLabels = []
    allBarsNonSpec = defaultdict(list)
    allBarsSpec = defaultdict(list)
    for thisDict in allResults:
        if thisDict['epsilon'] != 1 or thisDict['speculation'] == 0 or thisDict['threadCount'] == 1: continue
        mapName = thisDict['mapName']
        threadCount = thisDict['threadCount']
        activeNonSpecThreads = thisDict['averageNonSpecThreadsPerExpansion']
        activeSpecThreads = thisDict['averageSpecThreadsPerExpansion']
        thisBarLabel = 't' + str(threadCount)
        if mapName not in xAxisLabels:
            xAxisLabels.append(mapName)
        allBarsNonSpec[thisBarLabel].append(activeNonSpecThreads)
        allBarsSpec[thisBarLabel].append(activeSpecThreads)

    yAxisTitle = 'Active Non-Speculative Threads Threads'
    chartTitle = 'Active Threads (Epsilon=1, Speculation=ON)'
    makeGroupedBarChart(xAxisLabels, allBarsNonSpec, chartTitle, yAxisTitle, False)

    yAxisTitle = 'Active Speculative Threads Threads'
    chartTitle = 'Active Threads (Epsilon=1)'
    makeGroupedBarChart(xAxisLabels, allBarsSpec, chartTitle, yAxisTitle, False)


    plt.show()

