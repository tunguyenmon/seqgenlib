import numpy as np

class AUX:
    def normalize(x):
        return 2/(1+np.exp(x))

    def magnify(x,m):
        return x**m

    def __getNonZeroIndexes(arr):
        res = []
        for i in arr:
            if i != 0:
                res.append(i)
        return res

    def getArraySum(arr):
        res = 0
        for i in arr:
            res += i
        return res

    def getPreviousSums(sequence):
        sumarray = [0]
        for i in range(1,5):
            sumarray.append(sumarray[-1]+sequence[-i])
        sumarray.pop(0)

        return sumarray

    def getMinimum(sumarray):
        minimum = 1
        if sumarray[0] < 5:
            minimum = 5-sumarray[0]
        if sumarray[1] < 10:
            if 10-sumarray[1] > minimum:
                minimum = 10-sumarray[1]
        if sumarray[2] < 15:
            if 15-sumarray[2] > minimum:
                minimum = 15-sumarray[2]
        
        return minimum
    
    def getGlobalLinearRunningVariable(n: int, N: int)-> float:
        return n/N
    
    def linearLayer(q: float, i: int) -> float:
        return (-q*i)/10 + q
        
    def exponentialLayer(q: float, i: int) -> list:
        return -(q*np.exp(i-3.803*q))/10 + (9*q)/10
    
    def softmax(inp: list) -> list:
        softmaxSum = 0
        for i in range(10):
            softmaxSum += np.exp(inp[i]) 

        index = 0
        for i in inp:
            inp[index] = np.exp(inp[index])/softmaxSum
            index +=1
        return inp
    
    def normalDistribution(x: int, sigma, mu=5):
        # NOTE!! This normal distribution is set so that 0 and 10 are exactly zero.
        return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(1/2)*((x-mu)/sigma)**2)-((1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(1/2)*(-mu/sigma)**2))
    
    def getGlobalStrictnessAmplifier(n: int, N : int, maximum)-> int:
        return n/N*maximum
    
    def translate(sequence: list[int]):
        outputList = []
        for i in range(len(sequence)):
            for j in range(i-1):
                #print("j: " + str(j))
                outputList.append("A")
            outputList.append("B")

        return outputList
    
class Layers:  
    def preprocess(**kwargs)-> list:
        seq = kwargs["sequence"]
        upperLimit = kwargs["upperLimit"]
        lowerLimit = kwargs["lowerLimit"]

        sumarray = AUX.getPreviousSums(seq)
        minimum = AUX.getMinimum(sumarray)

        output = []

        for i in range(upperLimit):
            if i < lowerLimit:
                output.append(0) 
            else:
                output.append(1)
        
        for i in range(len(output)):
            if i+1<minimum:
                output[i] = 0

        return output

    def normalDistributionLayer(output: list[int], **kwargs):
        sigma = kwargs["sigma"]
        for i in range(len(output)):
            output[i] *= 10*AUX.normalDistribution(i+1, sigma=sigma) + 1 
        
        return output
    
    def weightedBiasLayer(output: list, **kwargs) -> list:
        bias = kwargs["bias"]
        N = kwargs["targetLength"]
        n = kwargs["n"]
        weightFunction = kwargs["weightFunction"]
        
        q = (10/9)* AUX.getGlobalLinearRunningVariable(n, N)
        for i in range(len(output)):
            weight = weightFunction(q, i+1) 
            weightedBias = bias * (weight if weight > 0 else 0)
            output[i] = output[i]*(1+weightedBias)
            #print(weightedBias)
        return output
    
    def probabilityOutputLayer(output: list, **kwargs) -> list:
        outputTotal = AUX.getArraySum(output)    

        for i in range(len(output)): 
            if output[i] != 0:
                output[i] = 1000*(output[i]/outputTotal)
        
        return output
    
    def normalizeLayer(output: list, **kwargs) -> list:
        for i in range(len(output)):
            if output[i] != 0:
                output[i] = AUX.normalize(output[i])

        return output

    def magnifyLayer(output: list, **kwargs) -> list:
        strictness = kwargs["strictness"]
        for i in range(len(output)):
            if output[i] != 0:
                output[i] = AUX.magnify(output[i], strictness)
        
        return output

    def calculateLoss(output: list, **kwargs) -> list:
        sequence = kwargs["sequence"]
        runningVar = kwargs["runningVar"]
        EPSILON = 1e-10

        sequenceTotalSum = AUX.getArraySum(sequence)
        for i in range(len(output)):
            if output[i] != 0:
                n_plus_1 = len(sequence) + 1
                totalSum_plus_values = sequenceTotalSum + i + 1
                onePercentage = n_plus_1/totalSum_plus_values
                distance20 = 0.2-onePercentage
                output[i] = ((10+1100*runningVar)*(distance20))**2 + EPSILON if distance20 >= 0 else 0
        
        return output
    
class Model:
    # Attributes
    layerList = []
    startSequence = [5,5,5,5]

    # Methods 
    def __init__(self, layerList=[Layers.preprocess]):
        for i in layerList:
            self.layerList.append(i)

    def preset(self):
        self.layerList = [
            Layers.preprocess,
            Layers.calculateLoss,
            Layers.normalizeLayer,
            Layers.normalDistributionLayer,
            Layers.magnifyLayer,
            #Layers.weightedBiasLayer,
            Layers.probabilityOutputLayer
        ]
        print("Following layers have been preset:")
        self.log()

    def addLayer(self, layer):
        self.layerList.append(layer)

    def log(self):
        if not self.layerList:
            print("No Layers in this model")
        for e in self.layerList:
            print(e.__name__)

    def __nextInSequence(self, output: list) -> list:
        randomInt = np.random.randint(1, 1000) 
        outputsums = [0]

        for i in range(len(output)):
            outputsums.append(outputsums[-1]+output[i])

        outputsums.pop(0)
        res = 1
        for i in outputsums:
            if randomInt<i:
                break
            res += 1
        
        return res

    def __run(self,**kwargs):
        
        self.addLayer(Layers.probabilityOutputLayer)
        output = []

        for function in self.layerList:
            output = function(
                output=output, 
                sequence=kwargs["sequence"],
                upperLimit=kwargs["upperLimit"],
                lowerLimit=kwargs["lowerLimit"], 
                targetLength=kwargs["targetLength"], 
                bias=kwargs["bias"], 
                strictnessAmp=kwargs["strictnessAmp"], 
                sigma=kwargs["sigma"], 
                strictRep=kwargs["strictRep"],
                strictness=kwargs["strictness"],
                runningVar=kwargs["runningVar"],
                n=kwargs["n"],
                weightFunction=kwargs["weightFunction"]
            )
        return output
    
    def generate(self,
                targetLength,
                sequence=[5,5,5,5],
                upperLimit = 8,
                lowerLimit = 2, 
                bias = 1.5, 
                strictnessAmp = 22, 
                sigma = 5, 
                strictRep = 44,
                weightFunction = AUX.exponentialLayer):
        
        for i in range(targetLength):
            n = len(sequence)- 4 + 1
            runningVar = AUX.getGlobalLinearRunningVariable(n,targetLength)
            globalStrictnessAmplifier = AUX.getGlobalStrictnessAmplifier(n, targetLength, strictnessAmp)
            strictness = (n%strictRep+1) + globalStrictnessAmplifier
            
            output = self.__run(targetLength=targetLength,
                                sequence=sequence,
                                upperLimit=upperLimit,
                                lowerLimit=lowerLimit, 
                                bias=bias, 
                                strictnessAmp=strictnessAmp, 
                                sigma=sigma, 
                                strictRep=strictRep,
                                runningVar=runningVar,
                                strictness=strictness,
                                n=n,
                                weightFunction = weightFunction)
            
            sequence.append(self.__nextInSequence(output))
        
        for i in range(4):
            sequence.pop(0)
    
        return sequence
            


        

     

