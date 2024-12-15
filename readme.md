# Sequence-Generation-Library

Check testing.py to see a demo of how the library works.
The source code is in seqGenLib.py. 

## Requirements
python=3.10
numpy=1.26.2

## Usage
1. import seqGenLib 
2. Create model object with model = Model()
3. Add layers to the model with Model.addLayer(<layer name>) __OR__ Use the preset layers (with currently best performing parameters) with model.preset()
4. Generate the Sequence with output = model.generate(<number of sequence elements>)
5. Show the output with print(output) or simply use the output in further steps.
  
## Layers
There are 6 layers available. 
- Layers.preprocess
- Layers.normalDistributionLayer
- Layers.weightedBiasLayer
- Layers.normalizeLayer
- Layers.magnifyLayer
- Layers.calculateLoss
  
The first one must be the preprocessing layer, the second one is calculateLoss.
The layers after can be varied both in order and usage as well as input parameters.
For input parameters please refer to the source code or use an appropriate IDE.
  
To use the layers with the currently most optimal parameters and order use model.preset()
 
## Model
A Model Object has 4 exposed methods.
- Model.preset(): Set the optimal parameters
- Model.addLayer(): Add a layer to the list
- Model.log(): Show all currently selected layers in order
- Model.generate(): Generate the Sequence

## AUX(Auxiliary Functions)
These functions are used by the other two classes to perform certain calculations.
Its methods should generally not be called externally with the exception of AUX.translate(sequence).
It is used to translate the sequence from Distance Notation to A/B notation. 
