#!/bin/bash

#3.1.2
cd ./ConversionImages && python3 convimg_test.py && cd ..

#3.1.3

cd ./PreparationData/DecoupeDataset && python3 cut_test.py && cd ../..
#cd PreparationData/VideoToFrame && python3 cut_test.py

#3.1.5
cd ./InitialisationIA/GenTfRecord && python3 gen_test.py && cd ../..
cd ./InitialisationIA/ParseLabels && python3 parse_test.py && cd ../..
cd ./InitialisationIA/XmlToCsv && python3 parse_test.py && cd ../..

#3.2.3
cd ./TestModel && python3 model_test.py && cd ..