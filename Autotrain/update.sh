mkdir ./ScriptsIAHelios
cp -r ../PythonScripts/* ./ScriptsIAHelios/
#Tensorflow
git clone https://github.com/tensorflow/models
cp ./CustomTfScripts/model_main.py ./models/research/object_detection/
cd models/research
python3 setup.py sdist
cd slim && python3 setup.py sdist && cd ../../..
#CocoAPI
git clone https://github.com/cocodataset/cocoapi
cd cocoapi/PythonAPI
python3 setup.py install || sudo python3 setup.py install
