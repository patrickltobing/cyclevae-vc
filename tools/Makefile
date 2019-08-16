.PHONY: all clean

#all: venv/bin/activate venv/lib/python3.5/site-packages/torch
#all: venv/bin/activate venv/lib/python3.6/site-packages/torch
#all: venv37/bin/activate venv37/lib/python3.7/site-packages/torch
#all: venv37pt1/bin/activate venv37pt1/lib/python3.7/site-packages/torch
all: venv37pt1cu10/bin/activate venv37pt1cu10/lib/python3.7/site-packages/torch

#venv/bin/activate:
#venv37/bin/activate:
#venv37pt1/bin/activate:
venv37pt1cu10/bin/activate:
	#test -d venv37 || virtualenv -p python3.7 venv37
	#test -d venv37pt1 || virtualenv -p python3.7 venv37pt1
	test -d venv37pt1cu10 || virtualenv -p python3.7 venv37pt1cu10
	#test -d venv || virtualenv -p python3.6 venv
	#test -d venv || virtualenv -p python3.5 venv
	#. venv37/bin/activate; pip install -r requirements.txt; pip install matplotlib;
	#. venv37pt1/bin/activate; pip install -r requirements.txt; pip install matplotlib;
	. venv37pt1cu10/bin/activate; pip install -r requirements.txt; pip install matplotlib;
	#. venv/bin/activate; pip install -r requirements.txt; pip install matplotlib;
	#touch venv37/bin/activate
	#touch venv37pt1/bin/activate
	touch venv37pt1cu10/bin/activate
	#touch venv/bin/activate

#venv/lib/python3.5/site-packages/torch: venv/bin/activate
#venv/lib/python3.6/site-packages/torch: venv/bin/activate
#venv37/lib/python3.7/site-packages/torch: venv37/bin/activate
#venv37pt1/lib/python3.7/site-packages/torch: venv37pt1/bin/activate
venv37pt1cu10/lib/python3.7/site-packages/torch: venv37pt1cu10/bin/activate
	. venv37pt1cu10/bin/activate; pip install pip --upgrade; pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl; pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-linux_x86_64.whl
	#. venv37pt1/bin/activate; pip install pip --upgrade; pip3 install torch torchvision
	#. venv37/bin/activate; pip install pip --upgrade; pip install https://download.pytorch.org/whl/cu92/torch-0.4.1.post2-cp37-cp37m-linux_x86_64.whl; pip install torchvision
	#. venv/bin/activate; pip install pip --upgrade; pip install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl; pip install torchvision
	#. venv/bin/activate; pip install pip --upgrade; pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp35-cp35m-linux_x86_64.whl; pip install torchvision
	#. venv/bin/activate; pip install pip --upgrade; pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl; pip install torchvision

clean:
	#rm -fr venv37
	#rm -fr venv37pt1
	rm -fr venv37pt1cu10
	find -iname "*.pyc" -delete
