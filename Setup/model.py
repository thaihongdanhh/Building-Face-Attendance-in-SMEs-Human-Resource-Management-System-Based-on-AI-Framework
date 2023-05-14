def loadModel():
	base_model = ResNet34()
	inputs = base_model.inputs[0]
	arcface_model = base_model.outputs[0]
	arcface_model = keras.layers.BatchNormalization(momentum=0.9, epsilon=2e-5)(arcface_model)
	arcface_model = keras.layers.Dropout(0.4)(arcface_model)
	arcface_model = keras.layers.Flatten()(arcface_model)
	arcface_model = keras.layers.Dense(512, activation=None, use_bias=True, kernel_initializer="glorot_normal")(arcface_model)
	embedding = keras.layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name="embedding", scale=True)(arcface_model)
	model = keras.models.Model(inputs, embedding, name=base_model.name)
	
	#---------------------------------------
	#check the availability of pre-trained weights
	
	home = str(Path.home())
	
	url = "https://drive.google.com/uc?id=1LVB3CdVejpmGHM28BpqqkbZP5hDEcdZY"
	file_name = "arcface_weights.h5"
	output = home+'/.deepface/weights/'+file_name
	Path(home+'/.deepface/weights/').mkdir(parents=True, exist_ok=True)
	if os.path.isfile(output) != True:

		print(file_name," will be downloaded to ",output)
		gdown.download(url, output, quiet=False)	
	
	#---------------------------------------
	
	try:
		model.load_weights(output)
	except:
		print("pre-trained weights could not be loaded.")
		print("You might try to download it from the url ", url," and copy to ",output," manually")
	
	return model
