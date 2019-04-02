class master_data_predictor:
		
	
	def loader(self,par1,par2):
		import numpy as np
		import pandas as pd 
		import matplotlib.pyplot as plt
		path="masterdataaqi_clean.csv"
		dataset=pd.read_csv(path)
		
		#print("Enter the number for the required parameter || CO-4 || OZONE-5 ||  NO2-6 || SO2-7 || PM2.5-8 ||	TEMP-9 || CO_AQI-10 OZONE_AQI-11 || NO2_AQI-12 ||	SO2_AQI-13 || PM2.5_AQI- 14 || ")
		#DAY_NUMBER-0 || MONTH-1 || MONTH_NUM-2 ||	DATE-3 || CO-4 || OZONE-5 ||  NO2-6 || SO2-7 || PM2.5-8 ||	TEMP-9 || CO_AQI-10
		#OZONE_AQI-11 || NO2_AQI-12 ||	SO2_AQI-13 || PM2.5_AQI- 14 || AQI_AVG-15 ||STATUS-16
		X=dataset.iloc[:,[int(par1),int(par2)]].values
		y=dataset.iloc[:,16].values
		return X,y
	def split(self,par1,par2):
		X,y=self.loader(par1,par2)
		from sklearn.cross_validation import train_test_split
		X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0) #random_state=0 to keep the same data as train and test all the time)

		from sklearn.preprocessing import StandardScaler
		sc=StandardScaler() #Linear regression takes cae of this must define explicitly for KNN
		X_train=sc.fit_transform(X_train) #Use only for training data,Create an object that will be ready to fit our data, transform data according to the fit model
		X_test=sc.transform(X_test)
		return X_train,X_test,y_test,y_train
	def KNN(self,par1,par2):
		#self.file.write("USING KNN CLASSIFIER")
		X_train,X_test,y_test,y_train=self.split(par1,par2)
		from sklearn.neighbors import KNeighborsClassifier
		knnclassifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2) #k=5 , p=2 => Equilidian distance,p=power parameter, p=1 => ???
		knnclassifier.fit(X_train,y_train) #make machine learn from the training data
		y_pred=knnclassifier.predict(X_test)
		return y_pred
	def DT(self,par1,par2):

		X_train,X_test,y_test,y_train=self.split(par1,par2)
		from sklearn.tree import DecisionTreeClassifier
		dtclassifier = DecisionTreeClassifier(criterion='entropy',random_state=0) #entropy and information gain
		dtclassifier.fit(X_train,y_train)
		y_pred=dtclassifier.predict(X_test)
		return y_pred #make machine learn from the training data
	def NB(self,par1,par2):
		X_train,X_test,y_test,y_train=self.split(par1,par2)
		from sklearn.naive_bayes import GaussianNB
		nbclassifier=GaussianNB() #NB uses Gaussian Distribution
		nbclassifier.fit(X_train,y_train) #make machine learn from the training data
		y_pred=nbclassifier.predict(X_test)
		return y_pred
	def RF(self,par1,par2):
		X_train,X_test,y_test,y_train=self.split(par1,par2)
		from sklearn.ensemble import RandomForestClassifier
		RFclassifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0) #Using 10 DTs
		RFclassifier.fit(X_train,y_train) #make machine learn from the training data
		y_pred=RFclassifier.predict(X_test)
		return y_pred
	def LR(self,par1,par2):
		X_train,X_test,y_test,y_train=self.split(par1,par2)
		from sklearn.linear_model import LogisticRegression
		LRpredictor=LogisticRegression(random_state=0)
		LRpredictor.fit(X_train,y_train)
		y_pred=LRpredictor.predict(X_test)
		return y_pred
	def confusion_matrix(self,par1,par2):
		y_pred_knn=self.KNN(par1,par2)
		y_pred_DT=self.DT(par1,par2)
		y_pred_NB=self.NB(par1,par2)
		y_pred_LR=self.LR(par1,par2)
		y_pred_RF=self.RF(par1,par2)
		pred_list=[y_pred_knn,y_pred_DT,y_pred_NB,y_pred_LR,y_pred_RF]
		file=open("DATA_PREDICTION_REPORT.txt",'w')
		file.write("DATA PREDICTION REPORT"+'\n')
		file.write("SELECTED PARAMETERS "+str(par1)+ " AND "+str(par2)+'\n')
		file.write("|| CO-4 || OZONE-5 ||  NO2-6 || SO2-7 || PM2.5-8 ||	TEMP-9 || CO_AQI-10 OZONE_AQI-11 || NO2_AQI-12 || SO2_AQI-13 || PM2.5_AQI- 14 || AVG_AQI-15"+'\n')
		file.write("Step 1:The program takes the given parameter's readings and it's corresponding AQI_AVG as input "+'\n')
		file.write("Step 2:It trains itself in way that it can predict whether for any given paramater and AQI_AVG if the day is polluted or not "+'\n')
		#file.write("The user can select the parameter based on which the prediction is to be done using the following number codes "+'\n')
		#file.write("|| CO-4 || OZONE-5 ||  NO2-6 || SO2-7 || PM2.5-8 ||	TEMP-9 || CO_AQI-10 OZONE_AQI-11 || NO2_AQI-12 || SO2_AQI-13 || PM2.5_AQI- 14 ||"+'\n')
		X_train,X_test,y_test,y_train=self.split(par1,par2)
		coun=0
		from sklearn.metrics import confusion_matrix
		for i in pred_list:
			file.write("###############################################################################################"+'\n')
			if coun==0:
				file.write("USING KNN"+'\n')
			elif coun==1:
				file.write("USING DECESION TREE"+'\n')
			elif coun==2:
				file.write("USING NAIVE BAYES"+'\n')
			elif coun==3:
				file.write("USING LOGISTIC REGRESSION"+'\n')
			elif coun==4:
				file.write("USING RANDOM FOREST"+'\n')

			cm=confusion_matrix(y_test,i)
			file.write("CONFUSION MATRIX :"+str(cm)+'\n')
			import numpy as np
			arr=np.array(cm)
			TP=int(arr[1,1])
			TN=int(arr[0,0])
			FP=int(arr[1,0])
			FN=int(arr[0,1])

			total=TP+TN+FP+FN
			accuracy=(TP+TN)/total
			file.write("ACCURACY :"+str(accuracy*100)+'\n')
			MissclassificationRate=(FP+FN)/total
			file.write("MISS CLASSIFICATION RATE :"+str(MissclassificationRate*100)+'\n')
			TruePostiveRate=TP/(TP+FN)
			file.write("TRUE POSITIVE RATE :"+str(TruePostiveRate*100)+'\n')
			FalsePositiveRate= FP/(FP+TN)
			file.write("FALSE POSITIVE RATE :"+str(FalsePositiveRate*100)+'\n')
			TrueNegativeRate=TN/(TN+FP)
			file.write("TRUE NEGATIVE RATE :"+str(TrueNegativeRate*100)+'\n')
			Precesion=TP/(FP+TP)
			file.write("PRECESION :"+str(Precesion*100)+'\n')
			#Prevelance=total/(FP+TP)
			#file.write("PREVELANCE :"+str(Prevelance*100)+'\n')
			coun+=1
		return cm
		self.file.close()

p=master_data_predictor()
p.confusion_matrix(4,6)