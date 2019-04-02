class master_data_cleaner_classifier:
	def __init__(self):
		import pandas as pd
		m=master_data_loader()
		self.data=m.get_master_data()
		self.COaqi=[]
		self.Ozoneaqi=[]
		self.NO2aqi=[]
		self.SO2aqi=[]
		self.PM2aqi=[]

	def load_aqi(self,parameter):
		if parameter=='CO':
			self.COaqi=[]
			for idx,val in self.data['CO'].iteritems():
				if val>= 0 and val <=2.0:
					I=(50)*(val)
				elif val>2.0 and val <=17:
					I=(((199)/(14.9))*(val-2.1))+101
				elif val>17 and val <=34:
					I=(((199)/(17))*(val-17))+301
				elif val>34:
					I=(((199)/(17))*(val-17))+301
				self.COaqi.append(I)
			return self.COaqi
		if parameter=='OZONE':
			for idx,val in self.data['OZONE'].iteritems():
				if val>=0 and val <=100:
					I=val
				elif val>100 and val <=208:
					I=((1.86)*(val-101))+101
				elif val>208 and val <=748:
					I=((11.706)*(val-209))+301
				elif val > 748:
					I=((11.706)*(val-209))+301
				self.Ozoneaqi.append(I)
			return self.Ozoneaqi
		if parameter=='NO2':
			for idx,val in self.data['NO2'].iteritems():
				if val>=0 and val <=80:
					I=(1.25)*(val)
				elif val>80 and val <=280:
					I=(val-81)+101
				elif val>280 and val <=400:
					I=((1.672)*(val-281))+301
				elif val > 400:
					I=((1.672)*(val-281))+301
				self.NO2aqi.append(I)
			return self.NO2aqi
		if parameter=='SO2':
			for idx,val in self.data['SO2'].iteritems():
				if val>=0 and val <=80:
					I=((1.25)*(val))+0
				elif val>80 and val <=800:
					I=((0.277)*(val-81))+101
				elif val>800 and val <=1600:
					I=((0.249)*(val-801))+301
				elif val > 1600:
					I=((0.249)*(val-801))+301
				self.SO2aqi.append(I)
			return self.SO2aqi
		if parameter=='PM2.5':
			for idx,val in self.data['PM2.5'].iteritems():
				if val>=0 and val <=100:
					I=val
				elif val>100 and val <=350:
					I=((0.799)*(val-101))+101
				elif val>350 and val <=430:
					I=((2.519)*(val-351))+301
				elif val > 430:
					I=((2.519)*(val-351))+301
				self.PM2aqi.append(I)
			return self.PM2aqi
	
	def clean_classify_master_data(self):
		import pandas as pd
		from pandas import DataFrame as df
		COaqi=self.load_aqi('CO')
		Ozoneaqi=self.load_aqi('OZONE')
		NO2aqi=self.load_aqi('NO2')     
		SO2aqi=self.load_aqi('SO2')
		PM2aqi=self.load_aqi('PM2.5')
		frame=[self.data,pd.DataFrame({'CO_AQI':COaqi}),pd.DataFrame({'OZONE_AQI':Ozoneaqi}),pd.DataFrame({'NO2_AQI':NO2aqi}),pd.DataFrame({'SO2_AQI':SO2aqi}),pd.DataFrame({'PM2.5_AQI':PM2aqi})]
		aqi_data = pd.concat(frame,axis=1,ignore_index=False)
		aqi_data.to_csv("masterdataaqi.csv", sep=',', encoding='utf-8',index=False)
		#DATA CLEANING :
		#DROP ALL COLUMNS WHERE NO VALUES WERE RECORDED
		#ADD AVG_AQI COLUMN TO FIND THE AVG AQI
		#CLASSIFY AS POLLUTES AND NOT POLLUTED BASED ON MEAN OF ALL AQI VALUES
		#1==Polluted
		#0==Not Polluted

		clean_aqi_data = aqi_data.dropna(axis=0,how='any')
		means=[clean_aqi_data,pd.DataFrame({'AQI_AVG':list(clean_aqi_data.iloc[:,10:15].mean(axis=1))})]
		clean_aqi_data=pd.concat(means,axis=1,ignore_index=False)
		#clean_aqi_data['AQI_AVG']=(clean_aqi_data.iloc[:,10:15].mean(axis=1))
		critical_limit=clean_aqi_data['AQI_AVG'].mean()
		status_list=[]
		
		for idx,val in clean_aqi_data['AQI_AVG'].iteritems():
			if val > critical_limit:
				status_list.append('1')
			elif val<=critical_limit:
				status_list.append('0')
		ls=[clean_aqi_data,pd.DataFrame({'STATUS':status_list})]
		clean_aqi_data=pd.concat(ls,axis=1,ignore_index=False)
		old=int(aqi_data['DATE'].count())
		new=int(clean_aqi_data['DATE'].count())
		no_of_days_malfunction=old-new
		
		polluted_data=clean_aqi_data[clean_aqi_data['STATUS']=='1']
		not_polluted_data=clean_aqi_data[clean_aqi_data['STATUS']=='0']
		
		TOTAL_DAYS=old
		SYSTEM_WORKING_DAYS=new
		HARDWARE_ACCURACY=(new/old)*100
		POLLUTED_DAYS=polluted_data['DATE'].count()
		NOT_POLLUTED_DAYS=not_polluted_data['DATE'].count()

		print('TOTAL_DAYS :',TOTAL_DAYS)
		print('SYSTEM_WORKING_DAYS :',SYSTEM_WORKING_DAYS)
		print('HARDWARE_ACCURACY :',HARDWARE_ACCURACY)
		print('POLLUTED_DAYS :',POLLUTED_DAYS)
		print('NOT_POLLUTED_DAYS :',NOT_POLLUTED_DAYS)
		file = open("DATA_CLASSIFICATION_REPORT.txt","w")
		file.write('######################################################################################################################################'+'\n')
		file.write('DATA CLASSIFICATION REPORT FOR THE GIVEN DATA -  ANALYSIS OF AIR QUALITY IN CHENNAI'+'\n')
		file.write('TOTAL_DAYS :'+str(TOTAL_DAYS)+'\n')
		file.write('SYSTEM_WORKING_DAYS :'+str(SYSTEM_WORKING_DAYS)+'\n')
		file.write('HARDWARE_ACCURACY :'+str(HARDWARE_ACCURACY)+'\n')
		file.write('POLLUTED_DAYS :'+str(POLLUTED_DAYS)+'\n')
		file.write('NOT_POLLUTED_DAYS :'+str(NOT_POLLUTED_DAYS)+'\n')
		file.write('######################################################################################################################################'+'\n')
		file.write("FILE LOCATION OF CLASSFIED DATA :"+'\n'+"D:\\Mini-Project\\Chennai\\Final\\masterdataaqi_clean.csv"+'\n'+'\n'+"FILE LOCATION OF POLLUTED DATA :"+'\n'+"D:\\Mini-Project\\Chennai\\Final\\polluted_data.csv"+'\n'+'\n'+"FILE LOCATION OF UN-POLLUTED DATA :"+'\n'+"D:\\Mini-Project\\Chennai\\Final\\not_polluted_data.csv"+'\n')
		file.write('######################################################################################################################################'+'\n')
		file.write('# 1== > Polluted'+'\n'+ "# 0 ==> Not Polluted"+'\n')
		file.write('######################################################################################################################################'+'\n')
		#1==Polluted
		#0==Not Polluted
		file.close()


		clean_aqi_data.to_csv("masterdataaqi_clean.csv", sep=',', encoding='utf-8',index=False)
		polluted_data.to_csv("polluted_data.csv", sep=',', encoding='utf-8',index=False)
		not_polluted_data.to_csv("not_polluted_data.csv", sep=',', encoding='utf-8',index=False)
		
		return clean_aqi_data
	
	def get_master_aqi_data(self):
		import pandas as pd
		master_AQI_data=pd.read_csv("masterdataaqi.csv")
		return master_AQI_data
	
	def get_clean_master_aqi_data(self):
		import pandas as pd
		clean_aqi_data=pd.read_csv("masterdataaqi_clean.csv")
		return clean_aqi_data

	def get_polluted_data(self):
		import pandas as pd
		polluted_data=pd.read_csv("polluted_data.csv")
		return polluted_data

	def get_not_polluted_data(self):
		import pandas as pd
		not_polluted_data=pd.read_csv("not_polluted_data.csv")
		return not_polluted_data

a=master_data_cleaner_classifier()
a.clean_classify_master_data()