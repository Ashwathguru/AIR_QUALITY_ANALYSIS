class random_query_generator:
	def __init__(self):
		import matplotlib.pyplot as plt
		a=master_data_cleaner_classifier()
		self.all_data=a.get_master_aqi_data()
		self.clean_data=a.get_clean_master_aqi_data()
		self.polluted_days_data=a.get_polluted_data()
		self.not_polluted_days_data=a.get_not_polluted_data()
		self.aqi_thresh=int(self.clean_data['AQI_AVG'].mean())*1.25
		self.CO_thresh=int(self.clean_data['CO'].mean())*1.25
		self.ozone_thresh=int(self.clean_data['OZONE'].mean())*1.25
		self.so2_thresh=int(self.clean_data['SO2'].mean())*1.25
		self.no2_thresh=int(self.clean_data['NO2'].mean())*1.25
		self.pm25_thresh=int(self.clean_data['PM2.5'].mean())*1.25

	def random_months(self,iter):
		import random
		clean_data_month=self.clean_data.iloc[:,[1,2,15,16]]
		choices=[1,2,3,4,5,6,7,8,9,10,11,12]
		polluted_months=[]
		i=0
		while i<=iter:
			r=random.choice(choices)
			cl_data_loop=clean_data_month[clean_data_month['MONTH_NUM']==r]
			m=cl_data_loop['AQI_AVG'].mean()
			if m>=self.aqi_thresh:
				polluted_months.append(r)
			i+=1
		return polluted_months

	def random_pollutant(self,iter):
		import random
		choices=['CO','SO2','PM2.5','OZONE','NO2']
		i=0
		file = open("random_pollutant_report.txt","w")
		#file.write('###########################################################'+str(self.so2_thresh)+'\n')
		file.write('###########################################################'+'\n')
		file.write('Auto-Generated Report by Random Query written for Automating Analysis by Ashwath and Noopur for the Analysis of Air-Quality in Chennai'+'\n')
		file.write('Step 1: The program takes in the number of iterations to be performed'+'\n') 
		file.write('Step 2: The program randomly selectes a pollutant among [CO,OZONE,NO2,SO2,PM2.5]'+'\n')
		file.write('Step 3: The program randomly selectes fixes a lower limit and an upper limit for the selected Pollutant'+'\n')
		file.write('Step 4: If the AVERAGE AQI OF THAT SUB SET OF VALUES IS HIGHER THAN NORMAL, THEN THE SUBSET IS SUBJECTED TO ANALYSIS'+'\n')
		file.write('Step 5: The subset is then compared and anlysed with every other pollutant and a brief report is generated' +'\n')
		file.write('###########################################################'+'\n')
		file.write('Given Number of Iterations'+str(iter)+'\n')
		while i<=iter:
			r=random.choice(choices)
			
			print("ITERATION NO.",i)
			file.write('ITERATION NO. :'+str(i)+'\n')
			print("SELECTED PARAMETER: ",r)
			file.write('SELECTED PARAMETER :'+r+'\n')
			lower_limit=random.randint(int(min(self.clean_data[r])),int(max(self.clean_data[r])))
			upper_limit=random.randint(lower_limit,int(max(self.clean_data[r])))
			file.write('LOWER LIMIT :'+str(lower_limit)+'\n')
			file.write('UPPER LIMIT :'+str(upper_limit)+'\n')
			sub_set1=self.clean_data[self.clean_data[r] > lower_limit]
			sub_set2=sub_set1[sub_set1[r] < upper_limit]
				
			if sub_set2['AQI_AVG'].mean()>=self.aqi_thresh:
				print("For the ",r," level between the ranges ",lower_limit,' ',upper_limit)
				print("The Average AQI is ",sub_set2['AQI_AVG'].mean(),)
				print("These values majorly occurs in the month of",list(set(sub_set2['MONTH'].values)))
				print("Total No. of days in this range: ",len(sub_set2))
				s_p=sub_set2[sub_set2['STATUS']==1]
				print("For these ranges of, ",r,", No. of polluted days :",len(s_p))
				print("For these ",r, " ranges NO2 levels are :",list(set(sub_set2['NO2'].values)))
				print("For these ",r,"  ranges OZONE levels are :",list(set(sub_set2['OZONE'].values)))
				print("For these ",r,"  ranges SO2 levels are :",list(set(sub_set2['SO2'].values)))
				print("For these ",r,"  ranges PM2.5 levels are :",list(set(sub_set2['SO2'].values))) 
 
				file.write('ITERATION NO.:'+str(i)+'\n') 
				file.write("SELECTED PARAMETER: "+r+'\n') 
				file.write("For the "+r+" level between the ranges "+str(lower_limit)+' '+str(upper_limit)+'\n') 
				file.write("The Average AQI is "+str(sub_set2['AQI_AVG'].mean())+'\n')
				file.write("These values majorly occurs in the month of "+str(list(set(sub_set2['MONTH'].values)))+'\n')
				file.write("Total No. of days in this range: "+str(len(sub_set2))+'\n')
				file.write("For these ranges of, "+r+", No. of polluted days :"+str(len(s_p))+'\n')
				file.write("For these "+r+" ranges NO2 levels are :"+str(list(set(sub_set2['NO2'].values)))+'\n')
				file.write("For these "+r+"  ranges OZONE levels are :"+str(list(set(sub_set2['OZONE'].values)))+'\n')
				file.write("For these "+r+"  ranges SO2 levels are :"+str(list(set(sub_set2['SO2'].values)))+'\n')
				file.write("For these "+r+"  ranges PM2.5 levels are :"+str(list(set(sub_set2['SO2'].values)))+'\n')
 
					 
				if r!='NO2':
					if sub_set2['NO2'].mean()>self.no2_thresh:
						print("We also observe that average NO2 for these ranges are higher than normal")
						file.write("We also observe that average NO2 for these ranges are higher than normal  ********"+'\n')
					else:
						print("NO2 is normal for this range of ",r)
						file.write("NO2 is normal for this range of "+r+'\n')
				if r!='OZONE':
					if sub_set2['OZONE'].mean()>self.ozone_thresh:
						print("We also observe that average OZONE for these ranges are higher than normal")
						file.write("We also observe that average OZONE for these ranges are higher than normal  ********"+'\n')
					else:
						print("OZONE is normal for this range of ",r)
						file.write("OZONE is normal for this range of "+r+'\n')
				if r!='SO2':
					if sub_set2['SO2'].mean()>self.so2_thresh:
						print("We also observe that average SO2 for these ranges are higher than normal")
						file.write("We also observe that average SO2 for these ranges are higher than normal  ********"+'\n')
					else:
						print("SO2 is normal for this range of ",r)
						file.write("SO2 is normal for this range of "+r+'\n')
				if r!='PM2.5':
					if sub_set2['PM2.5'].mean()>self.pm25_thresh:
						print("We also observe that average PM2.5 for these ranges are higher than normal")
						file.write("We also observe that average PM2.5 for these ranges are higher than normal  ********"+'\n')
					else:
						print("PM2.5 is normal for this range of ",r)
						file.write("PM2.5 is normal for this range of "+r+'\n')
				if r!='CO':
					if sub_set2['CO'].mean()>self.CO_thresh:
						print("We also observe that average CO for these ranges are higher than normal")
						file.write("We also observe that average CO for these ranges are higher than normal ********"+'\n')
					else:
						print("CO is normal for this range of ",r)
						file.write("CO is normal for this range of "+r+'\n')

			else:
				print("THE AVERAGE AQI OBSERVED FOR THESE RANGES OF ",r," ARE NORMAL")
				file.write("THE AVERAGE AQI OBSERVED FOR THESE RANGES OF "+r+" ARE NORMAL"+'\n')
			file.write("#####################################################################################"+'\n')
			i+=1
		file.close()



rq=random_query_generator()
print(rq.random_months(15))
rq.random_pollutant(5)