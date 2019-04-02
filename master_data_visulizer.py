class master_data_visulaizer:
	import matplotlib.pyplot as plt
	def __init__(self):
		import matplotlib.pyplot as plt
		a=master_data_cleaner_classifier()
		self.all_data=a.get_master_aqi_data()
		self.clean_data=a.get_clean_master_aqi_data()
		self.polluted_days_data=a.get_polluted_data()
		self.not_polluted_days_data=a.get_not_polluted_data()
		self.tot_avg_aqi=self.clean_data['AQI_AVG'].mean()
		self.CO_mean=int(self.clean_data['CO'].mean())
		self.ozone_mean=int(self.clean_data['OZONE'].mean())
		self.so2_mean=int(self.clean_data['SO2'].mean())
		self.no2_mean=int(self.clean_data['NO2'].mean())
		self.pm25_mean=int(self.clean_data['PM2.5'].mean())

	def clean_data_visuals(self):
		import matplotlib.pyplot as plt
		#self.clean_data.groupby('MONTH').plot.bar()
		ax2=self.clean_data.plot(x="MONTH_NUM", y=['CO_AQI','OZONE_AQI','NO2_AQI','SO2_AQI','PM2.5_AQI'] ,figsize=(5,40), grid=True)
		#ax2.set_ylim(0,50)
		plt.show()
	#### ASK SIR ############################
	def polluted_data(self):
		ax1=self.polluted_days_data.groupby('MONTH_NUM')['AQI_AVG'].mean().plot.bar(figsize=(10,5))
		plt.title("Polluted Data")
		plt.ylabel("MEAN AQI_AVG FOR THE MONTH")
		plt.hlines(y=int(self.tot_avg_aqi),xmin=0, xmax=20, linewidth=2, color='r')
		ax1.set_ylim(0,1500)
		plt.show()

	def non_polluted_data(self):
		ax1=self.not_polluted_days_data.groupby('MONTH_NUM')['AQI_AVG'].mean().plot.bar(figsize=(10,5))
		plt.title("Non Polluted Data")
		plt.ylabel("MEAN AQI_AVG FOR THE MONTH")
		plt.hlines(y=int(self.tot_avg_aqi),xmin=0, xmax=20, linewidth=2, color='r')
		ax1.set_ylim(0,700)
		plt.show()


	def visualize_on_parameter(self,parameter):
		if parameter=="OZONE":
			self.all_data.groupby('MONTH_NUM')['OZONE'].mean().plot.bar()
			plt.hlines(y=self.ozone_mean,xmin=0, xmax=20, linewidth=2, color='r')
			plt.ylabel("MEAN OZONE CONC. FOR THE MONTH")
			plt.show()
		elif parameter=="SO2":
			self.all_data.groupby('MONTH_NUM')['SO2'].mean().plot.bar()
			plt.hlines(y=self.so2_mean,xmin=0, xmax=20, linewidth=2, color='r')
			plt.ylabel("MEAN SO2 CONC. FOR THE MONTH")
			plt.show()
		elif parameter=="PM2.5":
			self.all_data.groupby('MONTH_NUM')['PM2.5'].mean().plot.bar()
			plt.hlines(y=self.pm25_mean,xmin=0, xmax=20, linewidth=2, color='r')
			plt.ylabel("MEAN PM2.5 CONC. FOR THE MONTH")
			plt.show()
		elif parameter == "NO2":
			self.all_data.groupby('MONTH_NUM')['NO2'].mean().plot.bar()
			plt.hlines(y=self.no2_mean,xmin=0, xmax=20, linewidth=2, color='r')
			plt.ylabel("MEAN NO2 CONC. FOR THE MONTH")
			plt.show()
		elif parameter =="CO":
			self.all_data.groupby('MONTH_NUM')['CO'].mean().plot.bar()
			plt.hlines(y=self.CO_mean,xmin=0, xmax=20, linewidth=2, color='r')
			plt.ylabel("MEAN CO CONC. FOR THE MONTH")
			plt.show()

import matplotlib.pyplot as plt
v=master_data_visulaizer()
v.polluted_data()
v.non_polluted_data()
v.clean_data_visuals()
v.visualize_on_parameter('CO')
v.visualize_on_parameter('OZONE')
v.visualize_on_parameter('SO2')
v.visualize_on_parameter('NO2')
v.visualize_on_parameter('PM2.5')

plt.show()
