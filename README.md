# AIR_QUALITY_ANALYSIS
#PROJECT WAS FIRST STARTED WITH LIMITED DATA (150 ROWS)
#LATER REAL TIME DATA FROM CHENNAI WAS OBATINED FROM  cpcb.nic.in

#DATA LOADER : written in way to support fresh data , when run with a csv file it will append it without changing existing data

#DATA CLEANER : HANDLES NULL COLUMNS and CLASSIFIES DATA BASED ON AQI : MAKES LABELLED DATA OUT OF UNLABELLED DATA

#DATA VISUALIZER : PRESENTS DATA IN GRAPHS FOR ANALYSIS

#DATA PREDICTOR : PREDICTS WHETHER A DAY IS POLLUTED or NOT BASED ON ANY 2 parameters

#RANDOM GEN : AUTOMATES THE ANALYSIS PROCESS : REPLACES THE TIRESOME WORK OF GOING THROUGH 100s OF GRAPHS, 
              IT AUTOMATICALLY SELECTS PARAMETERS
              SELECTS RANDOM RANGES AND CHECKS FOR ABONORMALITIES
              SUCH READINGS ARE FLAGGED AND PREARED AS A TXT REPORT
              WRITTEN TO PERFORM ANY NUMBER OF GIVEN ITERATIONS ON ANY PARAMETER
