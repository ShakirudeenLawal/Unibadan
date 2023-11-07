import xarray
import numpy as np
import pandas as pd
import os
import netCDF4
import datetime

#file_path = './GFDL_histo_1986_2014_SA.nc'
file_path = './pr_day_INM-CM5-0_ssp126_r1i1p1f1_gr1_19500101-20641231-RE.nc'
#output_file = './Onset-A_Reana_histo_1979_2022.nc'
output_file = './Model_INM5-onsetA.nc'

#output_file = './GFDL_histo_1979_2022.nc'
#output_file = './OnsetA-2016_1986-2015.nc' ! 1986-2015

#Open nc file in xarray
ds = xarray.open_dataset(file_path)

#Convert to mm/day
convert_pr =lambda x: x*(60.*60.*24.)
ds.variables['pr'].data = convert_pr(ds.variables['pr'].data[:])

years = np.unique(ds.time.dt.year.data)

onset = np.empty(shape=[len(years),ds.pr.shape[1],ds.pr.shape[2]], dtype=np.float32)
cessation = np.empty(shape=[len(years),ds.pr.shape[1],ds.pr.shape[2]], dtype=np.float32)
mask = np.empty(shape=[len(years),ds.pr.shape[1],ds.pr.shape[2]], dtype=bool)
dates = []

for num, year in enumerate(years):
    print(year)
    #Slice out a year (July to June)
    year_ds = ds.sel(time=slice(f'{year}-07-01', f'{year+1}-06-30'))
    #Remove daily values < 5
    year_ds.pr.data[year_ds.pr.data < 5] = 0
    #Calculate running total
    year_running_total = np.add.accumulate(year_ds.pr.data)
    #Calc %
    pr_perct = year_running_total/year_running_total[-1,:,:]*100
    
    #Conditions
    onset_condition = pr_perct>=10
    cessation_condition = pr_perct>=90
    #Get index at which condtion is met
    onset_year = onset_condition.argmax(axis=0)
    cessation_year = cessation_condition.argmax(axis=0)
    
    #Add year data to array
    onset[num,:,:] = onset_year
    cessation[num,:,:] = cessation_year
    dates.append(pd.to_datetime(f'{year}-07-01'))
    mask[num,:,:] = np.isnan(year_ds.pr.data[0,:])

#Create masked array
onset_masked = np.ma.MaskedArray(data=onset, mask=mask)
cessation_masked = np.ma.MaskedArray(data=cessation, mask=mask)

#Create NetCDF File
ncfile = netCDF4.Dataset(output_file, 'w', format='NETCDF4_CLASSIC')

##Create coordinate data
###Time
time_units = 'days since 1986-01-01'
time_data = netCDF4.date2num(dates, time_units)
###Coordinate data
ysize = len(ds.lat.data)
latitudes = ds.lat.data
xsize = len(ds.lon.data)
longitudes = ds.lon.data

##Add Dimensions
latdim = ncfile.createDimension('lat', ysize)
londim = ncfile.createDimension('lon', xsize)
timedim = ncfile.createDimension('time')

##Add variables
###Coordinates
latvar = ncfile.createVariable('lat', 'f4', ('lat'))
latvar.standard_name = 'latitude'
latvar.long_name = 'latitude'
latvar.units = 'degrees_north'
latvar.axis = 'Y'
latvar[:] = latitudes
lonvar = ncfile.createVariable('lon', 'f4', ('lon'))
lonvar.standard_name = 'longitude'
lonvar.long_name = 'longitude'
lonvar.units = 'degrees_east'
lonvar.axis = 'X'
lonvar[:] = longitudes
###Time
timevar = ncfile.createVariable('time', 'f4', ('time',))
timevar.standard_name = 'time'
timevar.long_name = 'time'
timevar.calendar = 'proleptic_gregorian'
timevar.axis = 'T'
timevar.units = time_units
timevar[:] = time_data
###Now the data variable itself
metvar = ncfile.createVariable('Onset1', 'f4', ('time', 'lat', 'lon'), fill_value=-9999) #May change from f4 to d #fill_value=netCDF4.default_fillvals['f4']
metvar.standard_name = 'Onset of rainfall (year July to June)'
metvar.long_name = 'Onset of wet season is chosen as the day 10% of rainfall accumulates. Days with less than 5 mm of rainfall are omitted.'
metvar.units = 'Day of year from July 1st'
metvar[:] = onset_masked

metvar_1 = ncfile.createVariable('Cessation1', 'f4', ('time', 'lat', 'lon'), fill_value=-9999) #May change from f4 to d #fill_value=netCDF4.default_fillvals['f4']
metvar_1.standard_name = 'Cessation of rainfall (year July to June)'
metvar_1.long_name = 'Cessation is chosen as the day that 90% of annual rainfall accumulated. Days with less than 5 mm of rainfall are omitted.'
metvar_1.units = 'Day of year from July 1st'
metvar_1[:] = cessation_masked

# And we're done!
ncfile.close()


