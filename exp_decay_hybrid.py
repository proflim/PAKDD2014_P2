import pandas as pd
import numpy as np
import math
from pandas.stats.moments import ewma
from scipy.optimize import curve_fit

## Parameters
pred_period = 19
num_of_months_used = 5

def get_year_month(year_month):
    ''' splits the year_month column to year and month and returns them in int format'''
    years = []
    months = []
    for y_m in year_month:
        y,m = y_m.split('/')
        years.append(int(y))
        months.append(int(m))
    return years,months


def get_zero_repairs(st_year,end_year):
    import itertools
    zero_df = []
    years = list(range(st_year,end_year +1))
    months = list(range(1,13))
    zero_df = [(y,m,0) for y,m in itertools.product(years,months)]
    zero_df = pd.DataFrame(zero_df)
    zero_df.columns = ['year','month','number_repair']
    return zero_df


def get_repair_complete(module,component):
    zero_repair = get_zero_repairs(repair_min_year,repair_max_year)
    repair_module_component = repair_train[np.logical_and(
                                repair_train.module_category == module,
                                repair_train.component_category == component)]

    cols_req = ['year_repair','month_repair','number_repair']
    cols_groupby = ['year_repair','month_repair']
    cols_req_final = ['year','month','number_repair']
    repair_train_summ = repair_module_component[cols_req].groupby(cols_groupby,as_index=False).sum()
    repair_train_summ.columns = ['year','month','number_repair']
    repair_merged = pd.merge(zero_repair,repair_train_summ,how='left',on=['year','month'])
    repair_merged['number_repair'] = repair_merged['number_repair_x'] + repair_merged['number_repair_y']

    return repair_merged[cols_req_final]

#Use simple moving average
def predict_EWMA(x,span=3,periods = pred_period):
    x_predict = np.zeros((span+periods,))
    x_predict[:span] = x[-span:]
    pred =  ewma(x_predict,span)[span:]

    pred = pred.round()
    pred[pred < 0] = 0
    return pred


# logarithmic function
def log_func(x, a, b):
  return a*np.log(x)+b

def pow_func(x, a, b):
    return a*np.power(x, -b)

def exp_func(x, a,b):
    return a * np.exp( -b*x)
    #return a*np.power(b, -x)

def predict_logDecay(y, span, periods = pred_period):
    y1 = y[-span:]
    x = pd.DataFrame( [i+1 for i in range(span)] )
    x = np.array(x)
    x = x.reshape(span,)
    y1 = np.array(y1)

    popt, pcov = curve_fit(log_func,x, y1, maxfev=5000)
    print('log: '+str(popt[0])+' '+str(popt[1]))
    pred = np.zeros((periods,))  #(19,)
    for i in range(periods):
        pred[i] = log_func((span+i), popt[0], (popt[1]))
    pred = pred.round()
    pred[pred < 0] = 0
    return pred

def predict_powerDecay(y, span, periods = pred_period):
        y1 = y[-span:]
        x = pd.DataFrame( [i+1 for i in range(span)] )
        x = np.array(x)
        x = x.reshape(span,)
        y1 = np.array(y1)

        popt, pcov = curve_fit(pow_func,x, y1, maxfev=5000)
        print('power: '+str(popt[0])+' '+str(popt[1]))

        pred = np.zeros((periods,))  #(19,)
        for i in range(periods):
            pred[i] = pow_func((span+i), popt[0], (popt[1]))
        pred = pred.round()
        pred[pred < 0] = 0
        return pred

# y = N0 * e ^(-lambda*t)
def predict_expDecay(y, span, periods = pred_period):
    y1 = y[-span:]
    x = pd.DataFrame( [i+1 for i in range(span)] )
    #x = pd.DataFrame([1,2,3,4,5,6])
    x = np.array(x)
    x = x.reshape(span,)
    y1 = np.array(y1)

    popt,pcov = curve_fit(exp_func, x, y1, maxfev=5000)
    #print(popt)

    pred = np.zeros((periods,))  #(19,)
    for i in range(periods):
        pred[i] = exp_func((span+i), popt[0], (abs(popt[1])))#############


    #pred = pred.round()
    pred[pred < 0] = 0
    return pred






################################################################################

repair_train = pd.read_csv('Data/RepairTrain.csv')
sale_train = pd.read_csv('Data/SaleTrain.csv')
output_target = pd.read_csv('Data/Output_TargetID_Mapping.csv')
#submission = pd.read_csv('Data/SampleSubmission.csv')
submission = pd.read_csv('Isha_Pred.csv')


# Process RepairTrain
#Separate year/month to two columns
repair_train['year_sale2'],repair_train['month_sale2'] = get_year_month(repair_train['year/month(sale)'])
repair_train['year_repair'],repair_train['month_repair'] = get_year_month(repair_train['year/month(repair)'])

repair_min_year = repair_train['year_repair'].min()
repair_max_year = repair_train['year_repair'].max()
sale2_min_year = repair_train['year_sale2'].min()
sale2_max_year = repair_train['year_sale2'].max()

cols_requ = ['module_category','component_category','year_repair','month_repair','number_repair']
cols_groupby = ['module_category','component_category','year_repair','month_repair']
repair_train_summ = repair_train[cols_requ].groupby(cols_groupby).sum()

# Process SaleTrain: SaleTrain Not used.
'''
sale_train['year_sale'],sale_train['month_sale'] = get_year_month(sale_train['year/month'])

sale1_min_year = sale_train['year_sale1'].min()
sale1_max_year = sale_train['year_sale1'].max()

cols_requ = ['module_category','component_category','year_sale1','month_sale1','number_sale']
cols_groupby = ['module_category','component_category','year_sale1','month_sale1']
sale_train_summ = sale_train[cols_requ].groupby(cols_groupby).sum()
'''


print('printing')
selectedModule = pd.DataFrame()

#predict for each module and category
for i in range(0,output_target.shape[0],pred_period):
    module = output_target['module_category'][i]
    component = output_target['component_category'][i]
    # missing periods are filled with 0
    Y = get_repair_complete(module,component).fillna(0)
    Y = Y['number_repair']
    checkEmpty_data = Y[-num_of_months_used:]
    count=0
    for j in range(num_of_months_used):
        if (checkEmpty_data.iloc[j]==0):
            count+=1
    # NOT EFFECTIVE
    # special case for outliers
    # Use 12months: M8P06, M6P06, M8P12, M9P06, M9P12, M6P12
    # Use 18 months: M4P06
    # Use separate function: M7P26(24months, log), M7P04(24months, log), M7P13(24months, power)
    #arr1 = ['M8P06', 'M6P06', 'M8P12', 'M9P06', 'M9P12', 'M6P12'] # 12months
    #arr2 = ['M4P06'] # 18 months
    #arr3 = ['M7P26', 'M7P04'] # 24 months log
    #arr4 = ['M7P13'] # 24 months power

    arr = [
    'M2P01',
    'M2P02',
    'M2P04',
    'M4P04',
    'M4P06',
    'M4P09',
    'M4P17',
    'M4P22',
    'M4P30',
    'M5P05',
    'M5P11',
    'M5P12',
    'M5P13',
    'M5P16',
    'M5P20',
    'M5P24',
    'M5P30',
    'M7P02',
    'M7P04',
    'M7P05',
    'M7P09',
    'M7P12',
    'M7P13',
    'M7P15',
    'M7P17',
    'M7P22',
    'M7P25',
    'M7P26',
    'M8P11',
    'M8P17',
    'M8P25',
    'M8P28',
    'M9P02',
    'M9P05',
    'M9P12',
    'M9P21',
    'M9P22']


    modulecomponent = str(module)+str(component)
    '''
    if modulecomponent in arr1:
        print("1: "+modulecomponent)
        pred = predict_expDecay(Y, 12)
    elif (modulecomponent in arr2):
        print("2: "+modulecomponent)
        pred = predict_expDecay(Y, 18)

    elif (modulecomponent in arr3):
        print("3: "+modulecomponent)
        pred = predict_expDecay(Y, 24)
    elif (modulecomponent in arr4):
        print("4: "+modulecomponent)
        pred = predict_expDecay(Y, 24)
    '''
    if modulecomponent in arr:
        if(count<4 & count>0):
            pred = predict_expDecay(Y, num_of_months_used)
            selectedModule[modulecomponent]= pred
            submission['target'][i:i+pred_period] = pred
        print(modulecomponent)
    #elif (count>0):
    #    pred = predict_EWMA(Y)
    #else:
    #    pred = predict_expDecay(Y, num_of_months_used)

    #print(modulecomponent +": "+ str(pred))

submission.to_csv('Hybrid_pred3.csv',index=False)
selectedModule.to_csv('selectedModule.csv',index=False)
print('submission file created')
print('done')
