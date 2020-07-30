# import all needed libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# read csv data
data = pd.read_csv("time_series_covid19_confirmed_global.csv", index_col=0)
# get the countries' name
countries = data["Country/Region"]
# extract the starting date up to the last date
x = list(data.columns)
x = x[3:]
## percentage growth analysis ##
# get the last four week dates
xt = [x[-i] for i in range(1, 30, 7)]
# extract the data records for the associated week dates
yt, yt1, yt2, yt3, yt4 = ([] for i in range(5))
for i in range(len(countries)):
    yt.append(data.iloc[i][xt[0]])
    yt1.append(data.iloc[i][xt[1]])
    yt2.append(data.iloc[i][xt[2]])
    yt3.append(data.iloc[i][xt[3]])
    yt4.append(data.iloc[i][xt[4]])
# calculate the growth percentage based on four last weeks data
dt, dt_1w, dt_2w, dt_3w, dt_4w = ([] for i in range(5))
for i in range(len(countries)):
    dt_1w.append((yt[i] - yt1[i]) / yt1[i] if yt1[i] else 1)
    dt_2w.append((yt1[i] - yt2[i]) / yt2[i] if yt2[i] else 1)
    dt_3w.append((yt2[i] - yt3[i]) / yt3[i] if yt3[i] else 1)
    dt_4w.append((yt3[i] - yt4[i]) / yt4[i] if yt4[i] else 1)
    dt.append((dt_1w[i] + dt_2w[i] + dt_3w[i] + dt_4w[i]) / 4)
# save the growth percentage in a new data column 'Percentage'
data["Percentage"] = dt
# get the rank of each country and save it in a new data column 'Rank'
data["Rank"] = data["Percentage"].rank(ascending=False)
# select top ten rank and save it under new data column 'Selected'
data["Selected"] = data["Rank"] <= 10
# plot top ten countries with the highest growth percentage within a month
for i in range(len(countries)):
    if data["Selected"][i]:
        y = data.iloc[i,3:-3]
        if np.isnan(data.index[i]):
            lbl = str(data.iloc[i,0])
        else:
            lbl = str(data.iloc[i,0]) + " - " + str(data.index[i])
        plt.plot(x, y, label=lbl)
ind = [i for i in range(0, len(x), 7)]
date = [x[i] for i in ind]
plt.xticks(ind, date, rotation=60)
plt.xlabel("Time")
plt.ylabel("Number")
plt.title("COVID-19 Confirmed - Growth Percentage")
plt.legend()
plt.savefig('10 Growth Percentage.jpg', bbox_inches = 'tight')
plt.figure(figsize=(20,10))
## up-to-date confirmed cases analysis ##
# get the rank of each country based on the number of last date confirmed cases,
# save it in a new data column 'Rank_Global'
# select top ten rank and save it under new data column 'Selected_Global'
data["Rank_Global"] = data[x[-1]].rank(ascending=False)
data["Selected_Global"] = data["Rank_Global"] <= 10
# plot top ten countries based on the number of last date confirmed cases
for i in range(len(countries)):
    if data["Selected_Global"][i]:
        y = data.iloc[i,3:-5]
        if np.isnan(data.index[i]):
            lbl = str(data.iloc[i,0])
        else:
            lbl = str(data.iloc[i,0]) + " - " + str(data.index[i])
        plt.plot(x, y, label=lbl)
ind = [i for i in range(0, len(x), 7)]
date = [x[i] for i in ind]
plt.xticks(ind, date, rotation=60)
plt.xlabel("Time")
plt.ylabel("Number")
plt.title("COVID-19 Confirmed - Global")
plt.legend()
plt.savefig('10 Confirmed Global.jpg', bbox_inches = 'tight')
plt.figure(figsize=(20,10))
def trendCalc(src, seasonLength, section=1):
    trendNumerator = 0
    trendDenominator = 0
    for i in range(seasonLength):
        trendNumerator += (section * seasonLength - i) * src[-(i+1)]
        trendDenominator += section * seasonLength - i
    return trendNumerator/trendDenominator
  def allCalc(data, seaInd, tSmooth, oSmooth, alpha, beta, gamma):
    lInd = len(seaInd)
    lData = len(data)
    predicted = [None] * (len(data)+1)
    sIndices = [None] * len(data)
    for i in range(lInd, lData):
        if i < 2*lInd:
            oSmoothing = alpha * (data[i] - seaInd[i-lInd]) + ((1 - alpha) * (tSmooth + oSmooth))
            tSmoothing = gamma * (oSmoothing - oSmooth) + ((1 - gamma) * tSmooth)
            sIndices[i] = beta * (data[i] - oSmooth - tSmooth) + ((1 - beta) * seaInd[i-lInd])
            if i < 2*lInd - 1:
                predicted[i+1] = oSmoothing + tSmoothing + seaInd[i - lInd + 1]
            else:
                predicted[i+1] = oSmoothing + tSmoothing + sIndices[i - lInd + 1]
            oSmooth = oSmoothing
            tSmooth = tSmoothing
        else:
            oSmoothing = alpha * (data[i] - sIndices[i - lInd]) + ((1 - alpha) * (tSmooth + oSmooth))
            tSmoothing = gamma * (oSmoothing - oSmooth) + ((1 - gamma) * tSmooth)
            sIndices[i] = beta * (data[i] - oSmooth - tSmooth) + ((1 - beta) * sIndices[i - lInd])
            predicted[i+1] = oSmoothing + tSmoothing + sIndices[i - lInd + 1]
            oSmooth = oSmoothing
            tSmooth = tSmoothing
    return predicted
def err(act, pred, seasonLength=1):
    diff = [None] * len(act)
    error = [None] * len(act)
    mae = [None] * len(act)
    # change 0 values to 1 in act data to avoid inf
    for i in range(len(act)):
        if act[i] == 0:
            act[i] = 1
    for i in range(seasonLength + 1, len(act)):
        diff[i] = act[i] - pred[i]
        error[i] = abs(diff[i] / act[i])
        mae[i] = abs(act[i] - act[i - seasonLength])
    mae[seasonLength] = abs(act[seasonLength] - act[0])
    # MASE calculation
    sumMae = 0
    for e in mae:
        if e != None:
            sumMae += e
    Q = sumMae / (len(act) - seasonLength)
    errMase = [None] * len(act)
    sumMase = 0
    n = 0
    for i in range(seasonLength + 1, len(act)):
        errMase = abs((act[i] - pred[i]) / Q)
        sumMase += errMase
        n += 1
    mase = sumMase / n
    # MAPE calculation
    sumErr = 0
    for e in error:
        if e != None:
            sumErr += e
    mape = (sumErr / n) * 100
    return mape, mase
def mod_HW_Additive(src, seasonLength, span):
    dLength = len(src)
    seasonNo = dLength // seasonLength
    dataUsed = src[-(seasonNo * seasonLength):]
    # init trend calculation
    t1 = trendCalc(dataUsed[seasonLength:(2*seasonLength)], seasonLength, 2)
    t2 = trendCalc(dataUsed[0:seasonLength], seasonLength)
    tSmooth = (1 / (seasonLength**2)) * (t1 - t2)
    # seasons' level average
    season = []
    for i in range(seasonNo):
        season.append(np.mean(dataUsed[(i*seasonLength):(((i+1)*seasonLength))]))
   # seasonal data averaged
    aveSData = []
    for i in range(seasonNo):
        aveSData += [data / season[i] for data in dataUsed[(i*seasonLength):(((i+1)*seasonLength))]]
    aveSData = np.array(aveSData)
    aveSData[np.isnan(aveSData)] = 0
    # basic seasonal indices
    seaDataInd = []
    for i in range(seasonLength):
        cumData = 0
        for j in range(seasonNo):
            cumData += aveSData[i + (j * seasonLength)]
        seaDataInd.append(cumData / seasonNo)
    # init overall smoothing
    oSmooth = 0
    oDenominator = 0
    for i in range(span):
        oSmooth += (seasonLength - i) * dataUsed[seasonLength - i - 1]
        oDenominator += (seasonLength - i)
    oSmooth /= oDenominator
    # smooths and trends calculation
    initMape = 100
    it = 50
    finPred = []
    alpha = 0
    beta = 0
    gamma = 0
    mase = 0
    for a in range(it):
        for b in range(it):
            for c in range(it):
                pred = allCalc(dataUsed, seaDataInd, tSmooth, oSmooth, a/it, b/it, c/it)
                # error measurement
                mape, mase = err(dataUsed, pred, seasonLength)
                if mape < initMape:
                    finPred = pred
                    initMape = mape
                    mase = mase
                    alpha = a/it
                    beta = b/it
                    gamma = c/it
    return finPred, initMape, mase, alpha, beta, gamma
  # read csv data for each country
data = pd.read_csv("US_train.csv", index_col=0)
# get the countries' name
countries = data["Country/Region"]
# extract the starting date up to the last date
x = list(data.columns)
x = x[3:]
len(x)
# import all needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# np.seterr(divide='ignore', invalid='ignore')
# initial params
seasonLength = 7
span = 4
# check the data length
xn = x[-((len(x) // seasonLength) * seasonLength):]
# choose the data to be predicted - based on Growth Percentage
country = data["Country/Region"] == "US"
d = data[country].index.isnull()
for i in range(len(d)):
    if d[i]:
#         for the previous dataset
#         procData = list(data[country].iloc[i,3+(len(x)-len(xn)):-5])
        procData = list(data[country].iloc[i,3+(len(x)-len(xn)):])
pred, mape, mase, alp, bet, gam = mod_HW_Additive(procData, seasonLength, span)
print("Prediction Plot " + str(len(pred)) + " - "+str(type(pred)))
print(*pred)
plt.plot(xn, procData, label="Actual")
plt.plot(xn, pred[:-1], label="Prediction")
ind = [i for i in range(0, len(xn), 7)]
date = [xn[i] for i in ind]
plt.xticks(ind, date, rotation=60)
plt.xlabel("Time")
plt.ylabel("Number")
plt.title("COVID-19 Prediction - Train Phase - US")
plt.legend()
plt.savefig('US.jpg', bbox_inches = 'tight')
plt.figure(figsize=(20,10))
print("MAPE: " + str(mape) + "\nMASE: "+ str(mase) + "\nAlpha: " + str(alp) + "\nBeta: " + str(bet) + "\nGamma: " + str(gam))
print("Future Prediction: " + str(pred[-1]))
pct = (pred[-1] - procData[-1]) / procData[-1]
print("Percentage: " + str(pct * 100))
def mod_HW_Additive_test(src, seasonLength, span, alpha=0, beta=0, gamma=0):
    dLength = len(src)
    seasonNo = dLength // seasonLength
    dataUsed = src[-(seasonNo * seasonLength):]
    # init trend calculation
    t1 = trendCalc(dataUsed[seasonLength:(2*seasonLength)], seasonLength, 2)
    t2 = trendCalc(dataUsed[0:seasonLength], seasonLength)
    tSmooth = (1 / (seasonLength**2)) * (t1 - t2)
    # seasons' level average
    season = []
    for i in range(seasonNo):
        season.append(np.mean(dataUsed[(i*seasonLength):(((i+1)*seasonLength))]))
   # seasonal data averaged
    aveSData = []
    for i in range(seasonNo):
        aveSData += [data / season[i] for data in dataUsed[(i*seasonLength):(((i+1)*seasonLength))]]
    aveSData = np.array(aveSData)
    aveSData[np.isnan(aveSData)] = 0
    # basic seasonal indices
    seaDataInd = []
    for i in range(seasonLength):
        cumData = 0
        for j in range(seasonNo):
            cumData += aveSData[i + (j * seasonLength)]
        seaDataInd.append(cumData / seasonNo)
    # init overall smoothing
    oSmooth = 0
    oDenominator = 0
    for i in range(span):
        oSmooth += (seasonLength - i) * dataUsed[seasonLength - i - 1]
        oDenominator += (seasonLength - i)
    oSmooth /= oDenominator
    # smooths and trends calculation
    pred = allCalc(dataUsed, seaDataInd, tSmooth, oSmooth, alpha, beta, gamma)
    # error measurement
    mape, mase = err(dataUsed, pred, seasonLength)
    return pred, mape, mase, alpha, beta, gamma
  # read csv data for each country
data = pd.read_csv("Suriname_test.csv", index_col=0)
# get the countries' name
countries = data["Country/Region"]
# extract the starting date up to the last date
x = list(data.columns)
x = x[3:]
len(x)
# initial params
seasonLength = 7
span = 4
# check the data length
xn = x[-((len(x) // seasonLength) * seasonLength):]
# choose the data to be predicted - based on Growth Percentage
country = data["Country/Region"] == "Suriname"
d = data[country].index.isnull()
for i in range(len(d)):
    if d[i]:
#         for the previous dataset
#         procData = list(data[country].iloc[i,3+(len(x)-len(xn)):-5])
        procData = list(data[country].iloc[i,3+(len(x)-len(xn)):])
pred, mape, mase, alp, bet, gam = mod_HW_Additive_test(procData, seasonLength, span, 0.44, 0.56, 0.00)
print("Prediction Plot " + str(len(pred)) + " - "+str(type(pred)))
print(*pred)
plt.plot(xn, procData, label="Actual")
plt.plot(xn, pred[:-1], label="Prediction")
ind = [i for i in range(0, len(xn), 7)]
date = [xn[i] for i in ind]
plt.xticks(ind, date, rotation=60)
plt.xlabel("Time")
plt.ylabel("Number")
plt.title("COVID-19 Prediction - Test Phase - Suriname")
plt.legend()
plt.savefig('Suriname_test.jpg', bbox_inches = 'tight')
plt.figure(figsize=(20,10))
print("MAPE: " + str(mape) + "\nMASE: "+ str(mase) + "\nAlpha: " + str(alp) + "\nBeta: " + str(bet) + "\nGamma: " + str(gam))
print("Future Prediction: " + str(pred[-1]))
pct = (pred[-1] - procData[-1]) / procData[-1]
print("Percentage: " + str(pct * 100))


