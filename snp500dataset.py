import torch
import pandas as pd


class sp500Dataset:
    def __init__(self):
        self.data = pd.read_csv('SP500.csv')

    def getAllData(self):
        return self.data

     # normalize each sample separately
    def getDataForModel(self, batch_size, time=1007, show_data=False):
        unnormalize_data = self.data.copy()
        # convert to pandas date object
        self.data['date'] = pd.to_datetime(self.data.date)

        # sort by companies name and date
        sortedCompaniesDict = dict(tuple(self.data.sort_values(['symbol', 'date']).groupby('symbol')))
        sortedCompaniesList = list(sortedCompaniesDict.values())

        # filter companies with less then 1007 days of data
        filteredCompanies = list(filter(lambda company: company.shape[0] == 1007, sortedCompaniesList))

        # keep only high
        filteredCompanies = list(
            map(lambda company: torch.from_numpy(company.to_numpy()[:, 3].astype(float))[:time], filteredCompanies))

        normalized = list(
            map(lambda company:  self.norm_company(company), filteredCompanies))

        data_tesor = torch.stack(filteredCompanies)

        if show_data:
            self.show_amazon_and_google(unnormalize_data)

        m = len(data_tesor)
        train = data_tesor[:(int)(3 / 4 * m)].float()
        validation = data_tesor[(int)(3 / 4 * m):].float()

        # train, validation = train_test_split(data_tesor, shuffle=True, test_size=0.25)

        batchs = torch.split(train, batch_size, dim=0)
        # batchs = torch.stack(list(batchs), dim=0)
        return batchs, validation

    def norm_company(self, company):
        # normalize data
        company -= company.min()
        company /= company.max()
