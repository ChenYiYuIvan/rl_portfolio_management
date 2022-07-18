import csv

def get_date_list():

    date_list = []
    with open("data/AAPL_data.csv", 'r') as f:
        data = csv.reader(f)
        header = next(data)
        for row in data:
            date_list.append(row[0])

    return date_list