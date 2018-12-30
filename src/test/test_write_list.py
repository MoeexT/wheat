import csv

list = [
    ['list', 1],
    ['list', 2],
    [{'1': 2}, (1,2)]
]

# with open("write_list.txt", 'w') as f:
#     f.writelines(str(list))
with open("write_list.csv", 'w', newline='') as f:
    w = csv.writer(f)
    w.writerows(list)
