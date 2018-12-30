import time

url = "http://www.google.com"
file_name = "csv"

'''
for i in range(5):
    print("writing {} to {}.".format(url, file_name), end='', flush=True)
    time.sleep(1)
    print("\rwriten {} to {}.".format(url, file_name))
    time.sleep(1)
    # print("\r{}".format('='*i) + ">", end='')
'''

print("\rInserted {}...{}.".format(url[:5], url[-5:]))
