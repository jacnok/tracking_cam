#! /Library/Frameworks/Python.framework/Versions/Current/bin/python3
import argparse
import requests
import json
url = 'http://localhost:5000/callpreset'

# Data to be sent (as a dictionary)
data = {'preset': 0}
# response = requests.post(url, json=data)
# Send a POST request
def call_preset(preset):
    data = {'preset': preset}
    response = requests.post(url, json=data)
    print(response.text)
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Run specific methods from the command line.')
#     parser.add_argument('-method', help='Method to run', required=True)
#     parser.add_argument("-preset_num", type=int, help=f"preset number 1-255")
#     args = parser.parse_args()

#     if args.method == 'call_preset':
#         call_preset(args.preset_num)
call_preset(0)