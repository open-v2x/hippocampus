import requests
import json

def post(url):
    data = ""
    if url == "":
        return data
    try:
        res = requests.get(url).json()
        if isinstance(res,dict):
            if res.get("status","")==200:
                return res.get("data","")
            else:
                return data
        else:
            return data
    except Exception as error:
        print(error)
        return data

if __name__=="__main__":
    url="http://localhost:8090/control/get?room=test"
    data=post(url)
    print("rtmp://localhost:1935/live/" + data)