import asyncio
import json
import sys
import time

import pymongo
import requests

__mongo_client = pymongo.MongoClient("mongodb://192.168.0.61:27017/")
__mongo_db = __mongo_client["auto_request"]
__mongo_collection = __mongo_db["record"]


def request(method, url, **kwargs):
    return requests.request(method, url, **kwargs)


def request_and_record(method, url, **kwargs):
    try:
        response = request(method, url, **kwargs).content.decode()
        print("================================================")
        print(f"{time.asctime(time.localtime(time.time()))} {method} {url}")
        print(response)
        if __mongo_collection is not None:
            __mongo_collection.insert_one({
                "time": int(time.time() * 1000),
                "method": method,
                "url": url,
                "kwargs": kwargs,
                "response": response
            })
        return True
    except requests.exceptions.ConnectionError:
        return False


async def loop(loop_config, method, url, **kwargs):
    delay = float(loop_config["delay"])
    times = -1
    if "timeunit" in loop_config:
        timeunit = loop_config["timeunit"]
        if timeunit == "minutes":
            delay = delay * 60
        elif timeunit == "hours":
            delay = delay * 60 * 60
        elif timeunit == "days":
            delay = delay * 60 * 60 * 24
        else:
            print("unknown timeunit %s" % (timeunit,))
    if "times" in loop_config:
        times = int(loop_config["times"])
    if times >= 0:
        times -= 1
        while times > 0:
            times -= 1
            await asyncio.sleep(delay)
            if not request_and_record(method, url, **kwargs):
                times += 1
    else:
        while True:
            await asyncio.sleep(delay)
            request_and_record(method, url, **kwargs)


async def main():
    handle_file = []
    tasks = []

    for arg in sys.argv[1:]:
        handle_file.append(arg)

    for file_name in handle_file:
        file_open = open(file_name, "rb")
        text = file_open.read().decode()
        config_list = json.loads(text)
        for config in config_list:
            url = config["url"]
            method = "get"
            headers = {}
            params = {}
            if "method" in config:
                method = config["method"]
            if "headers" in config:
                headers = config["headers"]
            if "params" in config:
                params = config["params"]
            request_and_record(method, url, headers=headers, params=params)
            if "loop" in config:
                task = asyncio.create_task(loop(config["loop"], method, url, headers=headers, params=params))
                tasks.append(task)
    for task in tasks:
        await task


if __name__ == '__main__':
    asyncio.run(main())
