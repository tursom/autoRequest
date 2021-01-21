import json
import time

import pymongo

__mongo_client = pymongo.MongoClient("mongodb://192.168.0.61:27017/")
__mongo_db = __mongo_client["auto_request"]
__mongo_collection = __mongo_db["record"]

out_file = "data.txt"
url = "http://192.168.0.61:8088/rpc/user/app/v1/2468510352548115453"

# out_file = "group.txt"
# url = "http://192.168.0.61:8089/conversation/group/2498656631361897469/1/50"

size = 0
out = open(out_file, "wb")
find_document = __mongo_collection.find({
    "url": url
})
for e in find_document:
    size += 1
    out.write(f"{size} ================================================\n".encode())
    out.write(f"{time.asctime(time.localtime(e['time'] / 1000))} {e['method']} {e['url']}\n".encode())
    out.write(json.dumps(json.loads(e["response"]), indent=4, sort_keys=True, ensure_ascii=False).encode())
    out.write(b"\n\n")
    print(f"{size} ================================================")
    print(f"{time.asctime(time.localtime(e['time'] / 1000))} {e['method']} {e['url']}")
    print(json.dumps(json.loads(e["response"]), indent=4, sort_keys=True, ensure_ascii=False))
# print(time.asctime(time.localtime(int(time.time() * 1000) / 1000)))

# if __name__ == '__main__':
#     unittest.main()
