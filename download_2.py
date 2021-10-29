import os

import bs4
import requests

try:
    os.mkdir("毛泽东选集 第五卷")
except FileExistsError:
    pass
index = requests.get("http://www.quanxue.cn/LS_Mao/XuanJiEIndex.html").content
soup = bs4.BeautifulSoup(index, "html.parser", from_encoding="utf-8")
for i in soup.find_all("td", {"class": "index_left_td"}):
    url = f"http://www.quanxue.cn/LS_Mao/{i.findChild('a').attrs['href']}"
    name = i.text
    s = bs4.BeautifulSoup(requests.get(url).content, "html.parser", from_encoding="utf-8")
    with open(f"毛泽东选集 第五卷/{name}.txt", "wb") as out:
        for line in s.find_all("p"):
            text = line.text.replace('\r', '').replace('\n', '')
            print(text)
            # break
            out.write("　　".encode())
            out.write(text.encode())
            out.write("\n".encode())
        # break
