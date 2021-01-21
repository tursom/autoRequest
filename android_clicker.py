import asyncio
import sys

import cv2
from ppadb.client import Client as AdbClient
from datetime import datetime
import ppadb.device
import yaml
import numpy as np

__global_count = 0


async def delay(action):
    delay = 0.0
    if "second" in action:
        delay += float(action["second"])
    if "seconds" in action:
        delay += float(action["seconds"])
    if "minute" in action:
        delay += float(action["minute"]) * 60
    if "minutes" in action:
        delay += float(action["minutes"]) * 60
    if "hour" in action:
        delay += float(action["hour"]) * 60 * 60
    if "hours" in action:
        delay += float(action["hours"]) * 60 * 60
    if "day" in action:
        delay += float(action["day"]) * 60 * 60 * 24
    if "days" in action:
        delay += float(action["days"]) * 60 * 60 * 24
    print(f"delay {delay} seconds")
    await asyncio.sleep(delay)
    return True


def click(action, device: ppadb.device.Device, config):
    point_str = str(action["point"]).split(',')
    if len(point_str) < 2:
        return False
    width = int(point_str[0])
    height = int(point_str[1])
    if hasattr(config, "width"):
        width = int(width * float(config.width))
    if hasattr(config, "height"):
        height = int(height * float(config.height))
    point = (width, height)
    print(f"click point {point}")
    device.shell(f"input tap {point[0]} {point[1]}")
    return True


def screen_cap(action, device: ppadb.device.Device, time=None):
    name = "screen_cap.png"
    if "name" in action:
        name = str(action["name"])
        if "/d" in name:
            time_str = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
            name = name.replace("/d", time_str)
        if "/c" in name:
            global __global_count
            __global_count += 1
            name = name.replace("/c", str(__global_count))

    screen_cap = device.screencap()
    screen_file = open(name, "wb")
    screen_file.write(screen_cap)
    screen_file.close()
    print(f"screen cut write to {name}")
    return True


async def do_actions(actions, device: ppadb.device.Device, config):
    for action in actions:
        action_type = str(action["action"]).lower()
        success = False
        if action_type == "delay":
            success = await delay(action)
        elif action_type == "click":
            success = click(action, device, config)
        elif action_type == "screen":
            success = screen_cap(action, device)
        if not success:
            print(f"action {action_type} failed")


async def loop(motion, device: ppadb.device.Device, config):
    if "actions" not in motion:
        return
    delay = 0.0
    times = -1
    if "delay" in motion:
        delay = float(motion["delay"])
    if "timeunit" in motion:
        timeunit = motion["timeunit"]
        if timeunit == "seconds" or timeunit == "second":
            delay = delay * 60
        elif timeunit == "minutes" or timeunit == "minute":
            delay = delay * 60
        elif timeunit == "hours" or timeunit == "hour":
            delay = delay * 60 * 60
        elif timeunit == "days" or timeunit == "day":
            delay = delay * 60 * 60 * 24
        else:
            print("unknown timeunit %s" % (timeunit,))
    if "times" in motion:
        times = int(motion["times"])
    elif "time" in motion:
        times = int(motion["time"])

    if times >= 0:
        times -= 1
        while times > 0:
            times -= 1
            await asyncio.sleep(delay)
            if not do_actions(motion["actions"], device, config):
                times += 1
    else:
        while True:
            await asyncio.sleep(delay)
            await do_actions(motion["actions"], device, config)


class Object:
    def __init(self):
        return


async def main(argv):
    tasks = []
    config = Object()
    config.width = 1
    config.height = 1
    client = AdbClient(host="127.0.0.1", port=5037)
    devices = client.devices()
    if len(devices) == 0:
        print("no device detected!")
        exit(1)
        return
    elif len(devices) == 1:
        device = devices[0]
    else:
        # print("multiple device detected, chose one")
        index = 0
        for device in devices:
            print(f"{index}: {device.serial}")
        index = int(input("multiple device detected, chose one (index number): "))
        device = devices[index]
    config.device = device

    config_text = open("android_clicker.yaml").read()
    config_data = yaml.safe_load(config_text)
    print(config_data)

    if "screen" in config_data:
        screen = config_data["screen"]
        screen_cap = device.screencap()
        img = cv2.imdecode(np.frombuffer(screen_cap, np.uint8), cv2.IMREAD_COLOR)
        sp = img.shape
        height = sp[0]
        width = sp[1]
        print(f'screen width: {width:d}, height: {height:d}')
        if "width" in screen:
            config.width = width / int(screen["width"])
        if "height" in screen:
            config.height = height / int(screen["height"])

    available_motions = argv[1:]
    if "motions" in config_data:
        motions = config_data["motions"]
        if len(available_motions) != 0:
            motions = filter(lambda m: "name" in m and m["name"] in available_motions, motions)
        for motion in motions:
            task = asyncio.create_task(loop(motion, device, config))
            tasks.append(task)
    for task in tasks:
        await task


if __name__ == '__main__':
    asyncio.run(main(sys.argv))
