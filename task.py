import yaml
import json

with open("task.yaml") as task_file:
    task = yaml.safe_load(task_file.read())
print(json.dumps(task, indent=2))
