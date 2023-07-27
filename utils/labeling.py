import json
import random

def generate_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

data = []

with open('상품명_labeling용.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip().split('|')
        number = line[0]
        name = line[1]
        item = {
            "name": name,
            # "id": int(number),
            # "color": generate_random_color(),
            # "type": "Rectangle",
            "attributes": []
        }
        data.append(item)

json_data = json.dumps(data, indent=2, ensure_ascii=False)
print(json_data)
