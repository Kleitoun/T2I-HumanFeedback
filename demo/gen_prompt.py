import numpy as np
import argparse

colors = [
    "red",
    "yellow",
    "green",
    "blue",
    "black",
    "pink",
    "purple",
    "white",
    "brown",
    "",
]
colors[:-1] = list(map(lambda x: x + " colored", colors[:-1]))
numbers = [
    "one", 
    "two", 
    "three", 
    "four", 
    "five", 
    "six",
]
backgrounds = [
    "in the forest",
    "in the city",
    "on the moon",
    "in the field",
    "in the sea",
    "on the table",
    "under the table",
    "in the desert",
    "in San Francisco",
    "",
]
objects = [
    "dog",
    "cat",
    "lion",
    "orange",
    "vase",
    "cup",
    "apple",
    "chair",
    "bird",
    "cake",
    "bicycle",
    "tree",
    "donut",
    "box",
    "plate",
    "clock",
    "backpack",
    "car",
    "airplane",
    "bear",
    "horse",
    "tiger",
    "rabbit",
    "rose",
    "wolf",
]
objects_pl = [
    "dogs",
    "cats",
    "lions",
    "oranges",
    "vases",
    "cups",
    "apples",
    "chairs",
    "birds",
    "cakes",
    "bicycles",
    "trees",
    "donuts",
    "boxes",
    "plates",
    "clocks",
    "backpacks",
    "cars",
    "airplanes",
    "bears",
    "horses",
    "tigers",
    "rabbits",
    "roses",
    "wolves",
]


def gen_prompt(n_prompts):
    prompts = set()
    while len(prompts) < n_prompts:
        prompt = gen_prompt_once()
        prompts.add(prompt)
    return prompts


def get_perturbated_prompts(z, N):
    output = []
    count = 0
    while count < N:
        text = gen_prompt_once()
        if z != text:
            count += 1
            output.append(text)
    return output


def gen_prompt_once():
    number = np.random.choice(numbers, p=[0.3, 0.14, 0.14, 0.14, 0.14, 0.14])
    # number = np.random.choice(numbers,p=[0.4,0.2,0.2,0.2])
    object = (
        np.random.choice(objects) if number == "one" else np.random.choice(objects_pl)
    )
    color = np.random.choice(
        colors, p=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.55]
    )
    # color = np.random.choice(colors, p = [0.15,0.15,0.15,0.15,0.4])

    background = np.random.choice(
        backgrounds, p=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.55]
    )
    # background = np.random.choice(backgrounds,p=[0.14,0.14,0.14,0.14,0.14,0.3])

    prompt = " ".join([item for item in [number, color, object, background] if item])

    return prompt


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0)
parser.add_argument("--n_prompts", default=9000)
parser.add_argument("--fname", default="gen_prompts.txt")

if __name__ == "__main__":
    args = parser.parse_args()
    np.random.seed(args.seed)
    n_prompts = args.n_prompts
    prompts = list(gen_prompt(n_prompts))
    with open(args.fname, "w") as fp:
        fp.write("\n".join(prompts))
