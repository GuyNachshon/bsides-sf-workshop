import pickle

cake_ingridients = {
    'flour': 100,
    'sugar': 200,
    'eggs': 2,
    'butter': 50,
    'milk': 100,
    'baking_powder': 1,
}

with open('cake_recipe.pkl', 'wb+') as f:
    pickle.dump(cake_ingridients, f)

# Load the pickled data
with open('cake_recipe.pkl', 'rb') as f:
    cake_recipe = pickle.load(f)

print(cake_recipe)