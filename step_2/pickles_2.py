import pickle

class GordonsCake:
    def __init__(self):
        self.flour = 100
        self.sugar = 200
        self.eggs = 2
        self.butter = 50
        self.milk = 100
        self.baking_powder = 1

    def bake(self):
        print("Baking a cake...")


# Create an instance of the class
cake = GordonsCake()

# Pickle the object
with open('cake.pkl', 'wb+') as f:
    pickle.dump(cake, f)

# Load the pickled object
with open('cake.pkl', 'rb') as f:
    cake = pickle.load(f)

cake.bake()
