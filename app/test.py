from fastai.text.all import *

learn = load_learner("review.pkl")
print(learn.predict("I love this product!"))