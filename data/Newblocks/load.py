import pickle as pkl

f = open('newblocks.pkl','rb')
with open('newblocks.pkl',"rb") as f :
    A = 0
    while True:
        try:
            pkl.load(f)
        except:
            break