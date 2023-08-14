from sklearn.preprocessing import LabelEncoder

TOKEN_GAP = "-"
TOKENS_AA = list("ARNDCEQGHILKMFPSTWYV")
TOKENS_AHO = sorted([TOKEN_GAP, *TOKENS_AA])

ALPHABET_AHO = LabelEncoder().fit(TOKENS_AHO)
