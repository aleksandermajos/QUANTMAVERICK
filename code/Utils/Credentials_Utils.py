def exampleAuth(path):
    accountID, token = None, None
    with open(path+"account.txt") as I:
        accountID = I.read().strip()
    with open(path+"token.txt") as I:
        token = I.read().strip()
    return accountID, token