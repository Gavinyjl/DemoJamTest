import math


if __name__ == "__main__":
    inputs = []
    while ((ui := input('plz input sth:')) != '='):
        inputs.append(int(ui))
    print('sum=', sum(inputs))
