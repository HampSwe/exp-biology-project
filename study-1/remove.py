import os

def save_values_to_file(file_name, x_values, y_values):
    f = open(file_name, "w")

    for line in x_values:
        f.write(str(line) + "\n")
    
    f.write("NEXT_SERIES\n")

    for line in y_values:
        f.write(str(line) + "\n")

    f.write("END\n")

    f.close()


def read_file(file_name):
    x_values = []
    y_values = []

    reading_x = True

    f = open(file_name)
    content = f.readlines()

    for row in content:
        if reading_x:
            if row == "NEXT_SERIES\n":
                reading_x = False
            else:
                x_values.append(float(row[:-1]))
        else:
            if row == "END\n":
                break
            else:
                y_values.append(float(row[:-1]))

    f.close()

    return (x_values, y_values)


def main():

    files = os.listdir()

    for f in files:
        if not f == "remove.py":
            x, y = read_file(f)
            del y[0]
            save_values_to_file(f, x, y)

if __name__ == "__main__":
    main()