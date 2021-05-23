import os
import numpy as np

def save_values_to_file(file_name, x_values, y_values):
    f = open(file_name, "w")

    for line in x_values:
        f.write(str(line) + "\n")
    
    f.write("NEXT_SERIES\n")

    for line in y_values:
        f.write(str(line) + "\n")

    f.write("END\n")

    f.close()


def format_file(file_name, to_folder):

    f = open(file_name)

    x_start_string = "      <ColumnCells>\n"
    x_end_string = "</ColumnCells>\n"
    start_of_file = "<Document>\n"

    content = f.readlines()

    if content[0] != start_of_file:
        print("Incorrect file format!")
        return

    x_values = []
    y_values = []

    found_x = False
    do_y = False

    for row in content:
        if not do_y:
            if found_x:
                if row == x_end_string:
                    found_x = False
                    do_y = True
                else:
                    x_values.append(float(row))
            elif row == x_start_string:
                found_x = True
        else:
            if found_x:
                if row == x_end_string:
                    break
                else:
                    y_values.append(float(row))
            elif row == x_start_string:
                found_x = True  

    # print(x_values)
    # print()
    # print(y_values)

    if len(x_values) != len(y_values) or len(x_values) != 655:
        print()
        print("LENGTH OF ARRAYS NOT THE SAME!")
        print(len(x_values))
        print(len(y_values))
        print()
    
    f.close()

    this_path = os.getcwd()
    os.chdir(to_folder)

    new_name = file_name[:-5] + ".txt"

    save_values_to_file(new_name, x_values, y_values)
    os.chdir(this_path)


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

def copy_to_other_folder_and_change_name(folder_path, text):
    this_path = os.getcwd()

    files = os.listdir()

    for f in files:
        if f != "processing.py":
            this_file = open(f)
            content = this_file.readlines()
            this_file.close()

            os.chdir(folder_path)
            g = open(text + "_" + f[2:], "w")
            g.writelines(content)
            g.close()
            os.chdir(this_path)

    
def format_all_files(to_folder):
    files = os.listdir()

    for f in files:
        if f != "processing.py":
            format_file(f, to_folder)

#du antar att filerna kommer i inbördes rätt ordning... y-värdet spelar egenltigen ingen roll, men kan vara bra att checka med print
def k_difference(to_folder):
    current_path = os.getcwd()

    files = os.listdir()
    groups = []
    for i in range(int((len(files) - 5) / 5)):
        groups.append([])

    ks = []

    for f in files:
        if f != "processing.py":
            if "k" in f:
                ks.append(f)
            else:
                groups[int(f.split("_")[1]) - 1].append(f)
    
    print()
    for i in groups:
        print(i)
    print(ks)
    print()

    x_values = read_file(groups[0][0])[0]
    y_values = []

    for serie in groups:
        y_values.append(list(map(lambda x: read_file(x)[1], serie)))

    k_y = list(map(lambda x: read_file(x)[1], ks))

    for i in range(len(groups)):
        file_names = groups[i]
        values = y_values[i]

        for p in range(len(file_names)):
            these_y_values = values[p]
            these_k_values = k_y[p]
            name = file_names[p]

            new_values = []
            new_name = "d_" + name

            for q in range(len(these_y_values)):
                new_values.append(these_k_values[q] - these_y_values[q])
            
            os.chdir(to_folder)
            save_values_to_file(new_name, x_values, new_values)
            os.chdir(current_path)


def calculate_integral(interval, file_name):
    x_values, y_values = read_file(file_name)

    x = []
    y = []
    in_interval = False

    for i in range(len(x_values)):
        if in_interval:
            if x_values[i] > float(interval[1]):
                break
            else:
                x.append(x_values[i])
                y.append(y_values[i])
        elif x_values[i] >= float(interval[0]):
            in_interval = True

    value = np.trapz(y, x)

    # i = 0
    # while True:
    #     if x_values[i] == 676.5:
    #         break
    #     i += 1
    
    # for j in range(12):
    #     print(x_values[i+j])

    # for j in range(12):
    #     print(y_values[i+j])

    return value


# COULD DO IT IN THE WRONG ORDER!
def integrate(interval, to_folder, tag):
    current_path = os.getcwd()
    files = os.listdir()
    values = []
    names = []

    for f in files:
        if f != "processing.py":
            values.append(calculate_integral(interval, f))
            names.append(f)
            print(f)
            print(values[-1])
    
    for i in range(5):
        tmp = []
        for j in range(5):
            tmp.append(str(values[5*i + j]) + "\n")

        os.chdir(to_folder)
        new_file = open(tag + "_" + names[5*i][1:-6] + ".txt", "w")
        new_file.writelines(tmp)
        new_file.close()
        os.chdir(current_path)
    

if __name__ == "__main__":
    # format_file("test_data.cmbl")
    # x, y = read_file("test_data.cmbl")
    # print(x)
    # print()
    # print(y)
    
    #copy_to_other_folder_and_change_name("C:--Users--Hampus--VS_Code_Projects--GA--Spektrofotometri--tmp--72h", "72h")
    #format_all_files("C:\\Users\\Hampus\\VS_Code_Projects\\GA\\Spektrofotometri\\d_text_files")

    #k_difference("C:\\Users\\Hampus\\VS_Code_Projects\\GA\\Spektrofotometri\\f_differens_från_kontroll\\72h")
    #print(calculate_integral((675, 685), "0h_1_1.txt"))

    integrate((380, 800), "C:\\Users\\Hampus\\VS_Code_Projects\\GA\\Spektrofotometri_latest\\j_ickediff_integral_synligtljus\\0h", "id_div1")