import os
import values
from  wave import wavelength_to_rgb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import statsmodels.api as sm
import scipy.stats

ignore = ["regression.py", "values.py", "__pycache__", "combination", "wave.py", "coefficients", "programs", "processing.py", "test.py", "combination2", "data"]
coefficient_folder = "coefficients"
combination_folder = "combination"
data_folder = "data"

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

def read_file_from_folder(file_name, folder):
    x_values = []
    y_values = []
    reading_x = True

    cwd = os.getcwd()
    os.chdir(folder)

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
    os.chdir(cwd)

    return (x_values, y_values)

def get_file_components(file_name):
    return file_name[:-4].split("_")

def get_concentration(file_name):
    components = get_file_components(file_name)
    p = int(components[1]) / 100
    c = p ** (int(components[2]) - 1)

    if components[3] == "x":
        c = c * 0.73
        # det borde vara delat med 0.73, men blir konstigt av någon anledning
    
    return c

def get_total(species, c):
    if species == "n":
        return c * 2.019 * 10**7 * 0.73
    elif species == "a":
        return c * 1800
    else:
        return c

def get_files_such_that(species=None, concentration=None, n=None, extra=None):
    properties = (species, concentration, n, extra)
    indexes = []
    files = []

    if species != None:
        indexes.append(0)
    if concentration != None:
        indexes.append(1)
    if n != None:
        indexes.append(2)
    if extra != None:
        indexes.append(3)
    
    all_files = os.listdir()

    for f in all_files:
        if not f in ignore:
            components = get_file_components(f)
            add = True
            for i in indexes:
                if components[i] != properties[i]:
                    add = False
            if add:
                files.append(f)
    
    return files

def get_absorbance_and_concentration(file_name):
    y = read_file(file_name)[1]
    c = get_concentration(file_name)
    return (c, y)

def plot_curve(file_name):
    x, y = read_file(file_name)

    fig, ax = plt.subplots()
    del fig

    colors = []

    for i in range(len(x)):
        colors.append(i)

    ax.plot(x, y, c=colors)

    plt.show()

def plot_curve_with_color(file_name, bar=True, raw_values=None):
    if raw_values == None:
        x, y = read_file(file_name)
    else:
        x, y = raw_values

    c = np.array(values.color_spectrum)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    clim=(380,780)
    norm = plt.Normalize(*clim)
    wl = np.arange(clim[0],clim[1]+1,2)
    colorlist = list(zip(norm(wl),[wavelength_to_rgb(((w-380)/(780-380))*(750-380)+380) for w in wl]))
    spectralmap = LinearSegmentedColormap.from_list("spectrum", colorlist)

    fig, axs = plt.subplots()
    lc = LineCollection(segments, cmap=spectralmap, norm=norm)
    lc.set_array(c)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    if bar:
        fig.colorbar(line, ax=axs)

    axs.set_xlim(values.spectrum[0], 779)
    axs.set_ylim(0, 1.1)
    plt.show()


def plot_3d_curve(file_name):
    x, y = read_file(file_name)

    x = x[:len(values.color_spectrum)]
    y = y[:len(values.color_spectrum)]

    z = [0 for i in x]

    c = np.array(values.color_spectrum)

    points = np.array([x, z, y]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    clim=(380,780)
    norm = plt.Normalize(*clim)
    wl = np.arange(clim[0],clim[1]+1,2)
    colorlist = list(zip(norm(wl), [wavelength_to_rgb(((w-380)/(780-380))*(750-380)+380) for w in wl]))
    spectralmap = LinearSegmentedColormap.from_list("spectrum", colorlist)

    #lc = Line3DCollection(segments, cmap=spectralmap, norm=norm)

    lc = Line3DCollection(segments, cmap=spectralmap, norm=norm)
    lc.set_array(c)
    lc.set_linewidth(2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    line = ax.add_collection3d(lc, zs=z, zdir='z')
    del line

    ax.set_xlim(380, 780)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 1.2)

    # if bar:
    #     fig.colorbar(line, ax=axs)

    # axs.set_xlim(values.spectrum[0], 779)
    # axs.set_ylim(0, 1.1)

    #ax.plot(x, z, y, zdir='z', c="blue")

    plt.show()

    #ax.scatter(x, z, y, zdir='z', s=10, c="blue", depthshade=True)


def plot_3d_surface_curve(files):

    # Creates colormaps
    c = np.array(values.color_spectrum)
    clim=(380,780)
    norm = plt.Normalize(*clim)
    wl = np.arange(clim[0],clim[1]+1,2)

    colorlist1 = list(zip(norm(wl),[wavelength_to_rgb(((w-380)/(780-380))*(750-380)+380, opacity=1) for w in wl]))
    spectralmap1 = LinearSegmentedColormap.from_list("spectrum", colorlist1)

    colorlist2 = list(zip(norm(wl),[wavelength_to_rgb(((w-380)/(780-380))*(750-380)+380, opacity=0.03) for w in wl]))
    spectralmap2 = LinearSegmentedColormap.from_list("spectrum", colorlist2)

    # colorlist1 = list(zip(norm(wl),[(0, 0, 0, 1) for w in wl]))
    # spectralmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("spectrum", colorlist1)

    # Creates figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    all_x, all_y, all_z = [], [], []
    species = get_species(files[0])

    for f in files:
        x, y = read_file(f)
        x = x[:len(values.color_spectrum)]
        y = y[:len(values.color_spectrum)]
        conc = get_total(species, get_concentration(f))
        z = np.array([conc for i in x])

        all_x.append(x)
        all_y.append(y)
        all_z.append(z)

        points = np.array([x, z, y]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = Line3DCollection(segments, cmap=spectralmap1, norm=norm)
        lc.set_array(c)
        lc.set_linewidth(2)

        ax.add_collection3d(lc, zs=z, zdir='z')


    X = np.array(all_x)
    Y = np.array(all_y)
    Z = np.array(all_z)

    # # Imports values
    # x, y = read_file("n_40_1_0.txt")
    # x = x[:len(values.color_spectrum)]
    # y = y[:len(values.color_spectrum)]
    # z = np.zeros(len(x))

    # x2, y2 = read_file("n_40_2_0.txt")
    # x2 = x2[:len(values.color_spectrum)]
    # y2 = y2[:len(values.color_spectrum)]
    # z2 = np.ones(len(x2)) * 0.1

    # X = np.array((x, x2))
    # Y = np.array((y, y2))
    # Z = np.array((z, z2))


    # Adds the fourth dimension - colormap

    color_dimension = X
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap=spectralmap2)
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)

    # plot
    ax.plot_surface(X,Z,Y, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)

    ax.set_xlabel('Våglängd')
    ax.set_ylabel('Antal')
    ax.set_zlabel('Absorbans')
    ax.set_xlim(380, 780)

    if species == "r":
        ax.set_ylabel('Koncentration')
    elif species == "n":
        ax.set_zlim(0, 1.2)
    else:
        ax.set_zlim(0, 0.3)

    plt.show()


def calculate_integral(interval, file_name, raw_values=None):
    if raw_values == None:
        x_values, y_values = read_file(file_name)
    else:
        x_values, y_values = raw_values

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

    return value

def get_value_at(wavelength, file_name):
    y = read_file(file_name)[1]
    index = values.spectrum_dict[wavelength]
    return y[index]

def get_species(file_name):
    return get_file_components(file_name)[0]

def print_values(concentrations, integrals):
    print()
    concentrations = sorted(concentrations)
    integrals = sorted(integrals)
    for i in concentrations:
        print(str(i).replace(".", ","))
    print()
    for i in integrals:
        print(str(i).replace(".", ","))
    print()

def regress(files, interval=None, wavelength=None, show=False):
    if interval == None:
        f = lambda x: get_value_at(wavelength, x)
    else:
        f = lambda x: calculate_integral(interval, x)

    species = get_species(files[0])
    concentrations = list(map(get_concentration, files))
    total = list(map(lambda x: get_total(species, x), concentrations))
    absorbance_values = list(map(f, files))



    x, y = np.array(total), np.array(absorbance_values)
    #print(scipy.stats.pearsonr(x, y))
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()

    if show:
        print(results.summary())
        print(results.rsquared)
        print(np.sqrt(results.rsquared))
        print(len(absorbance_values))
    
        fig, ax = plt.subplots()
        del fig

        ax.scatter(total, absorbance_values)

        print_values(total, absorbance_values)

        plt.show()
    
    return tuple(results.params)


def get_regression_coefficients(files):
    coefficients = [regress(files, wavelength=wl) for wl in values.spectrum]
    return coefficients

def save_regression_coefficients(files, name, folder):
    coef = get_regression_coefficients(files)
    coef_formatted = map(lambda x: str(x[0]) + " " + str(x[1]) + "\n", coef)

    cwd = os.getcwd()
    os.chdir(folder)
    f = open(name, "w")
    
    f.writelines(coef_formatted)

    f.close()
    os.chdir(cwd)

def import_regression_coefficients(name, folder):
    cwd = os.getcwd()
    os.chdir(folder)
    f = open(name, "r")
    data = f.readlines()
    f.close()
    os.chdir(cwd)

    coef = [(float(i.split(" ")[0]), float(i.split(" ")[1])) for i in data]
    return coef


def get_interval_regression_coefficients(files, interval_length, number_of_intervals):
    if interval_length * number_of_intervals > len(values.spectrum):
        print("Could not fit interval!")
        return None
    
    intervals = []
    for i in range(number_of_intervals):
        intervals.append((values.spectrum[i * interval_length], values.spectrum[(i + 1) * interval_length - 1]))
    
    # for f in intervals:
    #     print(f)
    #     print()
    
    coefficients = [regress(files, interval=i) for i in intervals]
    return coefficients

def save_interval_regression_coefficients(files, interval_length, number_of_intervals, name, folder):
    coef = get_interval_regression_coefficients(files, interval_length, number_of_intervals)
    coef_formatted = map(lambda x: str(x[0]) + " " + str(x[1]) + "\n", coef)

    cwd = os.getcwd()
    os.chdir(folder)
    f = open(name, "w")

    f.write(str(interval_length) + " " + str(number_of_intervals) + "\n")
    f.writelines(coef_formatted)

    f.close()
    os.chdir(cwd)

def import_interval_regression_coefficients(name, folder):
    cwd = os.getcwd()
    os.chdir(folder)
    f = open(name, "r")
    data = f.readlines()
    f.close()
    os.chdir(cwd)

    interval_length = int(data[0].split(" ")[0])
    number_of_intervals = int(data[0].split(" ")[1])
    data.pop(0)

    coef = [(float(i.split(" ")[0]), float(i.split(" ")[1])) for i in data]
    return (coef, interval_length, number_of_intervals)

def model_regression(file_values, category, species1, species2, show=False):
    categories = {"all": "spectrum",
                "12": "42_12",
                "50": "10_50",
                "65": "10_65",
                "215": "3_215"}

    file_extension = "_coef_" + categories[category] + ".txt"
    f1 = species1 + file_extension
    f2 = species2 + file_extension

    if category == "all":
        coef1 = import_regression_coefficients(f1, coefficient_folder)
        coef2 = import_regression_coefficients(f2, coefficient_folder)
        y = file_values[1]
    else:
        coef1, interval_length, number_of_intervals = import_interval_regression_coefficients(f1, coefficient_folder)
        coef2 = import_interval_regression_coefficients(f2, coefficient_folder)[0]
        
        intervals = []
        for i in range(number_of_intervals):
            intervals.append((values.spectrum[i * interval_length], values.spectrum[(i + 1) * interval_length - 1]))
        
        y = list(map(lambda x: calculate_integral(x, "", raw_values=file_values), intervals)) 

    y_prime = []
    for i in range(len(coef1)):
        y_prime.append(y[i] - coef1[i][0] - coef2[i][0])

    y_bis = []
    x = []
    for i in range(len(coef1)):
        y_bis.append(y_prime[i] / coef2[i][1]) # går att förkorta?
        x.append(coef1[i][1] / coef2[i][1])

    x, y_bis = np.array(x), np.array(y_bis)
    x = sm.add_constant(x)
    model = sm.OLS(y_bis, x)
    results = model.fit()

    s1 = results.params[1]
    s2 = results.params[0]

    if show:
        print(results.summary())
    
    return (s1, s2)

def get_combo_actual_values(file_name):
    components = file_name[:-4].split("_")

    r_p = int(components[0][1:]) / 100
    r_c = r_p ** (int(components[2]) - 1) * 0.5 # här kan det behöva ändras

    n_p = int(components[1][1:]) / 100
    n_c = n_p ** (int(components[3]) - 1)  * 0.5
    n_total = get_total("n", n_c)

    if components[4] == "x":
        pass

    return (n_total, r_c)

def analyze_all(category):
    cwd = os.getcwd()
    os.chdir(combination_folder)
    files_tmp = os.listdir()
    
    files = []
    vals = []
    for f in files_tmp:
        if f[:-4].split("_")[4] == "0": # Ta bort filtret om du vill ha med x-filerna
            files.append(f)
            vals.append(read_file(f))
    os.chdir(cwd)

    results = []
    for i in range(len(files)):
        prediction = model_regression(vals[i], category, "n", "r")
        actual = get_combo_actual_values(files[i])
        results.append((prediction, actual))
    
    wrongs = 0
    n = 0
    a_tot = 0
    b_tot = 0

    q = []

    for i in range(len(results)):
        if results[i][0][0] < 0:
            wrongs += 1
        else:
            a = abs(results[i][1][0] - results[i][0][0]) / results[i][1][0]
            b = abs(results[i][1][1] - results[i][0][1]) / results[i][1][1]

            a_tot += a
            b_tot += b
            n += 1

            q.append((a, b, files[i]))
    
        print(results[i][0][0])
        print(results[i][1][0])
        print(results[i][0][1])
        print(results[i][1][1])
        print()

    q.sort(key=lambda x: x[0] + x[1], reverse=True)
    print()
    for i,j,f in q:
        print(f)
        print(i)
        print(j)
        print()
        

    if n > 0:
        print(a_tot / n)
        print(b_tot / n)
        print("LOL")

        return ((a_tot + b_tot) / (2 * n), wrongs)


def analyze_all_graph(category, do_prediction=False):
    cwd = os.getcwd()
    os.chdir(combination_folder)
    files_tmp = os.listdir()

    if do_prediction:
        pred = 0
    else:
        pred = 1
    
    ignore_these = ["r75_n40_2_3_0.txt", "r75_n40_2_4_0.txt", "r75_n40_3_5_0.txt", "r75_n40_4_3_0.txt", "r75_n40_4_5_0.txt"]

    #ignore_these = []

    # ignore_these = ["r40_n75_2_4_0.txt", "r40_n75_2_5_0.txt", "r40_n75_4_3_0.txt", "r40_n75_5_2_0.txt", "r40_n75_5_5_0.txt"]
    # ignore_these.extend(["r40_n75_3_1_0.txt", "r40_n75_3_2_0.txt", "r40_n75_4_4_0.txt", "r40_n75_5_3_0.txt", "r40_n75_5_4_0.txt"])
    
    files = []
    vals = []
    for f in files_tmp:
        components = f[:-4].split("_")
        if components[4] == "0" and not (f in ignore_these and do_prediction and True) and not (components[0] == "r40" and components[1] == "n40" and components[3] == "5" and do_prediction and False): # Ta bort filtret om du vill ha med x-filerna
            files.append(f)
            vals.append(read_file(f))
    os.chdir(cwd)

    print(len(files))

    results = []
    results2 = []
    for i in range(len(files)):
        prediction = model_regression(vals[i], category, "n", "r")
        actual = get_combo_actual_values(files[i])
        results.append((prediction, actual, files[i]))

    real_nanno, predicted_nanno, real_red, predicted_red = [], [], [], []
    for k in results:
        #if k[1][0] > 2 * 10 ** 6:
        real_nanno.append(k[1][0])
        predicted_nanno.append(k[0][0])
        real_red.append(k[1][1])
        predicted_red.append(k[0][1])


    # a_tot = 0
    # for i in range(len(real_nanno)):
    #     a_tot += abs(real_nanno[i] - predicted_nanno[i]) / real_nanno[i]
    # print("LOL")
    # print(a_tot / len(real_nanno))
    # print(len(real_nanno))

    wrongs = 0
    n = 0
    a_tot = 0
    b_tot = 0

    response = []
    nanno_error = []
    red_error = []
    regressor1 = []
    regressor2 = []

    for i in range(len(results)):
        if results[i][0][0] < 0 or results[i][0][1] < 0:
            wrongs += 1
        else:
            a = abs(results[i][1][0] - results[i][0][0]) / results[i][1][0]
            b = abs(results[i][1][1] - results[i][0][1]) / results[i][1][1]

            # a = (results[i][1][0] - results[i][0][0]) / results[i][1][0]
            # b = (results[i][1][1] - results[i][0][1]) / results[i][1][1]

            a_tot += a
            b_tot += b
            n += 1

            response.append((a + b) / 2)
            regressor1.append(results[i][pred][0]) # mitten kan ändras till 1
            regressor2.append(results[i][pred][1])

            nanno_error.append(a)
            red_error.append(b)

            if a > 1.1:
                print(results[i][2])
    
    
    # print()

    # for k in range(len(nanno_error)):
    #     if nanno_error[k] > 1.1:
    #         # print(nanno_error[k])
    #         print(files[k])
    
    response = np.array(response)
    nanno_error = np.array(nanno_error)
    red_error = np.array(red_error)
    regressor1 = np.array(regressor1)
    regressor2 = np.array(regressor2)

    if not do_prediction:
        f = lambda x: 1 / x
        g = lambda x: 1 / x
    else:
        k = 0.38
        j = 0.003

        f = lambda x: 1 / (np.sqrt(x))
        g = lambda x: (((x+j)/k)**2) / ((1.3*(((x+j)/k)**3)) + ((-3)*(((x+j)/k)**4)) + (1.85*(((x+j)/k)**5))) * 1.37

    # f = lambda x: (10 ** ((-x) / 10**7)) ** 2
    # g = lambda x: 1 / (x ** 2)

    old1 = regressor1
    old2 = regressor2
    regressor1 = f(regressor1)
    regressor2 = g(regressor2)

    regressors = np.column_stack((regressor1, regressor2))
    #regressors = sm.add_constant(regressors)

    model = sm.OLS(response, regressors)
    results = model.fit()

    print(results.summary())
    print()

    model1 = sm.OLS(nanno_error, regressor1)
    results1 = model1.fit()
    model2 = sm.OLS(red_error, regressor2)
    results2 = model2.fit()

    # for k in range(len(nanno_error)):
    #     if nanno_error[k] > 1.1:
    #         print(nanno_error[k])

    print(results1.summary())
    print()
    print(results2.summary())


    fig, ax = plt.subplots()
    del fig

    #print(results1.params[0])

    if not do_prediction:
        h = lambda x: results1.params[0] / x
        x_values = np.linspace(0.01 * 10**7, 0.8 * 10**7, 1000)
    else:
        h = lambda x: results1.params[0] * f(x)
        x_values = np.linspace(0.01 * 10**7, 0.8 * 10**7, 1000)

    estimate = h(x_values)

    ax.scatter(old1, nanno_error * 100, c="#99D17B")
    ax.set_ylabel("Procentuellt fel")
    ax.set_xlabel("Antal")

    ax.plot(x_values, estimate * 100, alpha=0.5)

    plt.show()

    fig, ax = plt.subplots()
    del fig

    if not do_prediction:
        h = lambda x: results2.params[0] / x
        x_values = np.linspace(0.01, 0.4, 1000)
    else:
        h = lambda x: results2.params[0] * g(x)
        #h = lambda x: 1 * g(x)
        x_values = np.linspace(0.005, 0.4, 1000)


    estimate = h(x_values)

    ax.scatter(old2, red_error * 100, c="#EF5B5B")
    ax.set_ylabel("Procentuellt fel")
    ax.set_xlabel("Andel av stamlösning")

    ax.plot(x_values, estimate * 100, alpha=0.5)

    plt.show()

    remove_these = []
    j = 0
    for i in range(len(predicted_nanno)):
        if predicted_nanno[i] < 0:
            remove_these.append(i - j)
            j += 1
    
    for i in remove_these:
        del predicted_nanno[i]
        del real_nanno[i]


    fig, ax = plt.subplots()
    del fig

    ax.scatter(real_nanno, predicted_nanno, c="#99D17B")
    ax.set_xlabel("Äkta antal N. salina")
    ax.set_ylabel("Antal uppskattade N. salina")

    predicted_nanno = np.array(predicted_nanno)
    real_nanno = np.array(real_nanno)

    #predicted_nanno.reshape(-1, 1)
    real_nanno = sm.add_constant(real_nanno)

    # print(predicted_nanno)
    # print(real_nanno)
    print("LOL")

    model3 = sm.OLS(predicted_nanno, real_nanno)
    results3 = model3.fit()
    print(results3.summary())
    #print(results3.rsquared)
    print(results.params)

    x_values = np.linspace(0, 10 ** 7, 1000)

    ax.set_xlim(0, 0.95 * 10 ** 7)
    ax.set_ylim(0, 0.95 * 10 ** 7)

    ax.plot(x_values, results3.params[0] + results3.params[1] * x_values, alpha=0.5)

    ax.plot(x_values, x_values, alpha=0.5, color="red")

    plt.show()


    fig, ax = plt.subplots()
    del fig

    predicted_nanno = np.array(predicted_red)
    real_nanno = np.array(real_red)

    ax.scatter(real_nanno, predicted_nanno, c="#EF5B5B")
    ax.set_xlabel("Äkta andel rödbetsjuice")
    ax.set_ylabel("Uppskattad andel rödbetsjuice")

    #predicted_nanno.reshape(-1, 1)
    real_nanno = sm.add_constant(real_nanno)

    # print(predicted_nanno)
    # print(real_nanno)
    print("LOL")

    model3 = sm.OLS(predicted_nanno, real_nanno)
    results3 = model3.fit()
    print(results3.summary())
    #print(results3.rsquared)
    print(results.params)

    x_values = np.linspace(0, 1, 1000)

    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)

    ax.plot(x_values, results3.params[0] + results3.params[1] * x_values, alpha=0.5)

    #ax.plot(x_values, x_values, alpha=0.5, color="red")

    plt.show()


    # print(response)
    # print()
    # print(regressors)

    # for i in errors:
    #     print(i)

    fig, ax = plt.subplots()
    del fig

    ax.scatter(old2, nanno_error * 100, c="#EF5B5B")
    ax.set_ylabel("Procentuellt fel")
    ax.set_xlabel("Andel av stamlösning")

    plt.show()


def analyze_all_graph_3d(category):
    cwd = os.getcwd()
    os.chdir(combination_folder)
    files_tmp = os.listdir()
    
    files = []
    vals = []
    for f in files_tmp:
        if f[:-4].split("_")[4] == "0": # Ta bort filtret om du vill ha med x-filerna
            files.append(f)
            vals.append(read_file(f))
    os.chdir(cwd)

    results = []
    for i in range(len(files)):
        prediction = model_regression(vals[i], category, "n", "r")
        actual = get_combo_actual_values(files[i])
        results.append((prediction, actual))
    
    wrongs = 0
    n = 0
    a_tot = 0
    b_tot = 0

    response = []
    nanno_error = []
    red_error = []
    regressor1 = []
    regressor2 = []

    for i in range(len(results)):
        if results[i][0][0] < 0:
            wrongs += 1
        else:
            a = abs(results[i][1][0] - results[i][0][0]) / results[i][1][0]
            b = abs(results[i][1][1] - results[i][0][1]) / results[i][1][1]

            a_tot += a
            b_tot += b
            n += 1

            response.append((a + b) / 2)
            regressor1.append(results[i][1][0])
            regressor2.append(results[i][1][1])

            nanno_error.append(a)
            red_error.append(b)
    
    response = np.array(response)
    nanno_error = np.array(nanno_error)
    red_error = np.array(red_error)
    regressor1 = np.array(regressor1)
    regressor2 = np.array(regressor2)

    a, b, c = response, regressor1, regressor2

    regressor1, regressor2 = np.meshgrid(regressor1, regressor2)
    response = np.outer(response.T, response)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    color_dimension = response
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    m.set_array([])
    #fcolors = m.to_rgba(color_dimension)

    # plot
    #ax.plot_surface(regressor1, regressor2, response, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)

    ax.scatter(a * 100, b, c)

    ax.set_xlabel('Andel av stamlösning rödbetsjuice')
    ax.set_ylabel('Antal N. salina')
    ax.set_zlabel('Procentuellt fel')

    plt.show()


def analyze_data(category, folder):
    cwd = os.getcwd()
    os.chdir(data_folder)
    os.chdir(folder)
    files_tmp = os.listdir()
    
    files = []
    vals = []
    for f in files_tmp:
        if True: # Ta bort filtret om du vill ha med x-filerna
            files.append(f)
            vals.append(read_file(f))
    os.chdir(cwd)

    results = []
    for i in range(len(files)):
        prediction = model_regression(vals[i], category, "n", "a")
        results.append((prediction, files[i]))
    
    results.sort(key=lambda x: int(x[1].split("_")[1]))

    for i in results:
        print(i[1])
        print(i[0])
        print()

    means = []
    for i in range(int(len(results) / 5)):
        nanno, artemia, n = 0, 0, 0

        for j in range(5):
            if results[5*i + j][0][1] > 0 and results[5*i + j][0][0] > 0:
                nanno += results[5*i + j][0][0]
                artemia += results[5*i + j][0][1]
                n += 1

                print(str(results[5*i + j][0][0] / 2).replace(".", ","))
                
        print()

        means.append((nanno / n, artemia / n)) 
    
    print(str(10211076.855164354 / 2) + "x")
    
    print(means[0])
    print()
    del means[0] # tar bort kontroll

    for i in means:
        print(i)
    


if __name__ == "__main__":

    files = get_files_such_that(species="n", extra="0")
    #species="n",

    # for f in files:
    #     tmp = read_file(f)
    #     print(len(tmp[1]))
    
    # print(len(files))
    
    # x = 1 / (5-5)

    #print(get_combo_actual_values("r75_n40_2_2_0.txt"))

    #analyze_all_graph_3d("65")

    analyze_all_graph("65", do_prediction=False)
    
    analyze_data("65", "72h")

    #plot_3d_surface_curve(files)

    # tmp = read_file_from_folder("r75_n40_2_2_0.txt", combination_folder)
    # print(model_regression(tmp, "all", "n", "r", show=True))
    # print(get_combo_actual_values("r75_n40_2_2_0.txt.txt"))

    #analyze_all_graph("65")
    #analyze_all_graph_3d("65")

    #65 eller all bäst

    # a = analyze_all("12"), analyze_all("50"), analyze_all("65"), analyze_all("215"), analyze_all("all")

    # print()
    # for i in a:
    #     print(i)

    #plot_curve_with_color("hej", raw_values=tmp)

    #print(calculate_integral((380.3, 387.6), "r75_n75_2_1_0copy.txt"))

    #print(get_concentration("n_75_4_0.txt"))
    #files = get_files_such_that(concentration="25", n="3")

    # s = values.spectrum
    # dicti = {}
    # for i in range(len(s)):
    #     dicti[s[i]] = i
    # print(dicti)

    #print(get_value_at(381.1, "n_75_1_0.txt"))

    #regress(files, interval=(380, 780))

    #regress(files, wavelength=651.1, show=True)
    #regress(files, wavelength=899.2, show=True)

    #print(len(values.spectrum))

    #save_interval_regression_coefficients(files, 3, 215, "n_coef_3_215.txt", coefficient_folder)

    # save_interval_regression_coefficients(files, 10, 65, "r_coef_10_65.txt", coefficient_folder)
    # save_interval_regression_coefficients(files, 10, 65, "a_coef_10_65.txt", coefficient_folder)

    #save_interval_regression_coefficients(files, 42, 12, "n_coef_42_12.txt", coefficient_folder)
    # save_interval_regression_coefficients(files, 10, 50, "n_coef_10_50.txt", coefficient_folder)
    # save_regression_coefficients(files, "n_coef_spectrum.txt", coefficient_folder)

    #regress(files, interval=(380, 780), show=True)

    #print(import_interval_regression_coefficients("n_coef_10_50.txt", coefficient_folder))

    #print(import_interval_regression_coefficients("r_coef.txt"))

    #print(get_regression_coefficients(files))
    #save_regression_coefficients(files, "hej.txt")
    #import_regression_coefficients("n_coef.txt")

    #regress(files, wavelength=681.0)

    #print(regress(files, interval=(388.4, 395.8)))

    #plot_curve_with_color(files[0])


    ## HÄR
    # cwd = os.getcwd()
    # os.chdir(data_folder)
    # os.chdir("72h")
    # tmp = read_file("72h_1_3.txt")
    # #print(tmp)
    # os.chdir(cwd)



    # print(len(tmp[0]))
    # print(len(tmp[1]))


    # tmp = read_file("n_75_1_0.txt")

    # print(len(tmp[0]))
    # print(len(tmp[1]))

    # a, b = tmp
    # b.append(0)
    # tmp = (a, b)

    # plot_curve_with_color("hej", raw_values=tmp, bar=True)

    #plot_3d_curve(files[0])

    #plot_3d_surface_curve(files)

    # for f in files:
    #     c, a = get_absorbance_and_concentration(f)
    #     print(c)
    #     print(a)
    #     print()

    #plot_curve_with_color("n_75_1_0.txt")

    #print(files)