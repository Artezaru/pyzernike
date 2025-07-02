import numpy
from pyzernike import zernike_polynomial_up_to_order, zernike_polynomial, xy_zernike_polynomial, xy_zernike_polynomial_up_to_order
from pyzernike.deprecated import zernike_polynomial as old_zernike_polynomial
from pyzernike.deprecated import common_zernike_polynomial
import time
import matplotlib.pyplot as plt
import csv
import os


def mesure_time_old_zernike(Npoints, Norder, derivative):
    tic = time.perf_counter()
    # Generate 100 random rho values between 0 and 1
    rho = numpy.linspace(0, 1, Npoints)
    theta = numpy.linspace(0, 2 * numpy.pi, Npoints)

    # Compute all in a loop
    for n in range(Norder + 1):
        for m in range(-n, n + 1, 2):
            old_zernike_polynomial(rho, theta, n, m)
            if derivative == True:
                old_zernike_polynomial(rho, theta, n, m, rho_derivative=1)
                old_zernike_polynomial(rho, theta, n, m, theta_derivative=1)

    toc = time.perf_counter()
    print(f"Old Zernike polynomial computation time for Npoints={Npoints}, Norder={Norder}, derivative={derivative}: {toc - tic:.4f} seconds")
    return toc - tic


def mesure_zernike_polynomial_single(Npoints, Norder, derivative):
    tic = time.perf_counter()
    # Generate 100 random rho values between 0 and 1
    rho = numpy.linspace(0, 1, Npoints)
    theta = numpy.linspace(0, 2 * numpy.pi, Npoints)

    # Compute all in a loop
    for n in range(Norder + 1):
        for m in range(-n, n + 1, 2):
            zernike_polynomial(rho, theta, [n], [m])[0]
            if derivative == True:
                zernike_polynomial(rho, theta, [n], [m], rho_derivative=[1])[0]
                zernike_polynomial(rho, theta, [n], [m], theta_derivative=[1])[0]

    toc = time.perf_counter()
    print(f"Single Zernike polynomial computation time for Npoints={Npoints}, Norder={Norder}, derivative={derivative}: {toc - tic:.4f} seconds")
    return toc - tic


def mesure_zernike_polynomial_mult(Npoints, Norder, derivative):
    tic = time.perf_counter()
    # Generate 100 random rho values between 0 and 1
    rho = numpy.linspace(0, 1, Npoints)
    theta = numpy.linspace(0, 2 * numpy.pi, Npoints)

    # Prepare the lists for n, m, rho_derivative, and theta_derivative
    list_n = []
    list_m = []
    list_dr = []
    list_dt = []
    for n in range(Norder + 1):
        for m in range(-n, n + 1, 2):
            list_n.append(n)
            list_m.append(m)
            list_dr.append(0)
            list_dt.append(0)
            if derivative:
                list_n.append(n)
                list_m.append(m)
                list_dr.append(1)
                list_dt.append(0)
                list_n.append(n)
                list_m.append(m)
                list_dr.append(0)
                list_dt.append(1)
    
    # Call the zernike_polynomial function with the lists
    result = zernike_polynomial(rho, theta, n=list_n, m=list_m, rho_derivative=list_dr, theta_derivative=list_dt)
    toc = time.perf_counter()
    print(f"Mult Zernike polynomial computation time for Npoints={Npoints}, Norder={Norder}, derivative={derivative}: {toc - tic:.4f} seconds")
    return toc - tic



def mesure_zernike_polynomial_commun(Npoints, Norder, derivative):
    tic = time.perf_counter()
    # Generate 100 random rho values between 0 and 1
    rho = numpy.linspace(0, 1, Npoints)
    theta = numpy.linspace(0, 2 * numpy.pi, Npoints)

    # Prepare the lists for n, m, rho_derivative, and theta_derivative
    list_n = []
    list_m = []
    list_dr = []
    list_dt = []
    for n in range(Norder + 1):
        for m in range(-n, n + 1, 2):
            list_n.append(n)
            list_m.append(m)
            list_dr.append(0)
            list_dt.append(0)
            if derivative:
                list_n.append(n)
                list_m.append(m)
                list_dr.append(1)
                list_dt.append(0)
                list_n.append(n)
                list_m.append(m)
                list_dr.append(0)
                list_dt.append(1)
    
    # Call the zernike_polynomial function with the lists
    result = common_zernike_polynomial(rho, theta, n=list_n, m=list_m, rho_derivative=list_dr, theta_derivative=list_dt)
    toc = time.perf_counter()
    print(f"Commun Zernike polynomial computation time for Npoints={Npoints}, Norder={Norder}, derivative={derivative}: {toc - tic:.4f} seconds")
    return toc - tic
                

def mesure_zernike_polynomial_up_to_order(Npoints, Norder, derivative):
    tic = time.perf_counter()
    # Generate 100 random rho values between 0 and 1
    rho = numpy.linspace(0, 1, Npoints)
    theta = numpy.linspace(0, 2 * numpy.pi, Npoints)

    # Call the zernike_polynomial_up_to_order function
    result = zernike_polynomial_up_to_order(rho=rho, theta=theta, order=Norder, rho_derivative=[0, 1, 0] if derivative else [0], theta_derivative=[0, 0, 1] if derivative else [0])
    
    toc = time.perf_counter()
    print(f"Up to order Zernike polynomial computation time for Npoints={Npoints}, Norder={Norder}, derivative={derivative}: {toc - tic:.4f} seconds")
    return toc - tic


def mesure_zernike_polynomial_xy(Npoints, Norder, derivative):
    tic = time.perf_counter()
    # Generate 100 random x and y values between -1 and 1
    x = numpy.linspace(-1, 1, Npoints)
    y = numpy.linspace(-1, 1, Npoints)

     # Prepare the lists for n, m, rho_derivative, and theta_derivative
    list_n = []
    list_m = []
    list_dx = []
    list_dy = []
    for n in range(Norder + 1):
        for m in range(-n, n + 1, 2):
            list_n.append(n)
            list_m.append(m)
            list_dx.append(0)
            list_dy.append(0)
            if derivative:
                list_n.append(n)
                list_m.append(m)
                list_dx.append(1)
                list_dy.append(0)
                list_n.append(n)
                list_m.append(m)
                list_dx.append(0)
                list_dy.append(1)

    # Call the zernike_polynomial_up_to_order function
    result = xy_zernike_polynomial(x=x, y=y, n=list_n, m=list_m, x_derivative=list_dx, y_derivative=list_dy, Rx=numpy.sqrt(2), Ry=numpy.sqrt(2), x0=0, y0=0)
    
    toc = time.perf_counter()
    print(f"XY Zernike polynomial computation time for Npoints={Npoints}, Norder={Norder}, derivative={derivative}: {toc - tic:.4f} seconds")
    return toc - tic



def mesure_zernike_polynomial_xy_up_to_order(Npoints, Norder, derivative):
    tic = time.perf_counter()
    # Generate 100 random x and y values between -1 and 1
    x = numpy.linspace(-1, 1, Npoints)
    y = numpy.linspace(-1, 1, Npoints)

    # Call the zernike_polynomial_up_to_order function
    result = xy_zernike_polynomial_up_to_order(x=x, y=y, order=Norder, x_derivative=[0, 1, 0] if derivative else [0], y_derivative=[0, 0, 1] if derivative else [0], Rx=numpy.sqrt(2), Ry=numpy.sqrt(2), x0=0, y0=0)
    
    toc = time.perf_counter()
    print(f"XY Zernike polynomial computation time for Npoints={Npoints}, Norder={Norder}, derivative={derivative}: {toc - tic:.4f} seconds")
    return toc - tic


if __name__ == "__main__":

    filename = os.path.join(os.path.dirname(__file__), 'compare_speed_all_polynomials.csv')
    Npoints = 1_000_000
    exist_csv = os.path.isfile(filename)

    if not exist_csv:
        Norder = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        time_old = [mesure_time_old_zernike(Npoints, n, False) for n in Norder]
        print("============================================= time_old FINISHED =============================================")
        time_old_derivative = [mesure_time_old_zernike(Npoints, n, True) for n in Norder]
        print("============================================= time_old_derivative FINISHED =============================================")
        time_single = [mesure_zernike_polynomial_single(Npoints, n, False) for n in Norder]
        print("============================================= time_single FINISHED =============================================")
        time_single_derivative = [mesure_zernike_polynomial_single(Npoints, n, True) for n in Norder]
        print("============================================= time_single_derivative FINISHED =============================================")
        time_mult = [mesure_zernike_polynomial_mult(Npoints, n, False) for n in Norder]
        print("============================================= time_mult FINISHED =============================================")
        time_mult_derivative = [mesure_zernike_polynomial_mult(Npoints, n, True) for n in Norder]
        print("============================================= time_mult_derivative FINISHED =============================================")
        time_commun = [mesure_zernike_polynomial_commun(Npoints, n, False) for n in Norder]
        print("============================================= time_commun FINISHED =============================================")
        time_commun_derivative = [mesure_zernike_polynomial_commun(Npoints, n, True ) for n in Norder]
        print("============================================= time_commun_derivative FINISHED =============================================")
        time_up_to_order = [mesure_zernike_polynomial_up_to_order(Npoints, n, False) for n in Norder]
        print("============================================= time_up_to_order FINISHED =============================================")
        time_up_to_order_derivative = [mesure_zernike_polynomial_up_to_order(Npoints, n, True) for n in Norder] 
        print("============================================= time_up_to_order_derivative FINISHED =============================================")
        time_xy = [mesure_zernike_polynomial_xy(Npoints, n, False) for n in Norder]
        print("============================================= time_xy FINISHED =============================================")
        time_xy_derivative = [mesure_zernike_polynomial_xy(Npoints, n, True) for n in Norder]
        print("============================================= time_xy_derivative FINISHED =============================================")
        times_xy_up_to_order = [mesure_zernike_polynomial_xy_up_to_order(Npoints, n, False) for n in Norder]
        print("============================================= time_xy_up_to_order FINISHED =============================================")
        times_xy_up_to_order_derivative = [mesure_zernike_polynomial_xy_up_to_order(Npoints, n, True) for n in Norder]
        print("============================================= time_xy_up_to_order_derivative FINISHED =============================================")


    if exist_csv:
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)

            Norder = []
            time_old = []
            time_old_derivative = []
            time_single = []
            time_single_derivative = []
            time_mult = []
            time_mult_derivative = []
            time_commun = []
            time_commun_derivative = []
            time_up_to_order = []
            time_up_to_order_derivative = []
            time_xy = []
            time_xy_derivative = []
            times_xy_up_to_order = []
            times_xy_up_to_order_derivative = []
            for row in reader:
                Norder.append(int(row[0]))
                time_old.append(float(row[1]))
                time_old_derivative.append(float(row[2]))
                time_single.append(float(row[3]))
                time_single_derivative.append(float(row[4]))
                time_mult.append(float(row[5]))
                time_mult_derivative.append(float(row[6]))
                time_commun.append(float(row[7]))
                time_commun_derivative.append(float(row[8]))
                time_up_to_order.append(float(row[9]))
                time_up_to_order_derivative.append(float(row[10]))
                time_xy.append(float(row[11]))
                time_xy_derivative.append(float(row[12]))
                times_xy_up_to_order.append(float(row[13]))
                times_xy_up_to_order_derivative.append(float(row[14]))

    # Save the results to a CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Norder', 'Old Zernike', 'Old Zernike Derivative', 'Single Zernike', 'Single Zernike Derivative',
                        'Mult Zernike', 'Mult Zernike Derivative', 'Commun Zernike', 'Commun Zernike Derivative',
                        'Up to Order Zernike', 'Up to Order Zernike Derivative', 'XY Zernike', 'XY Zernike Derivative',
                        'XY Up to Order Zernike', 'XY Up to Order Zernike Derivative'])
        for i in range(len(Norder)):
            writer.writerow([Norder[i], time_old[i], time_old_derivative[i], time_single[i], time_single_derivative[i],
                            time_mult[i], time_mult_derivative[i], time_commun[i], time_commun_derivative[i],
                            time_up_to_order[i], time_up_to_order_derivative[i], time_xy[i], time_xy_derivative[i],
                            times_xy_up_to_order[i], times_xy_up_to_order_derivative[i]])

    # Plot the results
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(121)
    ax_derivative = fig.add_subplot(122)
    ax.plot(Norder, time_old, label='Old Zernike', color='blue', marker='o')
    ax_derivative.plot(Norder, time_old_derivative, label='Old Zernike Derivative', color='blue', linestyle='--', marker='x')
    ax.plot(Norder, time_single, label='Single Zernike', color='orange', marker='o')
    ax_derivative.plot(Norder, time_single_derivative, label='Single Zernike Derivative', color='orange', linestyle='--', marker='x')
    ax.plot(Norder, time_mult, label='Mult Zernike', color='green', marker='o')
    ax_derivative.plot(Norder, time_mult_derivative, label='Mult Zernike Derivative', color='green', linestyle='--', marker='x')
    ax.plot(Norder, time_commun, label='Commun Zernike', color='red', marker='o')
    ax_derivative.plot(Norder, time_commun_derivative, label='Commun Zernike Derivative', color='red', linestyle='--', marker='x')
    ax.plot(Norder, time_up_to_order, label='Up to Order Zernike', color='purple', marker='o')
    ax_derivative.plot(Norder, time_up_to_order_derivative, label='Up to Order Zernike Derivative', color='purple', linestyle='--', marker='x')
    ax.plot(Norder, time_xy, label='XY Zernike', color='brown', marker='o')
    ax_derivative.plot(Norder, time_xy_derivative, label='XY Zernike Derivative', color='brown', linestyle='--', marker='x')
    ax.plot(Norder, times_xy_up_to_order, label='XY Up to Order Zernike', color='cyan', marker='o')
    ax_derivative.plot(Norder, times_xy_up_to_order_derivative, label='XY Up to Order Zernike Derivative', color='cyan', linestyle='--', marker='x')
    ax.plot(Norder, times_xy_up_to_order, label='XY Up to Order Zernike', color='magenta', marker='o')
    ax_derivative.plot(Norder, times_xy_up_to_order_derivative, label='XY Up to Order Zernike Derivative', color='magenta', linestyle='--', marker='x')

    ax.set_xlabel('Maximal Order (Norder)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'Comparison of Zernike Polynomial Computation Times for {Npoints} Points.\nAll polynomials with n <= Norder are computed.')
    ax.legend()
    ax.set_xticks(Norder)
    y_min, y_max = ax.get_ylim()
    ax.set_yticks(numpy.arange(0, y_max + 0.5, 0.5))  # Intervalle d'0.5 seconde
    ax.set_yticklabels([f'{y:.2f}' for y in numpy.arange(0, y_max + 0.5, 0.5)])
    ax.grid(True)

    ax_derivative.set_xlabel('Maximal Order (Norder)')
    ax_derivative.set_ylabel('Time (seconds)')
    ax_derivative.set_title(f'Comparison of Zernike Polynomial + First Derivative Computation Times for {Npoints} Points.\nAll polynomials with n <= Norder are computed.')
    ax_derivative.legend()
    ax_derivative.set_xticks(Norder)
    y_min, y_max = ax_derivative.get_ylim()
    ax_derivative.set_yticks(numpy.arange(0, y_max + 1, 1))  # Intervalle d'1 seconde
    ax_derivative.set_yticklabels([f'{y:.2f}' for y in numpy.arange(0, y_max + 1, 1)])
    ax_derivative.grid(True)

    plt.tight_layout()
    plt.show()