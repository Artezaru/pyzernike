from pyzernike import zernike_display

list_n = []
list_m = []
for n in range(0, 5):
    for m in range(-n, n + 1, 2):
        list_n.append(n)
        list_m.append(m)

zernike_display(n=list_n, m=list_m)