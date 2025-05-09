import matplotlib.pyplot as plt

def plot_graph(x_start, x_end, y_start, y_end, points, rectangles):
    # Crea una nuova figura
    fig, ax = plt.subplots()

    # Imposta i limiti degli assi
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(y_start, y_end)

    # Assicura che le unità sui due assi abbiano la stessa lunghezza in pixel
    ax.set_aspect('equal', adjustable='box')

    # Traccia i punti con etichette
    for point in points:
        x, y, label = point
        ax.plot(x, y, 'o')  # Traccia il punto
        ax.text(x, y, f' {label}', verticalalignment='bottom', horizontalalignment='right')  # Aggiungi l'etichetta

    # Traccia i riquadri con etichette
    for rect in rectangles:
        x_center, y_center, width, height, label = rect
        # Calcola l'angolo in basso a sinistra del riquadro
        x_left = x_center - width / 2
        y_bottom = y_center - height / 2
        # Traccia il riquadro
        rect_patch = plt.Rectangle((x_left, y_bottom), width, height, linestyle='--', edgecolor='black', facecolor='none')
        ax.add_patch(rect_patch)
        # Aggiungi l'etichetta
        ax.text(x_center, y_center, f' {label}', verticalalignment='bottom', horizontalalignment='right')

    # Mostra il grafico
    plt.show()

# Esempio di utilizzo
# x_start = 84.09
# x_end = 584.09
# y_start = 3899.4
# y_end = 4399.4
x_start = 360
x_end = 500
y_start = 4200
y_end = 4320
points = [
    (418.6+20.7, 4280.7-6.9, '1744'), 
    (418.6+20.7, 4280.8-6.8, '1745,7,8'), 
    (418.6+20.7, 4280.9-6.8, '1749,50'),
    (418.6-5.0, 4280.9-5.0, '1751'),
    (418.6+20.0, 4280.9-6.8, '1752-6'),
    (418.6-5.0, 4280.9-5.4, '1757'),
    ]
rectangles = [
    (448.8, 4260.7, 80, 56, '1742'),
    (418.7, 4280.7, 80, 56, '1743')
    ]

plot_graph(x_start, x_end, y_start, y_end, points, rectangles)
