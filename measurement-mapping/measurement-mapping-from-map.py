import matplotlib.pyplot as plt
import numpy as np

def plot_graph_from_file(file_path, points, rectangles):
    # Carica i dati dal file, gestendo celle vuote e dati non numerici
    data = np.genfromtxt(file_path, delimiter='\t', filling_values=np.nan)
    
    # Estrai i valori di x e y
    x_values = data[1:, 0]
    y_values = data[0, 1:]
    
    # Estrai i valori di intensità
    intensity_values = data[1:, 1:].T
    
    # Crea una nuova figura
    fig, ax = plt.subplots()
    
    # Traccia i valori di intensità come sfondo sbiadito
    c = ax.pcolormesh(x_values, y_values, intensity_values, shading='auto', cmap='Greys_r', alpha=0.8)
    # fig.colorbar(c, ax=ax)
    
    # Assicura che le unità sui due assi abbiano la stessa lunghezza in pixel
    ax.set_aspect('equal', adjustable='box')
    
    # Traccia i punti con etichette
    for point in points:
        x, y, label = point
        # ax.plot(x, y, 'o')  # Traccia il punto
        ax.plot(x, y, 'o', label=label) # Traccia il punto e aggiungi l'etichetta alla legenda
        # ax.text(x, y, f' {label}', verticalalignment='bottom', horizontalalignment='left')  # Aggiungi l'etichetta
    
    # Traccia i riquadri con etichette
    for rect in rectangles:
        x_stage, y_stage, x_scan_exten, y_scan_exten, label = rect
        # Calcola l'angolo in basso a sinistra del riquadro
        x_left = x_stage + x_scan_exten[0]
        y_bottom = y_stage + y_scan_exten[0]
        width = x_scan_exten[1] - x_scan_exten[0]
        height = y_scan_exten[1] - y_scan_exten[0]
        # Traccia il riquadro
        rect_patch = plt.Rectangle((x_left, y_bottom), width, height, linestyle='--', edgecolor='black', facecolor='none')
        ax.add_patch(rect_patch)
        # Aggiungi l'etichetta
        ax.text(x_left, y_bottom, f' {label}', verticalalignment='bottom', horizontalalignment='left')
    
    # Aggiungi la legenda
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Mostra il grafico
    plt.show()

# Esempio di utilizzo
file_path = 'sample_overview_1740.txt'

# points = [
#     (418.6+20.7, 4280.7-6.9, '1744'), 
#     (418.6+20.7, 4280.8-6.8, '1745,7,8'), 
#     (418.6+20.7, 4280.9-6.8, '1749,50'),
#     (418.6-5.0, 4280.9-5.0, '1751'),
#     (418.6+20.0, 4280.9-6.8, '1752-6'),
#     (418.6-5.0, 4280.9-5.4, '1757'),
#     ]
points = [
    (418.6 +20.5, 4280.7 - 6.8, '1744-50,52-56,60'), 
    (418.6 - 5.0, 4280.9 - 5.2, '1751,57-59,61'),
    (418.6 - 5.0, 4280.9 +11.6, '1762,66'),
    (418.6 +20.3, 4280.9 -18.0, '1763,67,68'),
    (418.6 -33.0, 4280.9 +20.0, '1764,65')
    ]
rectangles = [
    (448.8, 4260.7, (-40,40), (-30,30), '1742'),
    (418.7, 4280.7, (-40,40), (-30,30), '1743'),
    (418.6, 4280.9, (- 5,20), (-18,11), '1769')
    ]

plot_graph_from_file(file_path, points, rectangles)
