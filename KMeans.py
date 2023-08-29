import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def randomkumemerkezisec(data, k):  #  veri setindeki noktalardan rastgele olarak k adet başlangıç merkezi seçer ve bu merkezleri bir numpy dizisinde saklar.
    centroids = []
    for i in range(k):
        centroid = random.choice(data)
        centroids.append(centroid)
    return np.array(centroids) 

def mesafehesapla(x,centroids):
    # veri noktalarının her biri ile mevcut merkezler arasındaki mesafeleri hesaplar. 
    # Her bir veri noktası, en yakın merkez ile ilişkilendirilir ve bir küme numarası atanır.
    clustered = []
    for i in x:
        distances = np.sqrt(np.sum((centroids - i) ** 2, axis=1))
        enyakin = np.argmin(distances)
        clustered.append(enyakin)
    return clustered

def kumemerkeziguncelle(x,kumeleme,k):
    #  fonksiyonu, her bir küme için yeni merkez noktası hesaplar. 
    #  Her kümeye dahil olan veri noktalarının koordinatları toplanarak ortalaması alınır ve bu ortalama noktası yeni bir merkez olarak atanır.
    newcentroids = []
    for i in range(k):
        adet = 0
        toplam = 0
        for j in range(len(x)):
            if(kumeleme[j] == i):
                adet += 1  
                toplam += x[j]
        newcentroids.append(toplam/adet)
    return newcentroids

def plot(x,kumeleme,merkezler,k):  # sonuçları görselleştirmek için kullanılır. Veri noktaları, küme merkezleri ve her küme için ayrı bir renk kullanarak grafik üzerinde gösterilir.
    renkler = ['g', 'b', 'c', 'm','r','b','y']
    for i in range(k):
        kumenoktalari = np.array([point for j, point in enumerate(x) if kumeleme[j] == i])
        plt.scatter(kumenoktalari[:, 0], kumenoktalari[:, 1], s=5, color=renkler[i])
        plt.scatter(merkezler[i, 0], merkezler[i, 1],  color='k',s=100, marker='*')    
    plt.show()


#---------------------------------------------------------------------------------------------------------

x , y = make_blobs(n_samples=350, n_features=2, centers=3,random_state=804) # üç farklı merkeze sahip yapay bir veri seti oluşturulur.
k = 7 # k değişkeni, k-means algoritması için seçilen küme sayısını belirtir.
maxiter = 100 #  algoritmanın maksimum iterasyon sayısını belirtir.
centroids = randomkumemerkezisec(x, k)


for i in range(maxiter):
    kumeleme = mesafehesapla(x, centroids)
    centroids = np.array(kumemerkeziguncelle(x, kumeleme, k))
    
# randomkumemerkezisec() fonksiyonu, veri setindeki noktalardan rastgele olarak k adet başlangıç merkezi seçer ve bu merkezleri bir numpy dizisinde saklar.

# Algoritma, maxiter sayısı kadar tekrar edilerek, kümeleme ve merkez noktası güncellemesi yapılır.


plot(x, kumeleme, centroids, k) # Sonuçlar plot() fonksiyonu kullanılarak görselleştirilir.
