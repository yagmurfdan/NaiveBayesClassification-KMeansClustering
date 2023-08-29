import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

veriset = pd.read_csv('verisetilast.csv')

x = veriset.iloc[:,:-1]  # tüm satırlar ve son sütun hariç tüm sütunlar x değişkenine atanır.
y = veriset.iloc[:,-1] #  tüm satırlar ve son sütun y değişkenine atanır.
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3,random_state=12)
# train_test_split() fonksiyonu ile "x" ve "y" değişkenleri test ve eğitim veri kümelerine ayrılır. test_size=0.3 ifadesi ile verilerin %30'u test setine ayrılır.
#  random_state=12 ifadesi ile rastgele bölme işlemi tekrarlandığında aynı sonuçların elde edilmesi sağlanır.
#  Ayrılmış eğitim ve test verileri xtrain, xtest, ytrain, ytest değişkenlerine atanır.

model = GaussianNB() # GaussianNB() sınıfından bir nesne oluşturularak "model" adında bir nesneye atanır. Bu, bir Gaussian Naive Bayes sınıflandırıcısıdır.
model.fit(xtrain,ytrain) # fit() yöntemi ile eğitim verileri ("xtrain" ve "ytrain") üzerinde model eğitilir.
ytahmin = model.predict(xtest) 
# predict() yöntemi kullanılarak test verileri ("xtest") üzerinde tahmin yapılır ve tahmin sonuçları "ytahmin" adlı bir değişkene atanır.


karmasiklikmatrix = confusion_matrix(ytest,ytahmin)
# confusion_matrix() fonksiyonu, gerçek hedef sınıfları ("ytest") ve model tahminlerini ("ytahmin") alır.
#  karmaşıklık matrisini hesaplar ve "karmasiklikmatrix" adlı bir değişkene atanır.
print(karmasiklikmatrix)