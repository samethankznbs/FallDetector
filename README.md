# Ultralytics YOLOv8 İle Özel Bir Veri Kümesi Eğiterek Nesne Algılama

Gerçek zamanlı nesne algılama ve görüntü bölümleme modelinin en son sürümü olan Ultralytics YOLOv8 ile derin öğrenme ve bilgisayarlı görü alanında hız ve doğruluk açısından iyi bir performans alabiliriz.

Bu dökümanda Yolov8 ile kendi veri kümemizi eğiterek nesne algıladığımız bir projeyi adım adım yapacağız.

#### Özel Veri Kümesi Oluşturma
Veri kümeniz ne kadar çeşitli ve büyük olursa eğittiğiniz model o kadar verimli çalışır. 
Veri kümesi oluştururken kaynak olarak ;
https://storage.googleapis.com/openimages/web/index.html gibi siteler kullanılabilir ya da spesifik olarak algılamak istediğiniz nesnenin yer aldığı farklı videolardaki frameler işinize  yarayabilir. 
###### (Örneğin köpek tespiti için yapacağım modelde kullandığım video https://www.youtube.com/watch?v=TJQ93FJWIPY)
###### (Videolardaki framleri fotoğraf olarak indirebilmek için https://www.youtube.com/watch?v=6dLFVXiM4QA)

Özel Veri Kümenizi oluşturabilmek için indirdiğiniz fotoğraflardaki algılamak istediğiniz nesneleri işaretlemeniz gereklidir. Bu işleme Data Annotation adı verilir . Bu işlemi yapmaya yarayan bir çok araç vardır. Örneğin ; labelImg vb.
Fakat benim bu projede kullandığım araç CVAT oldu . (Sitenin linki : https://www.cvat.ai/)
##### CVAT ile Data Annotation

![](https://hackmd.io/_uploads/rkFS9NPC3.png)
1.Start Using CVAT ile giriş yapın.

![](https://hackmd.io/_uploads/H1ZsqVwCn.png)
2.Create a new project ile projemizi oluşturuyoruz

![](https://hackmd.io/_uploads/HkuZo4D0n.png)
3.Projemizin adını giriyoruz

![](https://hackmd.io/_uploads/HJ1VsNwAh.png)
4.Add label butonuyla alglayacağımız nesne/nesneleri ekleyip Continue butonuna basıyoruz.

![](https://hackmd.io/_uploads/ryEKs4PAh.png)
5.Panelde girdiğimiz label çeşitleri gözüküyor. İstersek add label ile ekleme yapabiliriz.Eğer labellarla işimiz bittiyse sağ alttaki + butonuna basıp create a new task butonuna basıp taskimizi oluşturuyoruz.

![](https://hackmd.io/_uploads/HyVG3VPA3.png)
6.Gelen ekranda taskin adını giriyoruz ve veri kümesinde kullanacağımız fotoğrafları yüklüyoruz. Submit & Open butonuna basıp taskimizi açıyoruz.

![](https://hackmd.io/_uploads/H1Ss3Vw02.png)
7.Gelen ekranda Jobs bölümünden ilgili projemizin üstüne tıklıyoruz.

![](https://hackmd.io/_uploads/BkCkR4vC2.png)
8.Sol taraftaki panelin label kısmından ilgili nesnenin labelını seçip nesnelerimizi teker teker seçiyoruz. Üst panelden her bir fotoğraf için seçimlerimizi yapıyoruz.(N Tuşuyla hızlı label oluşturabilirsiniz. Ctrl+S komutuyla sık sık projenizi kaydediniz.)

![](https://hackmd.io/_uploads/SyYiAVDA3.png)
9.Label seçme bittikten sonra üsteki panelin task kısmından projenizin Actions kısmından Export task dataset butonuna basıyoruz.

![](https://hackmd.io/_uploads/SyzC1SPAn.png)
10.Gelen ekranda Export format seçeneğinden YOLO 1.1 seçeneğini seçip OK butonuna basıp datasetimizi indiriyoruz.

CVAT  aracını kullanarak indirdiğimiz klasörün içindeki obj_train_data klasöründeki .txt uzantılı dosyalar içerisinde ilgili foroğrafların bilgilerini tutuyor.

Örnek bir .txt uzantılı dosya

![](https://hackmd.io/_uploads/By5ZwrvCn.png)

Dosyadaki her bir satır ilgili fotoğraftaki labelın değerlerini tutuyor.
İlk değer labelın türünü temsil ediyor.(Örneğin 0 köpeği temsil ediyor. Farklı label türleri eklersek bu değerler farklı farklı olabilir.)
Diğer değerler ise ilgili fotoğraftaki labelın konumunu ifade ediyor.

-------------------------------------------------------------

(https://drive.google.com/file/d/1EnPbZfQTsLPO_mYqVOM4hbF9IJSf8xcQ/view?usp=sharing)

Yukarda verdiğim klasör spesifik olarak eğitilecek datasetin klasörü.
###### (Kendi klasörünüzü oluşturabilirsiniz fakat programın label bulamama hatasını almamak için aşağıda gösterdiğim pathlere dikkat ediniz.)

CVAT  aracını kullanarak indirdiğimiz klasörün içindeki obj_train_data klasöründeki .txt uzantılı dosyaları

detector/code/data/labels/train 
detector/code/data/val/labels/train 

uzantılarına kopyalıyoruz.

----------------------------------------------------------

detector/code/data/images/train
detector/code/data/val/images/train

klasörlerine ise datasetimizde kullandığımız fotoğrafları kopyalıyoruz.

###### (Her bir .txt uzantılı dosya bir fotoğrafı temsil ediyor. Fotoğraflar ve .txt uzantılı dosyaların isimlerini değiştirmediğinizden emin olun)

Modelimiz şuanda eğitilmeye hazır.


-----------------------------------



#### Train YOLOv8(Model Eğitimi)
Öncelikle başlayabilmemiz için Ultralytics kurmamız gerekiyor. Komut istemi üzerinden aşğıdaki gibi kurabilirsiniz.

##### Ultralytics Kurulumu

```
pip install ultralytics
```


###### (Uyumluluk problemleri yaşamamak için Python sürümünüzün 3.8 ve üzeri olması gerekiyor. Eğer PyTorch kütüphanesi kullanılacaksa Pytorch sürümünün 1.8 ve üzeri olması gerekiyor .)
###### (Alternatif kurulumlar için : https://github.com/ultralytics/ultralytics )

Bir proje klasörü oluşturun.  İçerisinde .yaml uzantılı ve .py uzantılı iki dosya olmalı.

##### .yaml Uzantılı Dosya (config.yaml)

Bu dosya eğitilmeye hazır modelinizin dosya uzantılarını tutar. Bu dosyada algılayacağınız nesne/nesneler class olarak yer alır.


Örnek config.yaml dosyası
```
path: C:/Users/samethan/Desktop/detector/code/data
train: images/train
val:  val/images/train

names:
  0: dog
  
```
Bu klasörde path kısmı CVAT programıyla oluşturduğumuz klasörün data klasörüne kadarki yolunu gösterir. 
Train yolu ise data klasöründen sonraki, resimleri tuttuğumuz klasörün yolunu gösterir.
Val yolu ise data klasöründen sonraki,val klasöründe tuttuğumuz resimleri tutan klasörü gösterir.

###### (!)(Val klasöründeki resimlerle images klasöründeki resimler aynı fakat farklı klasörülerde yer alıyorlar. Eğer train yolu ile val yolu aynı olursa benim sistemimde labelları görmedi ve hata verdi. Farklı bilgisayarlarda çalışabilir. Ama farklı bir val klasörü açmanızı önerirm.)

##### .py Uzantılı Dosya (main.py)
Bu dosya eğitilmeye hazır modelinizin hangi modelde eğitileceğini ve eğitim sırasındaki önemli değişkenleri tutar.

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
model.train(data="config.yaml", epochs=100)  # train the model

```
Model türünü yüklediğimiz kısımda ben bu projede yolov8n kullandım. Farklı modeller ve özellikleri aşağıdaki gibidir.

![](https://hackmd.io/_uploads/Hk6yoU2R2.png)

Modeli kullandığımız ve eğittiğimiz kısımda ise train fonksiyonu bir çok parametreye sahiptir.Bu ayarlar modelin performansını, hızını ve doğruluğunu etkileyebilir.Belirli bir görev için mümkün olan en iyi performansı elde etmek amacıyla bu ayarları dikkatli bir şekilde ayarlamak ve denemek önemlidir.

Parametrelere aşağıdaki siteden daha detaylı ulaşabilirsiniz.

https://docs.ultralytics.com/modes/train/#arguments

###### data="config.yaml" kısmı sizin oluşturduğunuz .yaml uzantılı dosyayı belirtmelidir. Eğer aynı klasörde değillerse uzantısını yazabilirsiniz. 
###### "device" parametresi model eğitilirken GPU ,CPU arasında geçiş yapmanızı sağlar.MAC kullanıcısıysanız bu parametreyi kullanmanız gerekir.


##### Epochs Değeri 

Basit tabirle modelinizin tekrar tekrar eğitilme değeridir. Ne kadar büyük olursa o kadar verimli bir sonuç alabiliriz gibi bir genelleme yapmak da yanlış olur. Aşağıda epochs=10 ile eğitilmiş bir model ve epochs=100 ile eğitilmiş modelin sonuç farkları gösterilmiştir.

![](https://hackmd.io/_uploads/S1pG1whC2.png)

![](https://hackmd.io/_uploads/Hkaf1Pn0h.png)

Görüldüğü gibi epochs değeri arttıkça sonuçlar daha düzgün bir eğriye dönüyor. Belli bir değere kadar arttırmak en iyi sonucu elde etmemizi sağlayabilir.

Epochs değerinin gereğinden fazla olması eğittiğimiz modelin yanlış çalışmasına neden olur. Eğittiğimiz model bir süre sonra datasetimizde kullandığımız ve belirlediğimiz nesneler dışında nesneler bulamayabilir.Tabii ki epochs değeri datasetimizin büyüklüğü ile de alakalıdır.

###### (Örneğin bu projede benim kullandığım dataset çok büyük değildi(120 frame) bu yüzden epochs değerini çok arttırırsam bir süre sonra model yanlış çaışmaya başlayabilir.)

##### Eğitimin başlatılması

main.py dosyanızı çalıştırınız.
Datasetinizin büyüklüğü,cpu modeliniz, yolo model türünüz ve değişik parametreler(epochs vb.) eğitim hızınızı belirler. 
Path hatası almamak için özellikle config.yaml dosyasındaki uzantıların doğru olduğundan emin olunuz.
("C:\Users\samethan\AppData\Roaming\Ultralytics\settings.yaml" dosyasındaki datasets_dir pathinin doğru olduğundan emin olunuz.)
Eğitim bitikten sonra proje klasörünüzde \runs\detect\train klasörü oluşacak.
Bu klasörün içinde eğitimizin sonuçlarını görebilirsiniz.(Aynı modeli farklı parametrelerle birden çok eğitirseniz train klasörleri train1,train2 vb. gibi şekilde oluşur.)
\runs\detect\train\weights\best.pt dosyası modelinizi test ederken kullanacağınız dosyayı belirtir.

#### Modelinizi Test Etme

Modelinizi test etmeniz için birden çok yöntem vardır.
Modelinizi komut isteminden aşağıdaki gibi test edebilirsiniz.
```
yolo task=detect mode=predict model="runs/train/weights/best.pt" source="test.png"

or

yolo task=detect mode=predict model="runs/train/exp/weights/best.pt" source="test.mp4"
```
Modelinizi test etmenize yarayan Python kodu aşağıdaki gibidir.
(Bu kod test etmek istedğiniz videoyu nesne algılamaya çalışarak oluşturduğu labellarla tekrar kaydeder.)

predict.py
```
import os

from ultralytics import YOLO
import cv2


VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'dog.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
```
###### (Path hatası almamak için yüklediğiniz videonun  yolunu kodda değiştirin ya da benim yaptığım gibi proje dosyamızın içerisinde videos adında bir klasör oluşturun ve bu klasörün içine test etmek istediğiniz videoyu atın. predict.py çalıştıktan sonra en son oluşan videoyu da videos kalsörüne kaydeder.)

#### Projenizin Gelişmesi İçin Yapılabilecekler

*Data seti büyük tutulmalı ve model uzun süre eğitilmeli.
*PyTorch kütüphanesi kullanarak modelinizi GPU üzerinden eğitebilirsiniz. Böylece eğitim daha kısa sürer ve zamandan taasarruf edebilirsiniz.Gpu ile eğitebilmek için aşağıdaki siteden yardım alabilirsiniz.
https://thinkinfi.com/train-yolov8-on-custom-dataset-in-windows-gpu/
*Local olarak eğitimin dışında Google Colab sayesinde Google'ın ücretsiz olarak sunduğu GPU'ları kullanarak eğitimi hızlıca bitirebilirsiniz. Aşağıdan Colab üzerinden YOLOv8 eğitimine ulaşabilirsiniz.
https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb

















